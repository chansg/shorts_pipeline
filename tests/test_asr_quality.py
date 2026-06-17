"""ASR-quality fixes that don't need a GPU: the 16k-mono audio prep, the option
assembly, and the work-audio naming invariant. WhisperX/torch are never imported
here (prepare_audio shells out to ffmpeg), so these run on any machine."""
import shutil
import subprocess

import pytest

from gameplay import transcribe as tx
from gameplay import config as gconf
from gameplay.state import GameplayClip

_FFMPEG = shutil.which("ffmpeg")
_needs_ffmpeg = pytest.mark.skipif(not _FFMPEG, reason="ffmpeg not on PATH")


def _make_stereo_48k(path):
    """Synthesize a 1s 48kHz STEREO wav (stand-in for a game capture)."""
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", "sine=frequency=300:duration=1:sample_rate=48000",
         "-ac", "2", str(path)],
        check=True, capture_output=True,
    )


@_needs_ffmpeg
def test_prepare_audio_yields_16k_mono(tmp_path):
    src = tmp_path / "src.wav"
    _make_stereo_48k(src)
    assert tx.probe_audio(src) == (48000, 2)   # the condition WhisperX mishandles

    out = tx.prepare_audio(src, tmp_path / tx.PREP_AUDIO_NAME)
    sr, ch = tx.probe_audio(out)
    assert sr == 16000, f"expected 16k, got {sr}"
    assert ch == 1, f"expected mono, got {ch} channels"


@_needs_ffmpeg
def test_prepare_audio_downmixes_without_loudnorm_or_highpass(tmp_path, monkeypatch):
    # Even with the filters disabled, the downmix to 16k mono must still happen.
    monkeypatch.setattr(tx.gconf, "WHISPERX_AUDIO_LOUDNORM", False)
    monkeypatch.setattr(tx.gconf, "WHISPERX_AUDIO_HIGHPASS_HZ", 0)
    assert tx._audio_filter_chain() == ""
    src = tmp_path / "src.wav"
    _make_stereo_48k(src)
    out = tx.prepare_audio(src, tmp_path / "out.wav")
    assert tx.probe_audio(out) == (16000, 1)


def test_probe_audio_missing_file_is_zero():
    assert tx.probe_audio("does_not_exist.wav") == (0, 0)


def test_prep_audio_name_does_not_shadow_source(tmp_path, monkeypatch):
    """Regression: the prepped 16k wav must NOT be mistaken for the clip's source
    video. Before the fix it was named source.16k.wav and sorted ahead of
    source.mp4 in source_path()'s glob, so reframe/build fed the WAV to a video
    filtergraph ("Stream specifier ':v' matches no streams")."""
    assert not tx.PREP_AUDIO_NAME.startswith("source.")

    monkeypatch.setattr(gconf, "GAMEPLAY_DIR", tmp_path)
    clip = GameplayClip("vid40")
    (clip.dir / "source.mp4").write_bytes(b"")          # the real video
    (clip.dir / tx.PREP_AUDIO_NAME).write_bytes(b"")    # our prepped work audio
    assert clip.source_path().name == "source.mp4"


# ---- VAD merge-window (chunk_size) — the continuous-speech dropout fix ------

class _StubModel:
    """Records the kwargs passed to .transcribe; optionally OOMs on the first call."""
    def __init__(self, oom_first=False):
        self.calls = []
        self._oom_first = oom_first

    def transcribe(self, audio, batch_size=None, chunk_size=None):
        self.calls.append((batch_size, chunk_size))
        if self._oom_first and len(self.calls) == 1:
            raise RuntimeError("CUDA failed with error out of memory")
        return {"segments": [], "language": "en"}


def test_transcribe_passes_chunk_size_to_model():
    # The dropout fix: a small VAD-merge window must reach model.transcribe, or dense
    # speech collapses to one giant window the model abandons after a few words.
    m = _StubModel()
    out = tx._transcribe_with_oom_retry(m, object(), 16, 4, None, 8)
    assert m.calls == [(16, 8)]                  # batch_size + chunk_size both passed
    assert out == {"segments": [], "language": "en"}


def test_transcribe_oom_retry_keeps_chunk_size():
    # The OOM retry drops batch_size but must KEEP the chunk_size window.
    m = _StubModel(oom_first=True)
    tx._transcribe_with_oom_retry(m, object(), 16, 4, None, 6)
    assert m.calls == [(16, 6), (4, 6)]


def test_chunk_size_default_is_small():
    # Guard the default away from WhisperX's 30s (which caused the 26s->5 words bug).
    assert 0 < gconf.WHISPERX_CHUNK_SIZE <= 15
