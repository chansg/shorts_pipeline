"""ASR-quality fixes that don't need a GPU: the 16k-mono audio prep and the
load-time option assembly. WhisperX/torch are never imported here (prepare_audio
shells out to ffmpeg), so these run on any machine with ffmpeg on PATH."""
import shutil
import subprocess

import pytest

from gameplay import transcribe as tx

_FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(not _FFMPEG, reason="ffmpeg not on PATH")


def _make_stereo_48k(path):
    """Synthesize a 1s 48kHz STEREO wav (stand-in for a game capture)."""
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", "sine=frequency=300:duration=1:sample_rate=48000",
         "-ac", "2", str(path)],
        check=True, capture_output=True,
    )


def test_prepare_audio_yields_16k_mono(tmp_path):
    src = tmp_path / "src.wav"
    _make_stereo_48k(src)
    # the source really is 48k stereo (the condition WhisperX mishandles)
    assert tx.probe_audio(src) == (48000, 2)

    out = tx.prepare_audio(src, tmp_path / "prepped.16k.wav")
    sr, ch = tx.probe_audio(out)
    assert sr == 16000, f"expected 16k, got {sr}"
    assert ch == 1, f"expected mono, got {ch} channels"


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
