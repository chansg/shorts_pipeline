"""Clean-voice transcription: track-aware extraction (OBS Track 2 = isolated voice),
single-track fallback, fail-safe, the speaker debug sidecar, and the SPEAKER_STYLE_MAP
caption-colour seeding. The content test (which physical track got extracted) needs
ffmpeg; the rest are pure/no-GPU."""
import shutil
import subprocess
import wave

import numpy as np
import pytest

from gameplay import config as gconf
from gameplay import transcribe as tx
from gameplay.state import GameplayClip
from gameplay.transcript import Transcript, Word

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


# ---- helpers ---------------------------------------------------------------

def _two_track_clip(path, f0=440, f1=880):
    """A 1s clip: video + a:0 = sine f0 (the 'mix') + a:1 = sine f1 (the 'voice')."""
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", "testsrc=size=320x240:rate=15:duration=1",
         "-f", "lavfi", "-i", f"sine=frequency={f0}:duration=1",
         "-f", "lavfi", "-i", f"sine=frequency={f1}:duration=1",
         "-map", "0:v", "-map", "1:a", "-map", "2:a",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         str(path)], check=True, capture_output=True)
    return path


def _one_track_clip(path, f0=440):
    subprocess.run(
        ["ffmpeg", "-y",
         "-f", "lavfi", "-i", "testsrc=size=320x240:rate=15:duration=1",
         "-f", "lavfi", "-i", f"sine=frequency={f0}:duration=1",
         "-map", "0:v", "-map", "1:a",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         str(path)], check=True, capture_output=True)
    return path


def _dominant_hz(wav_path):
    with wave.open(str(wav_path), "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(float)
    spec = np.abs(np.fft.rfft(data * np.hanning(len(data))))
    return float(np.fft.rfftfreq(len(data), 1 / sr)[int(np.argmax(spec))])


# ---- track counting + which-track extraction (ffmpeg) ----------------------

@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
def test_count_audio_tracks(tmp_path):
    assert tx.count_audio_tracks(_two_track_clip(tmp_path / "two.mp4")) == 2
    assert tx.count_audio_tracks(_one_track_clip(tmp_path / "one.mp4")) == 1


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
def test_prepare_audio_extracts_the_requested_track(tmp_path):
    # a:0 = 440 Hz, a:1 = 880 Hz. prepare_audio(track=1) must yield the 880 Hz voice.
    src = _two_track_clip(tmp_path / "two.mp4", f0=440, f1=880)
    a0 = tx.prepare_audio(src, tmp_path / "a0.wav", track=0)
    a1 = tx.prepare_audio(src, tmp_path / "a1.wav", track=1)
    assert abs(_dominant_hz(a0) - 440) < 40
    assert abs(_dominant_hz(a1) - 880) < 40           # the isolated voice track


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
def test_extract_voice_track_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(gconf, "GAMEPLAY_DIR", tmp_path)
    src = _two_track_clip(tmp_path / "two.mp4", f0=440, f1=880)
    clip = GameplayClip("vt")
    log = []
    voice = tx._extract_voice_track(src, clip, log.append)
    assert voice is not None and voice.exists()
    assert abs(_dominant_hz(voice) - 880) < 40        # extracted the voice, not the mix
    assert any("ISOLATED" in m or "Track 2" in m for m in log)


# ---- track selection logic (no ffmpeg; monkeypatched probe/extract) --------

def _clip(tmp_path, monkeypatch, name="c"):
    monkeypatch.setattr(gconf, "GAMEPLAY_DIR", tmp_path)
    clip = GameplayClip(name)
    (clip.dir / "source.mp4").write_bytes(b"")
    return clip


def test_single_track_falls_back_with_warning(tmp_path, monkeypatch):
    clip = _clip(tmp_path, monkeypatch)
    monkeypatch.setattr(tx, "count_audio_tracks", lambda s: 1)
    log = []
    assert tx._extract_voice_track("x.mp4", clip, log.append) is None
    assert not (clip.dir / tx.VOICE_AUDIO_NAME).exists()
    assert any("Single audio track" in m for m in log)


def test_two_track_picks_voice_index(tmp_path, monkeypatch):
    clip = _clip(tmp_path, monkeypatch)
    monkeypatch.setattr(tx, "count_audio_tracks", lambda s: 2)
    seen = {}

    def fake_prep(src, dest, track=None):
        seen["track"] = track
        dest.write_bytes(b"wav")
        return dest

    monkeypatch.setattr(tx, "prepare_audio", fake_prep)
    out = tx._extract_voice_track("x.mp4", clip, lambda m: None)
    assert out == clip.dir / tx.VOICE_AUDIO_NAME and out.exists()
    assert seen["track"] == gconf.VOICE_TRACK_INDEX == 1


def test_extraction_error_is_failsafe(tmp_path, monkeypatch):
    clip = _clip(tmp_path, monkeypatch)
    monkeypatch.setattr(tx, "count_audio_tracks", lambda s: 2)

    def boom(*a, **k):
        raise RuntimeError("ffmpeg blew up")

    monkeypatch.setattr(tx, "prepare_audio", boom)
    log = []
    assert tx._extract_voice_track("x.mp4", clip, log.append) is None   # no crash
    assert not (clip.dir / tx.VOICE_AUDIO_NAME).exists()
    assert any("failed" in m.lower() for m in log)


def test_disabled_flag_skips_extraction(tmp_path, monkeypatch):
    clip = _clip(tmp_path, monkeypatch)
    monkeypatch.setattr(gconf, "TRANSCRIBE_VOICE_TRACK", False)
    monkeypatch.setattr(tx, "count_audio_tracks",
                        lambda s: (_ for _ in ()).throw(AssertionError("probed!")))
    assert tx._extract_voice_track("x.mp4", clip, lambda m: None) is None


def test_transcribe_clip_feeds_voice_track_when_present(tmp_path, monkeypatch):
    clip = _clip(tmp_path, monkeypatch)
    (clip.dir / tx.VOICE_AUDIO_NAME).write_bytes(b"wav")        # pretend import made it
    monkeypatch.setattr(gconf, "KEEP_PREP_AUDIO", True)         # keep so we can assert
    captured = {}

    def fake_transcribe(src, **kw):
        captured.update(kw)
        return Transcript([Word("hi", 0.0, 0.4)])

    monkeypatch.setattr(tx, "transcribe", fake_transcribe)
    tx.transcribe_clip(clip, force=True)
    assert captured["voice_audio"] == clip.dir / tx.VOICE_AUDIO_NAME


def test_transcribe_clip_passes_none_without_voice(tmp_path, monkeypatch):
    clip = _clip(tmp_path, monkeypatch)
    captured = {}
    monkeypatch.setattr(tx, "transcribe",
                        lambda src, **kw: captured.update(kw) or Transcript([Word("a", 0, 1)]))
    tx.transcribe_clip(clip, force=True)
    assert captured["voice_audio"] is None


# ---- speaker debug sidecar -------------------------------------------------

def test_speaker_samples_longest_utterance_per_label():
    t = Transcript([
        Word("hey", 0.0, 0.3, "SPEAKER_00"),
        Word("there", 0.3, 0.6, "SPEAKER_00"),
        Word("yo", 1.0, 1.3, "SPEAKER_01"),
        Word("what", 1.3, 1.6, "SPEAKER_01"),
        Word("up", 1.6, 1.9, "SPEAKER_01"),
        Word("ok", 2.0, 2.3, "SPEAKER_00"),       # shorter later run, shouldn't win
    ])
    s = tx.speaker_samples(t)
    assert s["SPEAKER_00"] == "hey there"
    assert s["SPEAKER_01"] == "yo what up"


def test_write_speaker_sidecar_lists_labels(tmp_path, monkeypatch):
    import json
    clip = _clip(tmp_path, monkeypatch)
    t = Transcript([Word("hi", 0, 0.4, "SPEAKER_00"), Word("yo", 0.5, 0.9, "SPEAKER_01")])
    log = []
    out = tx.write_speaker_sidecar(clip, t, log.append)
    assert out and out.exists()
    data = json.loads(out.read_text("utf-8"))
    assert set(data) == {"SPEAKER_00", "SPEAKER_01"}
    assert any("Speakers detected" in m for m in log)


def test_write_speaker_sidecar_skips_single_speaker(tmp_path, monkeypatch):
    clip = _clip(tmp_path, monkeypatch)
    t = Transcript([Word("hi", 0, 0.4)], single_speaker=True)
    assert tx.write_speaker_sidecar(clip, t, lambda m: None) is None
    assert not (clip.dir / "speakers.json").exists()


# ---- caption-colour seeding from SPEAKER_STYLE_MAP -------------------------

def test_speaker_style_map_seeds_caption_colour(monkeypatch):
    from gameplay import gui as gui_mod
    monkeypatch.setattr(gconf, "SPEAKER_STYLE_MAP", {"SPEAKER_00": "#123456"})
    rows = gui_mod._speaker_rows(
        Transcript([Word("hi", 0, 0.4, "SPEAKER_00"), Word("yo", 0.5, 0.9, "SPEAKER_01")]))
    by = {r[0]: r[1] for r in rows}
    assert by["SPEAKER_00"] == "#123456"               # mapped label -> configured colour
    assert by["SPEAKER_01"] != "#123456"               # unmapped -> auto palette


# ---- faststart remux keeps every audio track -------------------------------

def test_playable_preview_keeps_all_audio_tracks(tmp_path, monkeypatch):
    import modules.assemble as assemble
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"not really mp4")
    monkeypatch.setattr(tx, "_is_faststart", lambda p: False)   # force the remux path
    cmds = []
    monkeypatch.setattr(assemble, "_run", lambda cmd, cwd=None: cmds.append(cmd))
    tx.playable_preview(str(src))
    assert cmds, "expected a remux ffmpeg call"
    cmd = cmds[0]
    # all audio streams mapped (default selection would drop the voice track a:1)
    assert "-map" in cmd and "0:a?" in cmd and "-c" in cmd and "copy" in cmd
