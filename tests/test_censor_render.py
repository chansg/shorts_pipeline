"""Censor render integration (ffmpeg only, no GPU): the audio is modified ONLY
within the hit window, and the caption burns the masked form."""
import shutil
import subprocess

import numpy as np
import pytest

from gameplay import censor

_FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(not _FFMPEG, reason="ffmpeg not on PATH")


def _clip_with_tone(path, dur=3.0):
    # 9:16 video + a steady 440 Hz tone so we can measure where the audio changed.
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc2=size=1080x1920:rate=30:duration={dur}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={dur}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
        check=True, capture_output=True)


def _rms(path, a, b):
    out = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-ss", f"{a}", "-to", f"{b}", "-i", str(path),
         "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        capture_output=True).stdout
    x = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt((x ** 2).mean())) if x.size else 0.0


def test_mute_changes_audio_only_in_window(tmp_path):
    src = tmp_path / "in.mp4"
    _clip_with_tone(src)
    out = tmp_path / "out.mp4"
    graph = censor.audio_graph([(1.0, 1.5)], dur=3.0, mode="mute")
    subprocess.run(["ffmpeg", "-y", "-i", str(src), "-filter_complex", graph,
                    "-map", "0:v", "-map", "[a]", "-c:v", "copy",
                    "-c:a", "aac", str(out)], check=True, capture_output=True)
    before = _rms(out, 0.2, 0.8)
    inside = _rms(out, 1.1, 1.4)
    after = _rms(out, 1.7, 2.8)
    assert inside < before * 0.1 and inside < after * 0.1   # silenced only in-window
    assert before > 0.05 and after > 0.05                    # untouched outside


def test_burn_censors_audio_and_masks_caption(tmp_path, monkeypatch):
    # Through the manual burn path: a censored word bleeps the audio AND the .ass
    # shows the masked caption — in one encode.
    from gameplay import config as gconf
    from gameplay.manual import ManualOptions, write_captions, burn_captions
    from gameplay.transcript import Transcript, Word
    monkeypatch.setattr(gconf, "GAMEPLAY_DIR", tmp_path)

    src = tmp_path / "reframed.mp4"
    _clip_with_tone(src)
    t = Transcript([Word("oh", 0.4, 0.8), Word("shit", 1.0, 1.5, censor=True)])
    opts = ManualOptions(censor="both")
    ass = write_captions(t, opts, tmp_path / "c.ass")
    text = ass.read_text(encoding="utf-8")
    assert "S***" in text.upper() and "SHIT" not in text.upper()   # caption masked

    spans = censor.merge_spans(t.censor_spans(), gconf.CENSOR_PAD_S, 3.0)
    graph = censor.audio_graph(spans, 3.0, mode="bleep")
    out = burn_captions(src, ass, tmp_path / "final.mp4", audio_graph=graph)
    # audio still present (bleep tone replaces the voice in-window, not silence)
    assert _rms(out, 1.1, 1.4) > 0.01
