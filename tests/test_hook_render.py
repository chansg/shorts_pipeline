"""Narrated-hook render integration (ffmpeg only, no GPU, ElevenLabs mocked): the
game bed ducks under the narration during the hook and swells back after — through
the real burn_captions final-encode path."""
import shutil
import subprocess

import numpy as np
import pytest

from gameplay import hook
from gameplay.manual import burn_captions
from modules.karaoke_captions import build_ass

_FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(not (_FFMPEG and shutil.which("ffprobe")),
                                reason="ffmpeg not on PATH")


def _reframed_with_bed(path, dur=4.0):
    # 9:16 video + a steady STEREO bed so the duck ratio is clean.
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc2=size=1080x1920:rate=30:duration={dur}",
         "-f", "lavfi", "-i", f"sine=frequency=300:duration={dur}",
         "-ac", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(path)], check=True, capture_output=True)


def _wav(path, lavfi, dur):
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"{lavfi}:duration={dur}",
                    "-c:a", "pcm_s16le", str(path)], check=True, capture_output=True)


def _rms(path, a, b):
    out = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-ss", f"{a}", "-to", f"{b}", "-i", str(path),
         "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        capture_output=True).stdout
    x = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt((x ** 2).mean())) if x.size else 0.0


def _streams(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    return out.split()


def test_burn_ducks_bed_under_hook_in_one_pass(tmp_path):
    src = tmp_path / "reframed.mp4"
    _reframed_with_bed(src)
    silent = tmp_path / "silent.wav"      # isolate the bed-duck (no narration energy)
    _wav(silent, "anullsrc=r=24000:cl=mono", 1.2)
    ass = tmp_path / "c.ass"
    ass.write_text(build_ass([("hi", 0.0, 0.3)]), encoding="utf-8")

    graph = hook.duck_mix_graph("[0:a]", 1.2, duck=0.25, release=0.3)
    out = burn_captions(src, ass, tmp_path / "out.mp4",
                        audio_graph=graph, audio_inputs=["-i", str(silent)])

    assert "video" in _streams(out) and "audio" in _streams(out)   # one combined output
    inside = _rms(out, 0.2, 1.0)
    after = _rms(out, 2.0, 3.0)
    assert inside < after * 0.5, f"bed not ducked: in={inside:.3f} after={after:.3f}"


def test_narration_audible_over_the_dip(tmp_path):
    src = tmp_path / "reframed.mp4"
    _reframed_with_bed(src)
    silent = tmp_path / "silent.wav"
    _wav(silent, "anullsrc=r=24000:cl=mono", 1.2)
    narr = tmp_path / "narr.wav"
    _wav(narr, "sine=frequency=900", 1.2)
    ass = tmp_path / "c.ass"
    ass.write_text(build_ass([("hi", 0.0, 0.3)]), encoding="utf-8")
    graph = hook.duck_mix_graph("[0:a]", 1.2)

    ducked = burn_captions(src, ass, tmp_path / "ducked.mp4",
                           audio_graph=graph, audio_inputs=["-i", str(silent)])
    full = burn_captions(src, ass, tmp_path / "full.mp4",
                         audio_graph=graph, audio_inputs=["-i", str(narr)])
    # with the narration mixed in, the in-window level is clearly higher than the
    # ducked bed alone — the voice is audible over the dip.
    assert _rms(full, 0.2, 1.0) > _rms(ducked, 0.2, 1.0) * 1.5
