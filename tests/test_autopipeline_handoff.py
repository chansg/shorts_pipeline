"""Full-auto pipeline: candidate (de)serialization, transcript slicing, the review
selector, and the 16:9 YouTube export. No GPU/LLM; ffmpeg only for cut/export.

After full-auto was promoted to its own landing entry, its output is a 16:9 YouTube
video — it no longer hands cuts to the 9:16 Shorts/manual backend. These tests lock
that: the export is landscape/native, never reframed to 1080x1920."""
import shutil
import subprocess

import pytest

from fullauto import pipeline as ap
from fullauto import export as export_mod
from fullauto import gui as fa_gui
from fullauto.highlight import Candidate
from gameplay import config as gconf
from gameplay.state import AutoSession
from gameplay.transcript import Transcript, Word


def test_slice_transcript_rebases_and_clamps():
    t = Transcript([Word("before", 2.0, 2.5, "S0"),
                    Word("inside", 11.0, 11.5, "S0"),
                    Word("edge", 19.5, 20.5, "S1"),    # straddles end -> kept
                    Word("after", 30.0, 30.5, "S0")])
    sub = ap.slice_transcript(t, 10.0, 20.0)
    assert [w.text for w in sub.words] == ["inside", "edge"]
    assert sub.words[0].start == 1.0 and sub.words[0].start >= 0.0


def test_candidate_roundtrip():
    c = Candidate(12.0, 28.5, "funny", "he fell off", 1.7, "energy+llm")
    d = ap._cand_to_dict(c)
    c2 = ap._cand_from_dict(d)
    assert (c2.start, c2.end, c2.category, c2.caption, c2.source) == \
           (12.0, 28.5, "funny", "he fell off", "energy+llm")


def test_candidate_name_stable_and_safe():
    s = AutoSession("My Long VOD.mp4")
    c = Candidate(73.4, 88.0, "clutch", "x")
    name = ap.candidate_name(s, c)
    assert name == ap.candidate_name(s, c)          # deterministic
    assert " " not in name and name.endswith("clutch")


def test_selected_indices_parses_labels():
    # The review selector maps "N. ..." labels back to 0-based indices.
    assert fa_gui._selected_indices(["1. [clutch] ...", "3. [funny] ...", "bad"]) \
        == [0, 2]
    assert fa_gui._selected_indices([]) == []


# ---- ffmpeg-backed (no GPU): cut/preview/export ----------------------------

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="ffmpeg/ffprobe not on PATH")


def _dims(path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        capture_output=True, text=True).stdout.strip()
    w, h = out.split("x")
    return int(w), int(h)


@pytest.fixture
def long_clip(tmp_path):
    # 16:9 landscape source (640x360), like real YouTube/gameplay footage.
    p = tmp_path / "vod.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30:duration=6",
         "-f", "lavfi", "-i", "sine=frequency=500:duration=6",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         str(p)], check=True, capture_output=True)
    return p


def test_make_preview_thumbnail(long_clip, tmp_path):
    out = ap.make_preview(long_clip, Candidate(1.0, 3.0, "funny", ""),
                          tmp_path / "thumb.jpg")
    assert out.exists() and out.stat().st_size > 0


def test_export_youtube_is_landscape_native(long_clip, tmp_path):
    # Two highlights assembled into ONE 16:9 video at the source's native resolution
    # — NOT blur-padded to a 1080x1920 Short.
    cands = [Candidate(0.5, 2.0, "funny", "a"), Candidate(3.0, 5.0, "clutch", "b")]
    out = export_mod.export_youtube(long_clip, cands, tmp_path / "yt.mp4")
    assert out.exists() and out.stat().st_size > 0
    w, h = _dims(out)
    assert (w, h) == (640, 360)          # native landscape, unchanged
    assert w > h                          # 16:9, not vertical
    assert (w, h) != (gconf.WIDTH, gconf.HEIGHT)   # not the 9:16 Short size


def test_export_youtube_empty_raises(long_clip, tmp_path):
    with pytest.raises(ValueError):
        export_mod.export_youtube(long_clip, [], tmp_path / "none.mp4")


def test_export_youtube_single_segment(long_clip, tmp_path):
    out = export_mod.export_youtube(long_clip, [Candidate(1.0, 3.0, "story", "x")],
                                    tmp_path / "one.mp4")
    assert _dims(out) == (640, 360)


def test_manual_gameplay_page_has_no_fullauto():
    # The manual Gaming page must be full-auto-free after the move: no import of the
    # full-auto backend and no detect/handoff symbols. (The docstring may *point* to
    # fullauto/, so we check imports + wiring, not the bare word.)
    import inspect
    from gameplay import gui as gameplay_gui
    src = inspect.getsource(gameplay_gui)
    assert "autopipeline" not in src and "ap_mod" not in src   # backend not imported
    assert "import fullauto" not in src and "from fullauto" not in src
    for sym in ("ap_mod", "_do_detect", "_load_into_manual", "_batch_build",
                "_load_candidates_ui"):
        assert not hasattr(gameplay_gui, sym), f"{sym} should be gone from gameplay.gui"
