"""Full-auto review-first flow: candidate (de)serialization, transcript slicing
for handoff, and the candidate→manual load. No GPU/LLM; ffmpeg only for cut/load."""
import shutil
import subprocess

import pytest

from gameplay import autopipeline as ap
from gameplay import config as gconf
from gameplay.autohighlight import Candidate
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


# ---- ffmpeg-backed (no GPU): detect persistence isn't tested here, but the
# load_candidate cut + slice + manual-input population are. ----

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="ffmpeg/ffprobe not on PATH")


@pytest.fixture
def long_clip(tmp_path):
    p = tmp_path / "vod.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=30:duration=6",
         "-f", "lavfi", "-i", "sine=frequency=500:duration=6",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         str(p)], check=True, capture_output=True)
    return p


def test_load_candidate_cuts_and_slices(long_clip, tmp_path, monkeypatch):
    monkeypatch.setattr(gconf, "GAMEPLAY_DIR", tmp_path)
    # rebuild a session + a long transcript on the patched dir
    session = AutoSession("vod")
    long_t = Transcript([Word("aim", 1.0, 1.4, "S0"), Word("fire", 3.0, 3.4, "S1"),
                         Word("reload", 5.0, 5.4, "S0")])
    long_t.save(session.transcript_path)
    cand = Candidate(2.0, 4.0, "clutch", "nice", 1.0, "energy")

    clip, sub = ap.load_candidate(session, long_clip, cand)
    assert clip.has_source()
    assert clip.transcript_path.exists()
    # only the word inside [2,4] survives, rebased to t=0
    assert [w.text for w in sub.words] == ["fire"]
    assert sub.words[0].start == 1.0


def test_make_preview_thumbnail(long_clip, tmp_path):
    out = ap.make_preview(long_clip, Candidate(1.0, 3.0, "funny", ""),
                          tmp_path / "thumb.jpg")
    assert out.exists() and out.stat().st_size > 0
