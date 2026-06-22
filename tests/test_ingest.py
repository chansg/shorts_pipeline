"""Ingest hardening — strip non-A/V (stray data/timecode tracks) + normalise to
CFR. ffmpeg-only, no GPU. The first real clip carried a stray timecode track."""
import json
import shutil
import subprocess

import pytest

from gameplay.transcribe import (needs_normalization, normalize_source,
                                 _is_faststart, playable_preview)

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="ffmpeg/ffprobe not on PATH")


def _streams(path):
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True, text=True).stdout
    return json.loads(out)["streams"]


@pytest.fixture
def clean_clip(tmp_path):
    p = tmp_path / "clean.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=1",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         str(p)], check=True, capture_output=True)
    return p


@pytest.fixture
def clip_with_timecode(tmp_path):
    # A .mov with a timecode track -> a stray "data" stream (codec_type=data).
    p = tmp_path / "tc.mov"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=1",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-timecode", "00:00:00:00",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         str(p)], check=True, capture_output=True)
    return p


@pytest.fixture
def non_faststart_clip(tmp_path):
    # default mp4 mux puts moov at the END (no faststart) -> browser "not playable".
    p = tmp_path / "slow.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(p)], check=True, capture_output=True)
    return p


def test_faststart_detection(clean_clip, non_faststart_clip, tmp_path):
    fast = tmp_path / "fast.mp4"
    subprocess.run(["ffmpeg", "-y", "-i", str(non_faststart_clip), "-c", "copy",
                    "-movflags", "+faststart", str(fast)], check=True, capture_output=True)
    assert _is_faststart(fast) is True
    assert _is_faststart(non_faststart_clip) is False


def test_playable_preview_remuxes_non_faststart_keeping_name(non_faststart_clip):
    out = playable_preview(str(non_faststart_clip))
    assert out != str(non_faststart_clip)          # a new (remuxed) file
    from pathlib import Path
    assert Path(out).name == non_faststart_clip.name   # original filename preserved
    assert _is_faststart(out) is True                  # now browser-playable


def test_playable_preview_passthrough_and_failsafe(clean_clip, tmp_path):
    fast = tmp_path / "fast.mp4"
    subprocess.run(["ffmpeg", "-y", "-i", str(clean_clip), "-c", "copy",
                    "-movflags", "+faststart", str(fast)], check=True, capture_output=True)
    assert playable_preview(str(fast)) == str(fast)    # already faststart -> unchanged
    assert playable_preview(None) is None               # no upload -> no-op
    assert playable_preview("nope.mp4") == "nope.mp4"  # bad path -> fail-safe passthrough


def test_clean_clip_needs_no_normalization(clean_clip):
    assert needs_normalization(clean_clip) is False


def test_timecode_track_triggers_normalization(clip_with_timecode):
    types = {s["codec_type"] for s in _streams(clip_with_timecode)}
    assert "data" in types                       # sanity: the stray track exists
    assert needs_normalization(clip_with_timecode) is True


def test_normalize_strips_non_av_and_is_cfr(clip_with_timecode, tmp_path):
    out = normalize_source(clip_with_timecode, tmp_path / "norm.mp4")
    streams = _streams(out)
    types = [s["codec_type"] for s in streams]
    assert "data" not in types                   # stray track dropped
    assert types.count("video") == 1 and types.count("audio") == 1
    vid = next(s for s in streams if s["codec_type"] == "video")
    assert vid["avg_frame_rate"] == vid["r_frame_rate"]   # constant frame rate
