"""Output-encoding quality: the shared encode helper, reframe layout modes, the
reduced encode count, and a real-ffmpeg bitrate check. Mostly no-GPU (the bitrate
test needs ffmpeg only)."""
import shutil
import subprocess

import pytest

from gameplay import encode as enc
from gameplay import config as gconf
from gameplay import reframe as reframe_mod
from gameplay import manual as manual_mod
from gameplay import overlay as overlay_mod
from gameplay.manual import ManualOptions, run_manual
from gameplay.state import GameplayClip
from gameplay.transcript import Transcript, Word


def _arg(args, flag):
    return args[args.index(flag) + 1]


# ---- encode helper ---------------------------------------------------------

def test_final_args_are_quality_targeted():
    a = enc.final_args()
    assert _arg(a, "-c:v") == "libx264"
    assert _arg(a, "-crf") == str(gconf.OUTPUT_CRF) == "18"
    assert _arg(a, "-preset") == "slow"
    assert _arg(a, "-profile:v") == "high"
    assert _arg(a, "-pix_fmt") == "yuv420p"
    assert _arg(a, "-movflags") == "+faststart"     # mobile streaming
    assert _arg(a, "-r") == str(gconf.FPS)


def test_intermediate_args_near_lossless_no_faststart():
    a = enc.intermediate_args()
    assert _arg(a, "-crf") == str(gconf.INTERMEDIATE_CRF) == "14"
    assert "-movflags" not in a and "-profile:v" not in a


def test_final_args_crf_override():
    assert _arg(enc.final_args(crf=16), "-crf") == "16"


# ---- reframe layout modes --------------------------------------------------

def test_reframe_filter_modes_and_default():
    blur = reframe_mod.reframe_filter("blur_pad", 1080, 1920)
    crop = reframe_mod.reframe_filter("fit_crop", 1080, 1920)
    zoom = reframe_mod.reframe_filter("zoom_blur", 1080, 1920)
    # blur-pad: split + blurred bars
    assert "split=2" in blur and "boxblur" in blur
    # fit_crop: fills + crops, no blur, no split
    assert "crop=1080:1920" in crop and "boxblur" not in crop and "split" not in crop
    # zoom_blur: blur-pad but the foreground enlarged by ZOOM_BLUR_SCALE
    assert "boxblur" in zoom and str(gconf.ZOOM_BLUR_SCALE) in zoom
    # default / unknown -> blur_pad
    assert reframe_mod.reframe_filter(None, 1080, 1920) == blur
    assert reframe_mod.reframe_filter("bogus", 1080, 1920) == blur


def test_default_reframe_mode_is_blur_pad():
    assert gconf.REFRAME_MODE == "blur_pad"


# ---- encode count (no GPU; record ffmpeg invocations) ----------------------

def _patch_runners(monkeypatch, tmp_path, cmds):
    monkeypatch.setattr(gconf, "GAMEPLAY_DIR", tmp_path)
    monkeypatch.setattr(manual_mod, "ensure_ffmpeg", lambda: None)
    for mod in (reframe_mod, manual_mod, overlay_mod):
        monkeypatch.setattr(mod, "_run", lambda cmd, cwd=None: cmds.append(cmd))
        monkeypatch.setattr(mod, "_has_audio", lambda p: False)


def _make_clip():
    clip = GameplayClip("q")
    (clip.dir / "source.mp4").write_bytes(b"")
    return clip


def test_run_manual_two_encodes_no_overlay(tmp_path, monkeypatch):
    cmds = []
    _patch_runners(monkeypatch, tmp_path, cmds)
    clip = _make_clip()
    t = Transcript([Word("hi", 0.0, 0.4), Word("yo", 0.4, 0.8)])
    list(run_manual(clip, t, ManualOptions(effects=[], overlay_name=None), force=True))
    encodes = [c for c in cmds if "libx264" in c]
    assert len(encodes) == 2                         # reframe (intermediate) + final
    finals = [c for c in encodes if "+faststart" in c]
    assert len(finals) == 1                          # exactly one quality-targeted final


def test_run_manual_three_encodes_with_overlay(tmp_path, monkeypatch):
    cmds = []
    _patch_runners(monkeypatch, tmp_path, cmds)
    monkeypatch.setattr(gconf, "OVERLAYS_DIR", tmp_path)
    monkeypatch.setattr(overlay_mod, "has_alpha", lambda p: True)
    (tmp_path / "ov.png").write_bytes(b"")
    clip = _make_clip()
    t = Transcript([Word("hi", 0.0, 0.4)])
    list(run_manual(clip, t, ManualOptions(overlay_name="ov.png"), force=True))
    encodes = [c for c in cmds if "libx264" in c]
    assert len(encodes) == 3                         # reframe + captions(intermediate) + overlay(final)
    finals = [c for c in encodes if "+faststart" in c]
    assert len(finals) == 1                          # only the overlay (last) pass is final


def test_run_manual_passes_reframe_mode(tmp_path, monkeypatch):
    cmds = []
    _patch_runners(monkeypatch, tmp_path, cmds)
    clip = _make_clip()
    t = Transcript([Word("hi", 0.0, 0.4)])
    list(run_manual(clip, t, ManualOptions(reframe_mode="fit_crop"), force=True))
    reframe_cmd = next(c for c in cmds if any("crop=1080:1920" in str(a) for a in c))
    assert not any("boxblur" in str(a) for a in reframe_cmd)   # fit_crop, not blur-pad


# ---- real bitrate (ffmpeg only, no GPU) ------------------------------------

@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                    reason="ffmpeg/ffprobe not on PATH")
def test_final_encode_bitrate_above_10mbps(tmp_path):
    out = tmp_path / "o.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", "testsrc2=size=1080x1920:rate=30:duration=3", *enc.final_args(), str(out)],
        check=True, capture_output=True)
    info = dict(
        line.split("=", 1) for line in subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=size,duration",
             "-of", "default=noprint_wrappers=1", str(out)],
            capture_output=True, text=True).stdout.splitlines() if "=" in line)
    mbps = float(info["size"]) * 8 / float(info["duration"]) / 1e6
    assert mbps > 10, f"expected >10 Mbps at CRF {gconf.OUTPUT_CRF}, got {mbps:.1f}"
