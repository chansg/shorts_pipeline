"""ffmpeg-only smoke test for the gameplay manual backend: blur-pad reframe,
effects (punch-zoom + shake), caption burn, and overlay composite all run on a
tiny synthetic 16:9 clip and produce a valid 1080x1920 video. No GPU required."""
import json
import shutil
import subprocess

import pytest

from gameplay import config as gconf
from gameplay import effects as fx_mod
from gameplay import overlay as ov_mod
from gameplay import reframe as reframe_mod
from gameplay.manual import burn_captions, write_captions, ManualOptions
from gameplay.transcript import Transcript, Word

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="ffmpeg/ffprobe not on PATH")

PLACEHOLDER = gconf.OVERLAYS_DIR / "like_subscribe_placeholder.png"


def _dims(path):
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "json", str(path)],
        capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    return s["width"], s["height"]


@pytest.fixture
def synthetic_clip(tmp_path):
    """A 2s 1280x720 test clip with a beeping sine track (gives the energy pass
    something to find)."""
    clip = tmp_path / "src.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=30:duration=2",
         "-f", "lavfi", "-i", "sine=frequency=600:duration=2",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         str(clip)], check=True, capture_output=True)
    return clip


def test_reframe_to_9x16(synthetic_clip, tmp_path):
    out = reframe_mod.reframe(synthetic_clip, tmp_path / "reframed.mp4")
    assert out.exists()
    assert _dims(out) == (gconf.WIDTH, gconf.HEIGHT)


def test_tall_reframe_filter_full_width_band_no_stretch():
    g = reframe_mod.reframe_filter("tall", 1080, 1920, tall_frac=0.8, blur=24, fps=30)
    # uniform cover-crop (force_original_aspect_ratio=increase) -> never an anamorphic
    # stretch; a full-width band shorter than the frame, over a blurred fill.
    assert "force_original_aspect_ratio=increase" in g and "boxblur" in g
    assert "overlay=" in g
    bh = (int(round(1920 * 0.8)) // 2) * 2
    assert f"crop=1080:{bh}:" in g and bh < 1920          # tall band + thin blur frame
    # a stretch would be `scale=1080:1920` WITHOUT force_original_aspect_ratio
    assert "scale=1080:1920," not in g.replace(
        "scale=1080:1920:force_original_aspect_ratio=increase", "")


def test_tall_reframe_renders_9x16(synthetic_clip, tmp_path):
    out = reframe_mod.reframe(synthetic_clip, tmp_path / "tall.mp4", mode="tall")
    assert _dims(out) == (gconf.WIDTH, gconf.HEIGHT)


def test_effects_filter_builds_and_renders(synthetic_clip, tmp_path):
    reframed = reframe_mod.reframe(synthetic_clip, tmp_path / "reframed.mp4")
    # explicit beats so the test doesn't depend on the energy detector's findings
    beats = [0.5, 1.2]
    out, used = fx_mod.apply_effects(reframed, tmp_path / "fx.mp4",
                                     ["punch_zoom", "shake"], beats=beats)
    assert out.exists()
    assert _dims(out) == (gconf.WIDTH, gconf.HEIGHT)
    assert used == beats


def test_effects_noop_when_disabled(synthetic_clip, tmp_path):
    reframed = reframe_mod.reframe(synthetic_clip, tmp_path / "reframed.mp4")
    out, _ = fx_mod.apply_effects(reframed, tmp_path / "fx.mp4", [], beats=[])
    assert out.exists() and _dims(out) == (gconf.WIDTH, gconf.HEIGHT)


def test_energy_envelope_detects_signal(synthetic_clip):
    times, rms = fx_mod.energy_envelope(synthetic_clip)
    assert times.size > 0 and rms.size == times.size


def test_caption_burn(synthetic_clip, tmp_path):
    reframed = reframe_mod.reframe(synthetic_clip, tmp_path / "reframed.mp4")
    t = Transcript([Word("hello", 0.1, 0.6, "S0"), Word("world", 0.6, 1.2, "S1")])
    opts = ManualOptions()
    ass = write_captions(t, opts, tmp_path / "caps.ass")
    assert ass.exists() and ass.read_text(encoding="utf-8").count("Dialogue") == 2
    out = burn_captions(reframed, ass, tmp_path / "capped.mp4")
    assert out.exists() and _dims(out) == (gconf.WIDTH, gconf.HEIGHT)


def _frame_png(video, out, t):
    subprocess.run(["ffmpeg", "-y", "-ss", str(t), "-i", str(video),
                    "-vframes", "1", str(out)], check=True, capture_output=True)
    return out


def test_overlay_composite_changes_pixels(synthetic_clip, tmp_path):
    # Regression guard: a static-png overlay must PERSIST across its enable window
    # (looped input), not show only at t=0. Compare a frame mid-window against the
    # un-overlaid base and require a visible difference.
    from PIL import Image, ImageChops
    assert PLACEHOLDER.exists(), "placeholder overlay should be committed"
    reframed = reframe_mod.reframe(synthetic_clip, tmp_path / "reframed.mp4")
    out = ov_mod.composite(reframed, PLACEHOLDER.name, tmp_path / "ov.mp4",
                           position="bottom-center", start=0.0, duration=1.5)
    assert out.exists() and _dims(out) == (gconf.WIDTH, gconf.HEIGHT)
    base_f = Image.open(_frame_png(reframed, tmp_path / "b.png", 0.6)).convert("RGB")
    ov_f = Image.open(_frame_png(out, tmp_path / "o.png", 0.6)).convert("RGB")
    diff = ImageChops.difference(base_f, ov_f).getbbox()
    assert diff is not None, "overlay produced no visible change mid-window"


def test_overlay_missing_asset_raises(synthetic_clip, tmp_path):
    from orchestrator.errors import FriendlyError
    reframed = reframe_mod.reframe(synthetic_clip, tmp_path / "reframed.mp4")
    with pytest.raises(FriendlyError):
        ov_mod.composite(reframed, "does_not_exist.mov", tmp_path / "x.mp4")


def test_preview_captions_renders(synthetic_clip, tmp_path, monkeypatch):
    # manual.preview_captions: caption-only preview on the first N seconds, no GPU.
    import gameplay.config as gconf
    from gameplay import manual as manual_mod
    from gameplay.state import GameplayClip
    from gameplay.transcript import Transcript, Word
    monkeypatch.setattr(gconf, "GAMEPLAY_DIR", tmp_path)
    monkeypatch.setattr(manual_mod.gconf, "GAMEPLAY_DIR", tmp_path)
    clip = GameplayClip("prev")
    import shutil as _sh
    _sh.copy2(synthetic_clip, clip.dir / "source.mp4")
    t = Transcript([Word("hi", 0.1, 0.6, "S0"), Word("yo", 0.6, 1.2, "S1")])
    out = manual_mod.preview_captions(clip, t, manual_mod.ManualOptions(), seconds=1.5)
    assert out.exists() and _dims(out) == (gconf.WIDTH, gconf.HEIGHT)
