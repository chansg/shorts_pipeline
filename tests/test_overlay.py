"""Branded like/subscribe overlay placement (no GPU; ffmpeg cmd captured, not run)."""
import pytest

from gameplay import overlay as ov
from gameplay import config as gconf
from orchestrator.errors import FriendlyError


def _capture(tmp_path, monkeypatch):
    monkeypatch.setattr(gconf, "OVERLAYS_DIR", tmp_path)
    (tmp_path / "banner.png").write_bytes(b"x")          # asset must exist
    cmds = []
    monkeypatch.setattr(ov, "_run", lambda cmd, cwd=None: cmds.append(cmd))
    monkeypatch.setattr(ov, "has_alpha", lambda p: True)
    monkeypatch.setattr(ov, "_has_audio", lambda p: False)
    return cmds


def _fc(cmd):
    return cmd[cmd.index("-filter_complex") + 1]


def test_composite_scales_to_width_frac_and_centre_y(tmp_path, monkeypatch):
    cmds = _capture(tmp_path, monkeypatch)
    ov.composite(tmp_path / "base.mp4", "banner.png", tmp_path / "o.mp4",
                 start=0, duration=0)
    cmd = cmds[0]
    fc = _fc(cmd)
    ow = round(gconf.WIDTH * gconf.OVERLAY_WIDTH_FRAC)
    assert f"scale={ow}:-2" in fc                          # scaled to ~85% width, aspect kept
    assert f"H*{gconf.OVERLAY_POS_Y_FRAC:.4f}-h/2" in fc   # vertical CENTRE at pos_y_frac
    assert "(W-w)/2" in fc                                 # centred horizontally
    # ONE pass: base + asset are the only inputs, single libx264 encode
    assert cmd.count("-i") == 2 and "libx264" in cmd
    assert "-filter_complex" in cmd                        # not a second pass


def test_composite_knobs_move_and_resize(tmp_path, monkeypatch):
    cmds = _capture(tmp_path, monkeypatch)
    ov.composite(tmp_path / "base.mp4", "banner.png", tmp_path / "o.mp4",
                 start=0, duration=0, width_frac=0.5, pos_y_frac=0.5)
    fc = _fc(cmds[0])
    assert f"scale={round(gconf.WIDTH*0.5)}:-2" in fc      # 540px wide
    assert "H*0.5000-h/2" in fc                            # banner centre moved to mid-frame


def test_composite_timing_window_preserved(tmp_path, monkeypatch):
    cmds = _capture(tmp_path, monkeypatch)
    ov.composite(tmp_path / "base.mp4", "banner.png", tmp_path / "o.mp4",
                 start=1.0, duration=4.0)
    assert "enable='between(t,1.000,5.000)'" in _fc(cmds[0])


def test_composite_missing_asset_is_friendly(tmp_path, monkeypatch):
    monkeypatch.setattr(gconf, "OVERLAYS_DIR", tmp_path)
    monkeypatch.setattr(ov, "_run", lambda cmd, cwd=None: None)
    with pytest.raises(FriendlyError):
        ov.composite(tmp_path / "base.mp4", "missing.png", tmp_path / "o.mp4")


def test_default_overlay_asset_present():
    # the branded banner ships in overlays/ and is the configured default
    assert gconf.LIKE_SUB_OVERLAY == "like_subscribe_overlay.png"
    assert (gconf.OVERLAYS_DIR / gconf.LIKE_SUB_OVERLAY).exists()
    # captions sit above the banner band (no collision by default)
    assert gconf.CAPTION_POS_Y_FRAC < gconf.OVERLAY_POS_Y_FRAC
