"""HUD event scan — PURE text/boost/crop logic + the fail-safe isolation. No ffmpeg,
no OCR install: frame source + recognizer are injected. The whole point is that this
booster NEVER blocks the robust audio candidate."""
import numpy as np
import pytest

from fullauto import hud


# ---- text -> canonical event -----------------------------------------------

def test_normalize_event_maps_banner_text():
    assert hud.normalize_event("PENTA KILL") == "pentakill"
    assert hud.normalize_event("Double Kill!") == "doublekill"
    assert hud.normalize_event("  ace  ") == "ace"
    assert hud.normalize_event("You have slain an enemy") == "kill"
    assert hud.normalize_event("random ui text") is None
    assert hud.normalize_event("") is None


def test_normalize_prefers_longer_phrase():
    # 'triple kill' must not be shadowed by the bare 'kill' substring
    assert hud.normalize_event("TRIPLE KILL") == "triplekill"


# ---- events -> boost --------------------------------------------------------

def test_hud_boost_sums_strongest_per_kind_and_caps():
    evs = [hud.HudEvent("doublekill"), hud.HudEvent("kill"), hud.HudEvent("kill")]
    # doublekill 0.4 + kill 0.25 (counted once, not twice) = 0.65
    assert abs(hud.hud_boost(evs) - 0.65) < 1e-9


def test_hud_boost_clamps_to_cap():
    evs = [hud.HudEvent("pentakill"), hud.HudEvent("ace"), hud.HudEvent("doublekill")]
    assert hud.hud_boost(evs, cap=1.5) == 1.5            # 1.0+0.7+0.4 -> capped
    assert hud.hud_boost([]) == 0.0


def test_ocr_available_uses_config_path_when_file_exists(tmp_path, monkeypatch):
    pytest.importorskip("pytesseract")
    fake = tmp_path / "tesseract.exe"
    fake.write_bytes(b"")                                  # a file that "exists"
    monkeypatch.setattr(hud.gconf, "TESSERACT_CMD", str(fake))
    hud._ocr_configured = False                            # force re-config
    assert hud.ocr_available() is True
    import pytesseract
    assert pytesseract.pytesseract.tesseract_cmd == str(fake)   # pointed at our binary


def test_ocr_available_falls_back_to_path_without_override(monkeypatch):
    pytest.importorskip("pytesseract")
    monkeypatch.setattr(hud.gconf, "TESSERACT_CMD", "Z:/nope/tesseract.exe")  # no file
    monkeypatch.setattr(hud.shutil, "which", lambda n: "/usr/bin/tesseract")  # on PATH
    hud._ocr_configured = False
    assert hud.ocr_available() is True                     # available via PATH


def test_ocr_unavailable_when_neither_resolves(monkeypatch):
    pytest.importorskip("pytesseract")
    monkeypatch.setattr(hud.gconf, "TESSERACT_CMD", "Z:/nope/tesseract.exe")
    monkeypatch.setattr(hud.shutil, "which", lambda n: None)  # not on PATH either
    hud._ocr_configured = False
    assert hud.ocr_available() is False
    msg = hud.ocr_unavailable_message()
    assert "TESSERACT_CMD" in msg and "PATH" in msg          # actionable


def test_multikill_streaks_collapse_escalating_banners_to_top_tier():
    # One penta fight shows Double->Triple->Quadra->Penta a few seconds apart -> ONE
    # streak reported as pentakill (not four candidates).
    evs = [hud.HudEvent("doublekill", 100.0), hud.HudEvent("triplekill", 103.0),
           hud.HudEvent("quadrakill", 106.0), hud.HudEvent("pentakill", 109.0)]
    streaks = hud.multikill_streaks(evs, gap=8.0, min_tier="triplekill")
    assert len(streaks) == 1
    assert streaks[0].tier == "pentakill"
    assert streaks[0].start == 100.0 and streaks[0].end == 109.0


def test_multikill_streaks_split_by_gap_and_filter_min_tier():
    evs = [hud.HudEvent("doublekill", 10.0),                 # lone double -> dropped
           hud.HudEvent("triplekill", 200.0),                # separate fight (gap) -> kept
           hud.HudEvent("doublekill", 203.0)]                # part of the triple's streak
    streaks = hud.multikill_streaks(evs, gap=8.0, min_tier="triplekill")
    assert [s.tier for s in streaks] == ["triplekill"]
    assert streaks[0].start == 200.0


def test_ace_times_dedup():
    evs = [hud.HudEvent("ace", 50.0), hud.HudEvent("ace", 51.0),   # same ace, dedup
           hud.HudEvent("ace", 300.0)]
    assert hud.ace_times(evs, gap=8.0) == [50.0, 300.0]


def test_scan_video_is_failsafe_and_uses_banner_roi():
    # injected frames + a recognizer that returns a penta banner -> a pentakill event;
    # a raising recognizer -> [] (fail-safe), like scan_window.
    frames = [(108.0, np.zeros((1080, 1920, 3), dtype=np.uint8))]
    evs = hud.scan_video("x.mp4", 120.0, enabled=True, frames=iter(frames),
                         recognizer=lambda c: "PENTA KILL")
    assert [e.kind for e in evs] == ["pentakill"]
    assert hud.scan_video("x.mp4", 120.0, enabled=False,
                          frames=iter(frames), recognizer=lambda c: "PENTA KILL") == []


def test_roi_crop_fractional():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = hud.roi_crop(frame, (0.5, 0.0, 0.5, 1.0))     # right half
    assert crop.shape == (100, 100, 3)


# ---- scan_window: isolation + injection ------------------------------------

def _one_frame():
    yield (1.0, np.zeros((1080, 1920, 3), dtype=np.uint8))


def test_scan_window_detects_with_injected_recognizer():
    evs = hud.scan_window("x.mp4", 0.0, 5.0, enabled=True,
                          rois={"banner": (0.3, 0.2, 0.4, 0.18)},
                          frames=_one_frame(), recognizer=lambda crop: "PENTAKILL")
    assert [e.kind for e in evs] == ["pentakill"]


def test_scan_window_disabled_returns_nothing():
    called = {"n": 0}

    def rec(crop):
        called["n"] += 1
        return "PENTAKILL"

    assert hud.scan_window("x.mp4", 0.0, 5.0, enabled=False,
                           frames=_one_frame(), recognizer=rec) == []
    assert called["n"] == 0                              # not even invoked


def test_scan_window_is_failsafe_when_recognizer_raises():
    def boom(crop):
        raise RuntimeError("no OCR backend")

    # a brittle HUD failure must NOT propagate — returns [] so the audio score stands
    assert hud.scan_window("x.mp4", 0.0, 5.0, enabled=True,
                           rois={"banner": (0.3, 0.2, 0.4, 0.18)},
                           frames=_one_frame(), recognizer=boom) == []


def test_scan_window_is_failsafe_when_frame_source_raises():
    def bad_frames():
        raise OSError("ffmpeg/decoder missing")
        yield  # pragma: no cover

    assert hud.scan_window("x.mp4", 0.0, 5.0, enabled=True,
                           frames=bad_frames(), recognizer=lambda c: "ace") == []
