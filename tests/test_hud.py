"""HUD event scan — PURE text/boost/crop logic + the fail-safe isolation. No ffmpeg,
no OCR install: frame source + recognizer are injected. The whole point is that this
booster NEVER blocks the robust audio candidate."""
import numpy as np

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
