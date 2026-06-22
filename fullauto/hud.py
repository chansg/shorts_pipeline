"""HUD event scan — the SECONDARY, score-boosting signal for full-auto detection.

A loud reaction is the trigger; a kill-feed / multikill banner near it explains and
scores the moment ("WHAT?!" + a Pentakill banner = a strong funny/rage clip). This is
League-UI-specific and brittle across resolutions, so it is built as a booster that is
ALLOWED TO FAIL: the whole scan is wrapped so any error (no OCR backend, an ffmpeg
hiccup, a weird ROI) yields no events and the audio candidate survives unchanged. A
master switch (HUD_SCAN_ENABLED) turns it off entirely.

It only ever samples frames INSIDE a candidate window (cheap), never the whole video.

The pure parts — text -> canonical event, events -> score boost, ROI crop — are
unit-tested. Frame sampling + OCR are injectable seams (`frames`, `recognizer`) so the
fail-safe and the scoring are testable without ffmpeg or an OCR install.
"""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from gameplay import config as gconf


@dataclass
class HudEvent:
    kind: str           # canonical kind, e.g. "pentakill" | "ace" | "kill" | "death"
    t: float = 0.0      # time (s) of the frame it was seen in
    text: str = ""      # the raw recognised text (debug)


@dataclass
class MultiKill:
    """One fight's multikill STREAK, reported at its top tier (Double..Penta)."""
    tier: str           # canonical kind, e.g. "pentakill"
    start: float        # time of the first banner of the streak
    end: float          # time of the last (top-tier) banner of the streak

    @property
    def t(self) -> float:
        return self.end


# ---- PURE: text -> event, events -> boost, ROI crop -------------------------

def normalize_event(text, lexicon=None) -> str | None:
    """PURE. Map recognised HUD text to a canonical event kind via substring match
    (case-insensitive, whitespace-collapsed), or None. Longest phrase wins so
    'triple kill' isn't shadowed by a bare 'kill'."""
    lexicon = gconf.HUD_EVENT_LEXICON if lexicon is None else lexicon
    low = " ".join(str(text or "").lower().split())
    if not low:
        return None
    for phrase in sorted(lexicon, key=len, reverse=True):
        if phrase in low:
            return lexicon[phrase]
    return None


def hud_boost(events, *, weights=None, cap=None) -> float:
    """PURE. Total score boost from HUD events: the strongest weight PER KIND summed
    (so a multi-frame banner counts once), clamped to `cap`. 0.0 for no events."""
    weights = gconf.HUD_EVENT_WEIGHTS if weights is None else weights
    cap = gconf.HUD_BOOST_CAP if cap is None else cap
    best: dict[str, float] = {}
    for e in events or []:
        kind = getattr(e, "kind", e)
        w = float(weights.get(kind, 0.0))
        best[kind] = max(best.get(kind, 0.0), w)
    return float(min(cap, sum(best.values())))


def multikill_streaks(events, *, tiers=None, gap=None, min_tier=None) -> list:
    """PURE. Collapse multikill banner events into per-fight STREAKS. Banners within
    `gap` seconds belong to one escalating fight (Double -> Triple -> ... -> Penta), so
    each streak is reported ONCE at its highest tier (no 4 candidates for one penta).
    Streaks whose top tier is below `min_tier` are dropped. Returns [MultiKill] by time."""
    tiers = gconf.ARAM_TIERS if tiers is None else tiers
    gap = gconf.ARAM_STREAK_GAP_S if gap is None else gap
    min_tier = gconf.ARAM_MIN_MULTIKILL if min_tier is None else min_tier
    rank = {k: i for i, k in enumerate(tiers)}
    mk = sorted((e for e in events if getattr(e, "kind", e) in rank),
                key=lambda e: e.t)
    streaks: list[dict] = []
    cur: dict | None = None
    for e in mk:
        if cur is not None and e.t - cur["last"] <= gap:
            cur["last"] = e.t
            if rank[e.kind] > rank[cur["tier"]]:
                cur["tier"] = e.kind
        else:
            if cur is not None:
                streaks.append(cur)
            cur = {"start": e.t, "last": e.t, "tier": e.kind}
    if cur is not None:
        streaks.append(cur)
    min_rank = rank.get(min_tier, 0)
    return [MultiKill(s["tier"], round(s["start"], 2), round(s["last"], 2))
            for s in streaks if rank[s["tier"]] >= min_rank]


def ace_times(events, *, gap=None) -> list[float]:
    """PURE. De-duplicated Ace banner times (an Ace persists several frames; collapse
    detections within `gap` to the first)."""
    gap = gconf.ARAM_STREAK_GAP_S if gap is None else gap
    out: list[float] = []
    for t in sorted(e.t for e in events if getattr(e, "kind", e) == "ace"):
        if not out or t - out[-1] > gap:
            out.append(round(t, 2))
    return out


def roi_crop(frame, roi):
    """PURE. Crop an (H, W, ...) array to the (x, y, w, h)-fraction ROI."""
    h, w = frame.shape[0], frame.shape[1]
    fx, fy, fw, fh = roi
    x0, y0 = int(fx * w), int(fy * h)
    x1, y1 = int((fx + fw) * w), int((fy + fh) * h)
    return frame[y0:y1, x0:x1]


# ---- brittle, isolated: frame sampling + OCR --------------------------------

def _default_recognizer(crop) -> str:
    """Try OCR on a ROI crop. Uses pytesseract if installed; otherwise raises so the
    caller's fail-safe kicks in (audio-only). Kept tiny + dependency-optional on
    purpose — HUD is a bonus, never a requirement."""
    import pytesseract                      # optional; ImportError -> handled upstream
    return pytesseract.image_to_string(crop)


def _sample_frames(video, start: float, end: float, sample_fps: float):
    """Yield (t, RGB ndarray) sampled at `sample_fps` within [start, end] only. Uses
    ffmpeg to write JPEGs to a temp dir (bounded: a window is seconds long), read via
    imageio. Any missing decoder dependency raises -> handled by scan_window."""
    import imageio.v3 as iio                # optional; ImportError -> handled upstream
    span = max(0.0, end - start)
    with tempfile.TemporaryDirectory() as td:
        patt = str(Path(td) / "f_%04d.jpg")
        subprocess.run(
            ["ffmpeg", "-v", "quiet", "-y", "-ss", f"{start:.3f}", "-t", f"{span:.3f}",
             "-i", str(video), "-vf", f"fps={sample_fps}", patt],
            check=True)
        for i, p in enumerate(sorted(Path(td).glob("f_*.jpg"))):
            yield (start + i / max(sample_fps, 1e-6), iio.imread(p))


def scan_window(video, start: float, end: float, *, enabled: bool | None = None,
                rois=None, sample_fps: float | None = None,
                frames=None, recognizer: Callable | None = None) -> list[HudEvent]:
    """Detect HUD events in [start, end]. FULLY ISOLATED: returns [] on the master
    switch being off OR on ANY failure (no OCR backend, ffmpeg/decoder error, bad
    frame) so a brittle HUD read can never block the robust audio candidate. `frames`
    and `recognizer` are injectable for testing without ffmpeg/OCR."""
    enabled = gconf.HUD_SCAN_ENABLED if enabled is None else enabled
    if not enabled:
        return []
    rois = gconf.HUD_ROIS if rois is None else rois
    sample_fps = gconf.HUD_SAMPLE_FPS if sample_fps is None else sample_fps
    recognizer = recognizer or _default_recognizer
    try:
        frame_iter = (frames if frames is not None
                      else _sample_frames(video, start, end, sample_fps))
        events: list[HudEvent] = []
        for t, frame in frame_iter:
            for roi in rois.values():
                kind = normalize_event(recognizer(roi_crop(frame, roi)))
                if kind:
                    events.append(HudEvent(kind, float(t)))
        return events
    except Exception:        # noqa: BLE001 — HUD is a fail-safe booster, never fatal
        return []


def scan_video(video, duration: float, *, sample_fps: float | None = None,
               roi=None, enabled: bool | None = None, frames=None,
               recognizer: Callable | None = None) -> list[HudEvent]:
    """Scan the WHOLE clip (sampled at `sample_fps`) for the centre multikill / ace
    banner — the ARAM money shots that don't always coincide with a loud reaction.
    Reuses scan_window's fail-safe machinery, restricted to the banner ROI. Returns []
    on the master switch being off or any failure (no OCR backend, etc.)."""
    sample_fps = gconf.ARAM_SCAN_FPS if sample_fps is None else sample_fps
    roi = gconf.HUD_ROIS.get("banner") if roi is None else roi
    return scan_window(video, 0.0, float(duration), enabled=enabled,
                       rois={"banner": roi}, sample_fps=sample_fps,
                       frames=frames, recognizer=recognizer)
