"""Full-auto highlight CLIPS — orchestrate detection -> rank -> export -> manifest.

Long mp4 in; ranked, generous 9:16 raw candidate clips out, each dropped into the
manual-mode input queue (a GameplayClip the Gameplay tab can refine in one click),
plus a candidates.json the GUI lists with score + why-it-was-picked.

  audio reaction (fullauto.reaction, robust)  ->  candidate windows
        -> HUD scan per window (fullauto.hud, isolated booster, may be empty)
        -> final score = audio_score * (1 + hud_boost)  -> rank, cap
        -> cut + 9:16 reframe (shared gameplay.reframe / encode)  -> manifest

The reframe/encode is reused verbatim (one source of truth) — nothing here re-derives
cutting or scaling. Pure ranking + manifest round-trip are unit-tested without ffmpeg;
the HUD path is fail-safe so a brittle read never blocks a candidate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from gameplay import config as gconf
from gameplay.state import AutoSession, GameplayClip, slugify
from fullauto import hud as hud_mod
from fullauto import reaction as rx

Progress = Callable[[str], None]

NO_PEAKS_MSG = ("No reactions above threshold — nothing crossed REACTION_THRESHOLD "
                f"({gconf.REACTION_THRESHOLD}). Lower it and re-run, or check the clip "
                "has a voice track.")


@dataclass
class HighlightClip:
    start: float
    end: float
    audio_score: float
    hud_boost: float = 0.0
    score: float = 0.0
    hud_events: list[str] = field(default_factory=list)
    peaks: list[float] = field(default_factory=list)
    rank: int = 0
    clip_name: str = ""        # GameplayClip name — the manual-mode hand-off
    source_path: str = ""      # the cut segment (manual input source)
    clip_path: str = ""        # the generous raw 9:16 clip
    preview_path: str = ""     # thumbnail for the review list

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)

    @property
    def why(self) -> str:
        """One-line 'why it was picked' for the review list. The HUD event leads when
        present — the ARAM multikill is the headline; the voice reaction is the tiebreak."""
        bits = []
        if self.hud_events:
            bits.append("+".join(self.hud_events))
        bits.append(f"reaction {self.audio_score:.2f}")
        return ", ".join(bits)


# ---- detection + ranking (pure of ffmpeg except reaction_envelope) ----------

def _curve_summary(score: np.ndarray) -> str:
    if score.size == 0:
        return "empty"
    return (f"max={score.max():.2f} mean={score.mean():.2f} "
            f"p95={np.percentile(score, 95):.2f}")


def detect_windows(video, *, max_candidates: int | None = None,
                   progress: Progress | None = None):
    """Audio-reaction detection -> generous merged candidate windows. Returns
    (windows, total_seconds). Streams the audio (memory-bounded)."""
    from modules.assemble import _probe_duration
    emit = progress or (lambda m: None)
    total = float(_probe_duration(video) or 0.0)
    emit("Scoring audio reactions (vocal band-pass, onset-weighted, streamed)...")
    times, rms = rx.reaction_envelope(video)
    if rms.size == 0:
        emit("  no audio track — no candidates.")
        return [], total
    score = rx.reaction_score(times, rms)
    emit(f"  score curve: {_curve_summary(score)} | threshold "
         f"{gconf.REACTION_THRESHOLD}")
    peaks = rx.pick_reaction_peaks(times, score)
    emit(f"  {len(peaks)} reaction peak(s) above threshold.")
    windows = rx.candidate_windows(peaks, total, max_candidates=max_candidates)
    emit(f"  {len(windows)} candidate window(s) after merge + cap.")
    return windows, total


def rank_windows(video, windows, *, hud_enabled: bool | None = None,
                 progress: Progress | None = None) -> list[HighlightClip]:
    """Score each window (audio * (1 + hud_boost)), rank desc, assign ranks. The HUD
    scan is fail-safe (returns [] on any error) so it can only ever ADD score."""
    emit = progress or (lambda m: None)
    hud_enabled = gconf.HUD_SCAN_ENABLED if hud_enabled is None else hud_enabled
    clips: list[HighlightClip] = []
    for i, w in enumerate(windows):
        events = hud_mod.scan_window(video, w.start, w.end, enabled=hud_enabled)
        kinds = [e.kind for e in events]
        boost = hud_mod.hud_boost(events)
        if hud_enabled:
            emit(f"  window {i + 1}/{len(windows)} ({w.start:.0f}-{w.end:.0f}s): "
                 f"HUD {kinds or 'none'} (+{boost:.2f})")
        clips.append(HighlightClip(
            w.start, w.end, round(w.audio_score, 4), round(boost, 3),
            round(w.audio_score * (1.0 + boost), 4), kinds, list(w.peaks)))
    clips.sort(key=lambda c: (-c.score, c.start))
    for i, c in enumerate(clips):
        c.rank = i + 1
    return clips


# ---- ARAM (League): multikill-driven candidates -----------------------------

def _window_reaction(score, times, start: float, end: float) -> float:
    """Max audio-reaction score (0..1) inside [start, end], or 0.0."""
    if score is None or getattr(score, "size", 0) == 0:
        return 0.0
    mask = (times >= start) & (times <= end)
    return float(score[mask].max()) if mask.any() else 0.0


def aram_clips_from_anchors(anchors, total: float, score=None, times=None
                            ) -> list[HighlightClip]:
    """PURE (given the audio arrays). Turn (kind, start, end) multikill/ace anchors into
    ranked HighlightClips: a generous window anchored BEFORE the streak, scored by tier
    (penta > quadra > ace > triple), with the in-window voice reaction as the tiebreak.
    Ranked strongest tier first."""
    clips: list[HighlightClip] = []
    for kind, s, e in anchors:
        start = max(0.0, s - gconf.ARAM_PRE_ROLL_S)
        end = (min(total, e + gconf.ARAM_POST_ROLL_S) if total
               else e + gconf.ARAM_POST_ROLL_S)
        tier_w = float(gconf.HUD_EVENT_WEIGHTS.get(kind, 0.4))
        audio = _window_reaction(score, times, start, end)
        clips.append(HighlightClip(
            round(start, 2), round(end, 2), round(audio, 4), round(tier_w, 3),
            round(tier_w, 4), [kind], [round((s + e) / 2.0, 2)]))
    # strict tier order; louder reaction breaks ties; then earliest
    clips.sort(key=lambda c: (-c.score, -c.audio_score, c.start))
    for i, c in enumerate(clips):
        c.rank = i + 1
    return clips


def detect_aram_candidates(video, total: float, *, hud_enabled: bool | None = None,
                           progress: Progress | None = None) -> list[HighlightClip]:
    """ARAM detection: scan the WHOLE clip for multikill / ace banners, collapse each
    fight's escalating banners to its top tier, and anchor a candidate on every streak
    at/above ARAM_MIN_MULTIKILL (+ Aces). The voice reaction in each window is a
    tiebreak. Needs HUD (OCR); returns [] (with a logged reason) when unavailable."""
    emit = progress or (lambda m: None)
    hud_enabled = gconf.HUD_SCAN_ENABLED if hud_enabled is None else hud_enabled
    if not hud_enabled:
        emit("  ARAM mode needs the HUD scan enabled (it reads the multikill banner).")
        return []
    emit("Scanning the whole clip for multikill / ace banners (ARAM)...")
    events = hud_mod.scan_video(video, total, enabled=hud_enabled)
    if not events:
        emit("  no banners detected — install OCR (pytesseract) and check the banner "
             "ROI matches your resolution, or use Generic mode.")
        return []
    streaks = hud_mod.multikill_streaks(events)
    aces = hud_mod.ace_times(events) if gconf.ARAM_INCLUDE_ACE else []
    emit(f"  {len(streaks)} multikill streak(s) >= {gconf.ARAM_MIN_MULTIKILL}"
         + (f" + {len(aces)} ace(s)" if aces else "")
         + f"; tiers: {[s.tier for s in streaks] or 'none'}.")
    anchors = [(s.tier, s.start, s.end) for s in streaks] + [("ace", t, t) for t in aces]
    if not anchors:
        emit("  banners found but none reached the minimum tier "
             f"({gconf.ARAM_MIN_MULTIKILL}); lower ARAM_MIN_MULTIKILL to include them.")
        return []
    times, rms = rx.reaction_envelope(video)
    score = rx.reaction_score(times, rms) if rms.size else None
    return aram_clips_from_anchors(anchors, total, score=score, times=times)


# ---- export (reuses the shared cut + reframe + encode) ----------------------

def export_clip(video, session: AutoSession, clip: HighlightClip,
                index: int) -> HighlightClip:
    """Cut [start,end] into a GameplayClip source (the manual input) and reframe it to
    a generous raw 9:16 clip — reusing fullauto.export.cut_segment + gameplay.reframe
    (the same quality path as manual). No captions/effects/overlay (added in manual).
    Also writes a preview thumbnail. Mutates + returns `clip` with the paths."""
    from fullauto.export import cut_segment
    from fullauto.pipeline import make_preview
    from gameplay import reframe as reframe_mod

    name = slugify(f"{session.name}_{int(round(clip.start))}s")
    gp = GameplayClip(name)
    source = gp.dir / "source.mp4"
    if not source.exists():
        cut_segment(video, clip.start, clip.end, source)
    raw916 = gp.dir / "raw916.mp4"
    reframe_mod.reframe(source, raw916)        # gameplay default layout; idempotent
    preview = session.preview_path(index)
    try:
        make_preview(video, clip, preview)
    except Exception:        # noqa: BLE001 — a missing thumb shouldn't sink the export
        preview = Path("")

    clip.clip_name = name
    clip.source_path = str(source)
    clip.clip_path = str(raw916)
    clip.preview_path = str(preview) if preview else ""
    return clip


# ---- manifest (candidates.json) ---------------------------------------------

def clip_to_dict(c: HighlightClip) -> dict:
    return {"rank": c.rank, "score": c.score, "start": c.start, "end": c.end,
            "duration": c.duration, "audio_score": c.audio_score,
            "hud_boost": c.hud_boost, "hud_events": c.hud_events, "peaks": c.peaks,
            "clip_name": c.clip_name, "source_path": c.source_path,
            "clip_path": c.clip_path, "preview_path": c.preview_path, "why": c.why}


def clip_from_dict(d: dict) -> HighlightClip:
    c = HighlightClip(
        float(d["start"]), float(d["end"]), float(d.get("audio_score", 0.0)),
        float(d.get("hud_boost", 0.0)), float(d.get("score", 0.0)),
        list(d.get("hud_events", [])), list(d.get("peaks", [])),
        int(d.get("rank", 0)), d.get("clip_name", ""), d.get("source_path", ""),
        d.get("clip_path", ""), d.get("preview_path", ""))
    return c


def write_manifest(session: AutoSession, clips: list[HighlightClip]) -> Path:
    session.candidates_path.write_text(
        json.dumps([clip_to_dict(c) for c in clips], indent=2), encoding="utf-8")
    return session.candidates_path


def load_manifest(session: AutoSession) -> list[HighlightClip]:
    if not session.has_candidates():
        return []
    data = json.loads(session.candidates_path.read_text(encoding="utf-8"))
    return [clip_from_dict(d) for d in data]


# ---- full run ---------------------------------------------------------------

def run_highlight_detection(video, *, mode: str | None = None,
                            hud_enabled: bool | None = None,
                            max_candidates: int | None = None,
                            progress: Progress | None = None
                            ) -> tuple[list[HighlightClip], AutoSession]:
    """Long mp4 -> ranked, exported 9:16 raw candidate clips + candidates.json.

    `mode="generic"` (default): audio-reaction-led detection, HUD as a booster.
    `mode="aram"` (League ARAM): MULTIKILL-led — candidates are the triple+/penta/ace
    streaks ONLY (this game mode's money shots), the voice reaction a tiebreak.
    Friendly (not a crash) when nothing is found."""
    from orchestrator.errors import FriendlyError, ensure_ffmpeg
    from modules.assemble import _probe_duration
    ensure_ffmpeg()
    video = Path(video)
    if not video.exists():
        raise FriendlyError(f"Video not found: {video}")
    emit = progress or (lambda m: None)
    mode = (gconf.GAME_MODE if mode is None else mode).lower()
    session = AutoSession(video.stem)

    if mode == "aram":
        total = float(_probe_duration(video) or 0.0)
        clips = detect_aram_candidates(video, total, hud_enabled=hud_enabled,
                                       progress=progress)
        if max_candidates:
            clips = clips[:int(max_candidates)]
        if not clips:
            emit("No ARAM multikill candidates — see the log above. (ARAM mode only "
                 "surfaces multikill/ace moments; use Generic mode for reaction clips.)")
            write_manifest(session, [])
            return [], session
    else:
        windows, _total = detect_windows(video, max_candidates=max_candidates,
                                         progress=progress)
        if not windows:
            emit(NO_PEAKS_MSG)
            write_manifest(session, [])
            return [], session
        clips = rank_windows(video, windows, hud_enabled=hud_enabled, progress=progress)

    emit(f"Exporting {len(clips)} candidate(s) — cut + 9:16 reframe (shared encode)...")
    for c in clips:
        try:
            export_clip(video, session, c, c.rank - 1)
        except Exception as e:        # noqa: BLE001 — one bad cut shouldn't sink the run
            emit(f"  export failed for rank {c.rank}: {type(e).__name__}")
    write_manifest(session, clips)
    emit(f"Done — {len(clips)} candidate(s) ready to refine in manual mode.")
    return clips, session
