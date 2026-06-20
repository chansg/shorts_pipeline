"""Audio-reaction detector — the robust core of full-auto highlight detection.

The user wants funny / rage moments, so the VOICE REACTION leads: a sudden "WHAT?!"
matters more than the mechanical play. This detector finds those without a transcript
or a model, on the raw mixed mono track, and is built to survive hours of footage:

  1. Stream the audio through a VOCAL band-pass (~300-3400 Hz) so fights/music in the
     low/high bands count less than a human exclamation.
  2. Fold the streamed PCM to a small per-window RMS array — never hold the whole
     signal in memory (the analysis is O(windows), the decode is block-by-block).
  3. Score each window by SUDDENNESS above a ROLLING baseline: an onset (attack)
     term dominates a sustained-energy term, so a sharp exclamation outscores a long
     teamfight roar (gradual onset, and the baseline catches up to it).
  4. Peak-pick (above REACTION_THRESHOLD, spaced so one reaction = one peak), then
     frame GENEROUS candidate windows anchored BEFORE the spike (setup + payoff),
     merging overlaps so a long fight is one candidate, not five.

Everything below the ffmpeg seam is PURE and unit-tested on synthetic arrays — no
ffmpeg, no GPU. Thresholds live in gameplay/config.py; the run logs the score curve
because the first passes on real captures are a calibration exercise.
"""
from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass, field

import numpy as np

from gameplay import config as gconf


@dataclass
class ReactionWindow:
    """A generous candidate clip window from one or more merged reaction peaks."""
    start: float
    end: float
    audio_score: float                       # 0..1 strongest reaction in the window
    peaks: list[float] = field(default_factory=list)   # peak times (HUD-scan anchors)

    @property
    def duration(self) -> float:
        return self.end - self.start


# ============================================================================
# streaming audio -> per-window RMS  (the only ffmpeg-touching code)
# ============================================================================

def _band_af(band) -> str:
    lo, hi = band
    return f"highpass=f={int(lo)},lowpass=f={int(hi)}"


def iter_pcm_blocks(stream, block_samples: int):
    """Yield normalised float64 PCM blocks (from s16le) off a binary stream. Reads a
    bounded `block_samples` at a time so memory never scales with clip length."""
    block_bytes = int(block_samples) * 2          # 2 bytes / int16 sample
    while True:
        buf = stream.read(block_bytes)
        if not buf:
            break
        yield np.frombuffer(buf, dtype=np.int16).astype(np.float64) / 32768.0


def fold_blocks_to_window_rms(block_iter, win: int) -> np.ndarray:
    """PURE. RMS per `win` samples across arbitrary-sized blocks, carrying a sub-window
    remainder so window edges need not align to block edges. Holds at most one block
    plus a <win remainder — this is what makes the analysis streaming, not full-decode.
    A trailing partial window is dropped."""
    win = int(win)
    out: list[float] = []
    ss = 0.0          # running sum of squares for the in-progress window
    cnt = 0
    for block in block_iter:
        b = np.asarray(block, dtype=np.float64)
        i = 0
        if cnt:                                    # finish the partial window first
            take = b[: win - cnt]
            ss += float(np.dot(take, take))
            cnt += take.size
            i = take.size
            if cnt == win:
                out.append(math.sqrt(ss / win))
                ss = 0.0
                cnt = 0
        rest = b[i:]
        nfull = rest.size // win
        if nfull:                                  # vectorise the whole windows
            chunk = rest[: nfull * win].reshape(nfull, win)
            out.extend(np.sqrt((chunk * chunk).mean(axis=1)).tolist())
            rest = rest[nfull * win:]
        if rest.size:                              # carry the remainder
            ss += float(np.dot(rest, rest))
            cnt += rest.size
    return np.asarray(out, dtype=np.float64)


def reaction_envelope(video, *, window_s: float | None = None, band=None,
                      sr: int | None = None, block_samples: int | None = None):
    """Stream the vocal-band-pass PCM of `video` and fold it to a per-window RMS array.
    Returns (times, rms). Empty arrays if the clip has no audio. Memory is bounded by
    `block_samples`, not by clip length."""
    window_s = gconf.REACTION_WINDOW_S if window_s is None else window_s
    band = gconf.REACTION_BAND_HZ if band is None else band
    sr = gconf.REACTION_SR if sr is None else sr
    block_samples = gconf.REACTION_BLOCK_SAMPLES if block_samples is None else block_samples
    win = max(1, int(round(sr * window_s)))
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "quiet", "-i", str(video), "-vn", "-ac", "1",
         "-ar", str(sr), "-af", _band_af(band), "-f", "s16le", "-"],
        stdout=subprocess.PIPE)
    try:
        rms = fold_blocks_to_window_rms(iter_pcm_blocks(proc.stdout, block_samples), win)
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()
    times = (np.arange(rms.size) + 0.5) * window_s
    return times, rms


# ============================================================================
# PURE scoring + peak picking + window framing
# ============================================================================

def _rolling_stats(x: np.ndarray, w: int):
    """Rolling median + std over a centred window of `w` samples (local baseline so a
    sudden clutch isn't normalised away by a loud whole-VOD)."""
    n = x.size
    half = max(1, w // 2)
    med = np.empty(n)
    std = np.empty(n)
    for i in range(n):
        seg = x[max(0, i - half):min(n, i + half + 1)]
        med[i] = np.median(seg)
        std[i] = seg.std()
    return med, std


def reaction_score(times, rms, *, window_s: float | None = None,
                   baseline_window_s: float | None = None,
                   onset_weight: float | None = None) -> np.ndarray:
    """PURE. Per-window 0..1 reaction score = onset-weighted SUDDENNESS above a rolling
    baseline. The onset term (positive rise vs the previous window) rewards a fast
    attack — a human exclamation — while the energy term rewards loudness above the
    local baseline. `onset_weight` blends them, then the curve is normalised to its
    max. A sustained teamfight swell scores LOW: its per-window onset is small and the
    rolling baseline climbs to meet it."""
    rms = np.asarray(rms, dtype=np.float64)
    if rms.size < 3:
        return np.zeros_like(rms)
    window_s = gconf.REACTION_WINDOW_S if window_s is None else window_s
    baseline_window_s = (gconf.REACTION_BASELINE_WINDOW_S if baseline_window_s is None
                         else baseline_window_s)
    onset_weight = gconf.REACTION_ONSET_WEIGHT if onset_weight is None else onset_weight
    w = max(3, int(round(baseline_window_s / max(window_s, 1e-6))))
    med, std = _rolling_stats(rms, w)
    eps = 1e-9
    energy = np.clip((rms - med) / (std + eps), 0.0, None)
    onset = np.clip(np.diff(rms, prepend=rms[:1]), 0.0, None) / (std + eps)
    raw = onset_weight * onset + (1.0 - onset_weight) * energy
    peak = float(raw.max())
    return raw / peak if peak > 0 else raw


def pick_reaction_peaks(times, score, *, threshold: float | None = None,
                        min_spacing_s: float | None = None,
                        max_peaks: int | None = None) -> list[tuple[float, float]]:
    """PURE. Local maxima of `score` at/above `threshold`, kept strongest-first with at
    least `min_spacing_s` between peaks (so one reaction yields one peak), capped to
    `max_peaks`. Returns [(time, score)] sorted by time."""
    times = np.asarray(times, dtype=float)
    score = np.asarray(score, dtype=float)
    threshold = gconf.REACTION_THRESHOLD if threshold is None else threshold
    min_spacing_s = gconf.REACTION_MIN_SPACING_S if min_spacing_s is None else min_spacing_s
    max_peaks = gconf.REACTION_MAX_PEAKS if max_peaks is None else max_peaks
    if score.size < 3:
        return []
    cand: list[tuple[float, float]] = []
    for i in range(1, score.size - 1):
        if (score[i] >= threshold and score[i] >= score[i - 1]
                and score[i] >= score[i + 1]):
            cand.append((float(times[i]), float(score[i])))
    kept: list[tuple[float, float]] = []
    for t, s in sorted(cand, key=lambda x: x[1], reverse=True):
        if all(abs(t - kt) >= min_spacing_s for kt, _ in kept):
            kept.append((t, s))
        if len(kept) >= max_peaks:
            break
    kept.sort(key=lambda x: x[0])
    return kept


def candidate_windows(peaks, total: float, *, pre_roll: float | None = None,
                      post_roll: float | None = None, merge_gap: float | None = None,
                      max_candidates: int | None = None) -> list[ReactionWindow]:
    """PURE. Frame each peak at `t` into a GENEROUS window [t - pre_roll, t + post_roll]
    — anchored BEFORE the spike so the setup is included — clamped to [0, total]. Merge
    windows whose gap is <= merge_gap into one candidate (carrying the max peak score
    and every peak time), then keep the highest-scoring `max_candidates`. Returned
    sorted by start."""
    pre_roll = gconf.PRE_ROLL_S if pre_roll is None else pre_roll
    post_roll = gconf.POST_ROLL_S if post_roll is None else post_roll
    merge_gap = gconf.MERGE_GAP_S if merge_gap is None else merge_gap
    max_candidates = gconf.MAX_CANDIDATES if max_candidates is None else max_candidates
    if not peaks:
        return []
    raw = []
    for t, s in sorted(peaks, key=lambda x: x[0]):
        start = max(0.0, t - pre_roll)
        end = min(total, t + post_roll) if total else t + post_roll
        raw.append(ReactionWindow(round(start, 2), round(end, 2), float(s), [round(t, 2)]))

    merged: list[ReactionWindow] = []
    for win in raw:
        if merged and win.start <= merged[-1].end + merge_gap:
            cur = merged[-1]
            cur.end = max(cur.end, win.end)
            cur.audio_score = max(cur.audio_score, win.audio_score)
            cur.peaks.extend(win.peaks)
        else:
            merged.append(win)

    merged.sort(key=lambda c: (-c.audio_score, c.start))
    kept = merged[:max_candidates]
    kept.sort(key=lambda c: c.start)
    return kept
