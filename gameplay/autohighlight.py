"""EXPERIMENTAL — game-agnostic highlight detection for full-auto mode.

A fused detector over a long, diarized gameplay transcript + its audio:

  A. cheap signals (no LLM): audio-energy peaks (locally-normalised, spaced) and a
     reaction-keyword scan (configurable per-category lexicon);
  B. window framing: expand each anchor to a clip window snapped to transcript
     word/sentence boundaries, clamped to a target length;
  C. an LLM judge over ~2-3 min overlapping transcript chunks returning strict JSON
     {start,end,category,reason,hook_caption,confidence} — catches moments energy /
     keywords miss (a deadpan funny line isn't loud);
  D. fuse + score + rank: a weighted sum of the four signals, deduped + capped.

Everything tunable lives in gameplay/config.py (the AUTO_* block) — the first runs
on real VODs are a calibration pass. The pure functions take arrays / transcripts
(not a GPU / a video file) so they unit-test without ffmpeg or a model. Scoring is
deterministic given the same inputs + config (no hidden randomness).

Failure-contained: a bad LLM chunk is skipped (not fatal); with the LLM backend
off or unavailable, the energy + reaction signals still produce candidates.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from typing import Callable

import numpy as np

from gameplay import config as gconf
from gameplay.effects import energy_envelope
from gameplay.transcript import Transcript

Progress = Callable[[str], None]


@dataclass
class Candidate:
    start: float
    end: float
    category: str          # one of gconf.AUTO_CATEGORIES, or "highlight"
    caption: str = ""
    score: float = 0.0
    source: str = "energy"  # which signals contributed, e.g. "llm+energy"
    reason: str = ""        # LLM rationale (debug / review aid)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Anchor:
    time: float
    kind: str                       # "energy" | "reaction"
    strength: float = 0.0           # energy: normalised prominence (0..1); reaction: hit count
    category_hint: str | None = None


@dataclass
class LLMMoment:
    start: float
    end: float
    category: str
    caption: str = ""
    reason: str = ""
    confidence: float = 0.0


# ============================================================================
# A. cheap signals
# ============================================================================

def _local_stats(rms: np.ndarray, w: int):
    """Rolling median + std with a centred window of `w` samples (local
    normalisation so a loud clutch isn't drowned out by a loud whole-VOD)."""
    n = len(rms)
    half = max(1, w // 2)
    med = np.empty(n)
    std = np.empty(n)
    for i in range(n):
        seg = rms[max(0, i - half):min(n, i + half + 1)]
        med[i] = np.median(seg)
        std[i] = seg.std()
    return med, std


def pick_energy_peaks(times, rms, *, k=None, roll_s=None, min_prominence=None,
                      min_spacing_s=None, window_s=None,
                      max_anchors=None) -> list[tuple[float, float]]:
    """PURE. Loud-moment peaks from a loudness envelope. Returns (time,
    prominence0..1) sorted by time. A peak is a local maximum above
    `rolling_median + k*rolling_std`; prominence (peak − local median) is
    normalised to the video's max; peaks below `min_prominence` or within
    `min_spacing_s` of a stronger peak are dropped."""
    rms = np.asarray(rms, dtype=float)
    times = np.asarray(times, dtype=float)
    if rms.size < 3:
        return []
    k = gconf.AUTO_ENERGY_K if k is None else k
    roll_s = gconf.AUTO_ENERGY_ROLL_S if roll_s is None else roll_s
    min_prom = gconf.AUTO_ENERGY_MIN_PROMINENCE if min_prominence is None else min_prominence
    min_spacing = gconf.AUTO_ENERGY_MIN_SPACING_S if min_spacing_s is None else min_spacing_s
    window_s = gconf.AUTO_ENERGY_WINDOW_S if window_s is None else window_s
    max_anchors = gconf.AUTO_ENERGY_MAX_ANCHORS if max_anchors is None else max_anchors

    w = max(3, int(round(roll_s / max(window_s, 1e-6))))
    med, std = _local_stats(rms, w)
    thresh = med + k * std

    raw: list[tuple[float, float]] = []   # (prominence, time)
    for i in range(1, len(rms) - 1):
        if rms[i] >= thresh[i] and rms[i] >= rms[i - 1] and rms[i] >= rms[i + 1]:
            prom = float(rms[i] - med[i])
            if prom > 0:
                raw.append((prom, float(times[i])))
    if not raw:
        return []
    maxprom = max(p for p, _ in raw)
    peaks = [(t, p / maxprom) for p, t in raw if (p / maxprom) >= min_prom]

    # greedy: keep strongest, enforce spacing, cap count
    kept: list[tuple[float, float]] = []
    for t, pn in sorted(peaks, key=lambda x: x[1], reverse=True):
        if all(abs(t - kt) >= min_spacing for kt, _ in kept):
            kept.append((t, pn))
        if len(kept) >= max_anchors:
            break
    kept.sort(key=lambda x: x[0])
    return kept


def energy_anchors(video) -> list[Anchor]:
    """Audio-energy anchors for `video` (wraps effects.energy_envelope)."""
    times, rms = energy_envelope(video, window_s=gconf.AUTO_ENERGY_WINDOW_S)
    return [Anchor(t, "energy", strength=pn) for t, pn in pick_energy_peaks(times, rms)]


def _utterances(transcript: Transcript):
    """Group words into per-consecutive-speaker utterances:
    yields (start, end, speaker, text)."""
    cur: list[str] = []
    spk = None
    st = en = 0.0
    for w in transcript.words:
        s = w.speaker or "SPEAKER"
        if s != spk and cur:
            yield (st, en, spk, " ".join(cur))
            cur = []
        if not cur:
            spk, st = s, w.start
        cur.append(w.text)
        en = w.end
    if cur:
        yield (st, en, spk, " ".join(cur))


def scan_reactions(transcript: Transcript, lexicon: dict | None = None) -> list[Anchor]:
    """PURE. Reaction-keyword anchors: substring-match each category's phrases over
    per-speaker utterances (case-insensitive). Each utterance-with-hits yields an
    anchor at its start carrying the category hint and the hit count as strength."""
    lexicon = gconf.AUTO_REACTION_LEXICON if lexicon is None else lexicon
    anchors: list[Anchor] = []
    for st, _en, _spk, text in _utterances(transcript):
        low = text.lower()
        for cat, phrases in lexicon.items():
            hits = sum(low.count(p.lower()) for p in phrases if p)
            if hits:
                anchors.append(Anchor(st, "reaction", strength=float(hits),
                                      category_hint=cat))
    return anchors


# ============================================================================
# B. window framing
# ============================================================================

def frame_window(anchor_time: float, transcript: Transcript, total: float, *,
                 lead_in=None, lead_out=None, cmin=None, cmax=None
                 ) -> tuple[float, float]:
    """PURE. Expand an anchor into a clip window: a lead-in before, a payoff after,
    snapped to transcript boundaries (start to a word start, end to a sentence end
    / word end), clamped to [cmin, cmax] and [0, total]."""
    lead_in = gconf.AUTO_LEAD_IN_S if lead_in is None else lead_in
    lead_out = gconf.AUTO_LEAD_OUT_S if lead_out is None else lead_out
    cmin = gconf.AUTO_CLIP_MIN_S if cmin is None else cmin
    cmax = gconf.AUTO_CLIP_MAX_S if cmax is None else cmax

    start = anchor_time - lead_in
    end = anchor_time + lead_out
    words = transcript.words
    if words:
        prior = [w.start for w in words if w.start <= start]
        if prior:
            start = max(prior)
        sent = [w.end for w in words
                if w.text.strip()[-1:] in ".?!" and w.end >= end]
        ends = [w.end for w in words if w.end >= end]
        if sent:
            end = min(sent)
        elif ends:
            end = min(ends)

    start = max(0.0, start)
    if total:
        end = min(end, total)
    if end - start < cmin:
        end = start + cmin
        if total:
            end = min(end, total)
    if end - start > cmax:
        end = start + cmax
    return round(start, 2), round(end, 2)


def merge_windows(windows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """PURE. Union overlapping/adjacent (start, end) intervals."""
    out: list[tuple[float, float]] = []
    for s, e in sorted(windows):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


# ============================================================================
# C. LLM judge (chunked)
# ============================================================================

_LLM_SYSTEM = (
    "You are a highlight finder for short vertical gameplay clips. You are given a "
    "timestamped, speaker-labelled transcript CHUNK from a longer session. Identify "
    "only genuinely clip-worthy moments — be selective; quality over quantity. "
    "Return STRICT JSON: a list of objects "
    '{"start": <sec>, "end": <sec>, "category": <one of clutch|funny|rage|hype|story>, '
    '"reason": <short>, "hook_caption": <=8 words, "confidence": <0..1>}. '
    "Use the ABSOLUTE timestamps shown in the transcript. If nothing stands out, "
    "return []. Return ONLY the JSON array, no prose."
)


def _digest_window(transcript: Transcript, start: float, end: float) -> str:
    """Per-speaker utterance lines (absolute timestamps) overlapping [start, end]."""
    lines = []
    for st, en, spk, text in _utterances(transcript):
        if en <= start or st >= end:
            continue
        lines.append(f"[{st:.1f}s {spk}] {text}")
    return "\n".join(lines)


def chunk_transcript(transcript: Transcript, chunk_s=None, overlap_s=None
                     ) -> list[tuple[float, float, str]]:
    """PURE. Split the transcript into ~chunk_s windows with overlap_s overlap so a
    moment on a boundary isn't split. Returns (chunk_start, chunk_end, digest).
    Digests carry ABSOLUTE timestamps, so LLM-returned times need no offset."""
    chunk_s = gconf.AUTO_LLM_CHUNK_S if chunk_s is None else chunk_s
    overlap_s = gconf.AUTO_LLM_CHUNK_OVERLAP_S if overlap_s is None else overlap_s
    words = transcript.words
    if not words:
        return []
    total = max(w.end for w in words)
    step = max(1.0, chunk_s - overlap_s)
    out: list[tuple[float, float, str]] = []
    cs = 0.0
    while cs < total:
        ce = min(cs + chunk_s, total)
        digest = _digest_window(transcript, cs, ce)
        if digest.strip():
            out.append((round(cs, 2), round(ce, 2), digest))
        if ce >= total:
            break
        cs += step
    return out


def _llm_raw(system: str, user: str, backend: str) -> str:
    if backend == "ollama":
        prompt = f"{system}\n\n--- TRANSCRIPT ---\n{user}\n\n--- JSON ---\n"
        r = subprocess.run(["ollama", "run", gconf.AUTO_OLLAMA_MODEL, prompt],
                            capture_output=True, text=True, timeout=180,
                            encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise RuntimeError(f"Ollama failed: {r.stderr}")
        return r.stdout
    if backend == "claude":
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(model="claude-3-5-haiku-latest",
                                     max_tokens=1200, system=system,
                                     messages=[{"role": "user", "content": user}])
        return msg.content[0].text
    raise ValueError(f"Unsupported AUTO_LLM_BACKEND: {backend}")


def parse_llm_moments(raw: str) -> list[LLMMoment]:
    """PURE + defensive. Extract the JSON array from a (possibly fenced / chatty)
    LLM reply; tolerate junk (return [] rather than raising)."""
    if not raw:
        return []
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(items, list):
        return []
    out: list[LLMMoment] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            s, e = float(it["start"]), float(it["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if e <= s:
            continue
        cat = str(it.get("category", "story")).lower().strip()
        if cat not in gconf.AUTO_CATEGORIES:
            cat = "story"
        try:
            conf = float(it.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        conf = min(1.0, max(0.0, conf))
        cap = str(it.get("hook_caption", it.get("caption", ""))).strip()
        out.append(LLMMoment(round(s, 2), round(e, 2), cat, cap,
                             str(it.get("reason", "")).strip(), conf))
    return out


def llm_judge(transcript: Transcript, backend: str | None = None,
              progress: Progress | None = None) -> list[LLMMoment]:
    """Run the LLM judge per transcript chunk; a bad chunk is skipped, not fatal."""
    backend = backend or gconf.AUTO_LLM_BACKEND
    if backend == "none" or not transcript.words:
        return []
    chunks = chunk_transcript(transcript)
    moments: list[LLMMoment] = []
    for i, (cs, ce, digest) in enumerate(chunks):
        if progress:
            progress(f"  LLM judge: chunk {i + 1}/{len(chunks)} ({cs:.0f}-{ce:.0f}s)")
        try:
            moments.extend(parse_llm_moments(_llm_raw(_LLM_SYSTEM, digest, backend)))
        except Exception as e:        # noqa: BLE001 — contain a bad chunk
            if progress:
                progress(f"    chunk {i + 1} skipped ({type(e).__name__})")
    return moments


# ============================================================================
# D. fuse + score + rank
# ============================================================================

def _overlap(a_s: float, a_e: float, b_s: float, b_e: float) -> bool:
    return min(a_e, b_e) - max(a_s, b_s) > 0


def _speakers_in(transcript: Transcript, s: float, e: float) -> int:
    return len({w.speaker for w in transcript.words
                if w.speaker and w.end > s and w.start < e})


def _reaction_density(reaction_anchors: list[Anchor], s: float, e: float) -> float:
    hits = sum(a.strength for a in reaction_anchors if s <= a.time < e)
    return hits / max(1e-6, e - s)


def fuse_and_rank(energy_anchors: list[Anchor], reaction_anchors: list[Anchor],
                  llm_moments: list[LLMMoment], transcript: Transcript,
                  total: float | None = None, top_n: int | None = None
                  ) -> list[Candidate]:
    """PURE. Build candidate windows from the three sources, score each by a
    weighted sum of four 0..1 signals, then sort (deterministic), greedily
    de-overlap keeping the higher score, and cap to top_n."""
    top_n = gconf.AUTO_TOP_N if top_n is None else top_n
    if total is None:
        total = max((w.end for w in transcript.words), default=0.0)

    windows: list[tuple[float, float, str]] = []
    for a in energy_anchors:
        s, e = frame_window(a.time, transcript, total)
        windows.append((s, e, "energy"))
    for a in reaction_anchors:
        s, e = frame_window(a.time, transcript, total)
        windows.append((s, e, "reaction"))
    for m in llm_moments:
        windows.append((m.start, m.end, "llm"))

    cands: list[Candidate] = []
    for s, e, base in windows:
        best_llm = None
        for m in llm_moments:
            if _overlap(s, e, m.start, m.end) and (
                    best_llm is None or m.confidence > best_llm.confidence):
                best_llm = m
        llm_conf = best_llm.confidence if best_llm else 0.0
        e_prom = max((a.strength for a in energy_anchors if s <= a.time < e),
                     default=0.0)
        react = min(1.0, _reaction_density(reaction_anchors, s, e)
                    / max(1e-6, gconf.AUTO_REACTION_DENSITY_CAP))
        spk = min(_speakers_in(transcript, s, e), 3) / 3.0

        score = (gconf.AUTO_W_LLM * llm_conf + gconf.AUTO_W_ENERGY * e_prom
                 + gconf.AUTO_W_REACTION * react + gconf.AUTO_W_OVERLAP * spk)

        if best_llm:
            cat, cap, reason = best_llm.category, best_llm.caption, best_llm.reason
        else:
            hint = next((a.category_hint for a in reaction_anchors
                         if s <= a.time < e and a.category_hint), None)
            cat, cap, reason = (hint or "highlight"), "", ""

        srcs = []
        if best_llm:
            srcs.append("llm")
        if e_prom > 0:
            srcs.append("energy")
        if react > 0:
            srcs.append("reaction")
        source = "+".join(srcs) or base
        cands.append(Candidate(round(s, 2), round(e, 2), cat, cap,
                               round(score, 4), source, reason))

    # deterministic order: score desc, then earliest, then shortest
    cands.sort(key=lambda c: (-c.score, c.start, c.end))
    final: list[Candidate] = []
    for c in cands:
        if any(_overlap(c.start, c.end, f.start, f.end) for f in final):
            continue
        final.append(c)
        if len(final) >= top_n:
            break
    return final


def detect_highlights(video, transcript: Transcript, backend: str | None = None,
                      progress: Progress | None = None, top_n: int | None = None
                      ) -> list[Candidate]:
    """Full fused detection: energy + reactions + LLM judge -> ranked candidates."""
    emit = progress or (lambda m: None)
    total = max((w.end for w in transcript.words), default=0.0)

    emit("Detecting audio-energy peaks...")
    e_anchors = energy_anchors(video)
    emit(f"  {len(e_anchors)} energy anchor(s).")

    emit("Scanning reaction keywords...")
    r_anchors = scan_reactions(transcript)
    emit(f"  {len(r_anchors)} reaction hit(s).")

    emit("LLM judge (chunked over the transcript)...")
    moments = llm_judge(transcript, backend, progress=progress)
    emit(f"  {len(moments)} LLM moment(s)"
         f"{' (LLM off/unavailable — using energy+reactions)' if not moments else ''}.")

    cands = fuse_and_rank(e_anchors, r_anchors, moments, transcript,
                          total=total, top_n=top_n)
    emit(f"{len(cands)} candidate(s) after fuse + dedupe.")
    return cands
