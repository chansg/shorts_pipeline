"""EXPERIMENTAL — game-agnostic highlight detection for the full-auto mode.

Combines two passes over a long, diarized clip:
  1. an audio-energy spike pass (loud reactions = action / laughter / clutch), and
  2. an LLM pass over the diarized transcript that finds and *categorises*
     candidate windows (clutch / funny / rage / story) with a suggested caption.

The two are merged and ranked. This is isolated from the manual path and degrades
gracefully: if the LLM backend is unavailable the energy pass still yields
candidates (uncategorised).
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass

from gameplay import config as gconf
from gameplay.effects import energy_envelope
from gameplay.transcript import Transcript


@dataclass
class Candidate:
    start: float
    end: float
    category: str          # one of gconf.AUTO_CATEGORIES, or "highlight"
    caption: str = ""
    score: float = 0.0
    source: str = "energy"  # "energy" | "llm" | "energy+llm"

    @property
    def duration(self) -> float:
        return self.end - self.start


def _clamp_window(center: float, total: float | None = None) -> tuple[float, float]:
    pre, post = 3.0, 4.0
    start = max(0.0, center - pre)
    end = center + post
    dur = end - start
    if dur < gconf.AUTO_CLIP_MIN_S:
        end = start + gconf.AUTO_CLIP_MIN_S
    if end - start > gconf.AUTO_CLIP_MAX_S:
        end = start + gconf.AUTO_CLIP_MAX_S
    if total:
        end = min(end, total)
    return round(start, 2), round(end, 2)


def energy_candidates(video, top_n: int | None = None) -> list[Candidate]:
    """Loud-moment windows ranked by peak loudness. Uncategorised (the LLM pass
    fills category/caption when it overlaps)."""
    top_n = top_n or gconf.AUTO_ENERGY_TOP_N
    times, rms = energy_envelope(video, window_s=0.25)
    if rms.size < 3:
        return []
    thresh = rms.mean() + 1.0 * rms.std()
    peaks = [(float(rms[i]), float(times[i])) for i in range(1, len(rms) - 1)
             if rms[i] >= thresh and rms[i] >= rms[i - 1] and rms[i] >= rms[i + 1]]
    peaks.sort(reverse=True)
    out: list[Candidate] = []
    for score, t in peaks:
        s, e = _clamp_window(t)
        if any(not (e <= c.start or s >= c.end) for c in out):   # overlap -> skip
            continue
        out.append(Candidate(s, e, "highlight", "", score, "energy"))
        if len(out) >= top_n:
            break
    return out


# ---- LLM categorization ----------------------------------------------------

_LLM_SYSTEM = (
    "You are a gameplay highlight finder for short vertical clips. You are given a "
    "timestamped, speaker-labelled transcript of a long gameplay session. Find the "
    "best 4-8 highlight moments. For each, return start and end times in SECONDS "
    "(6-45s long), a category that is EXACTLY one of: clutch, funny, rage, story, "
    "and a punchy caption of at most 8 words. Return ONLY a JSON array like: "
    '[{"start": 12.0, "end": 28.5, "category": "funny", "caption": "..."}]')


def _transcript_digest(transcript: Transcript, max_chars: int = 6000) -> str:
    """Group words into per-speaker utterances with a start timestamp."""
    lines, cur, cur_spk, cur_start = [], [], None, 0.0
    for w in transcript.words:
        spk = w.speaker or "SPEAKER"
        if spk != cur_spk and cur:
            lines.append(f"[{cur_start:.1f}s {cur_spk}] {' '.join(cur)}")
            cur = []
        if not cur:
            cur_spk, cur_start = spk, w.start
        cur.append(w.text)
    if cur:
        lines.append(f"[{cur_start:.1f}s {cur_spk}] {' '.join(cur)}")
    text = "\n".join(lines)
    return text[:max_chars]


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


def _parse_candidates(raw: str) -> list[Candidate]:
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out: list[Candidate] = []
    for it in items:
        try:
            start, end = float(it["start"]), float(it["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        cat = str(it.get("category", "story")).lower().strip()
        if cat not in gconf.AUTO_CATEGORIES:
            cat = "story"
        out.append(Candidate(round(start, 2), round(end, 2), cat,
                             str(it.get("caption", "")).strip(), 1.0, "llm"))
    return out


def llm_candidates(transcript: Transcript, backend: str | None = None) -> list[Candidate]:
    backend = backend or gconf.AUTO_LLM_BACKEND
    if backend == "none" or not transcript.words:
        return []
    try:
        raw = _llm_raw(_LLM_SYSTEM, _transcript_digest(transcript), backend)
    except Exception:        # noqa: BLE001 — degrade to energy-only on any LLM error
        return []
    return _parse_candidates(raw)


# ---- merge + rank ----------------------------------------------------------

def _overlaps(a: Candidate, b: Candidate) -> bool:
    return not (a.end <= b.start or a.start >= b.end)


def rank_candidates(energy: list[Candidate], llm: list[Candidate]) -> list[Candidate]:
    """Merge the two passes. An energy window that overlaps an LLM window inherits
    its category + caption and gets a boosted score (loud AND interesting).
    Remaining LLM windows are kept (categorised), and unmatched energy windows are
    kept as plain highlights. Sorted by score descending, de-overlapped."""
    merged: list[Candidate] = []
    used_llm = set()
    # normalise energy scores to ~0..1 so they combine with the LLM's 1.0 base
    emax = max((c.score for c in energy), default=1.0) or 1.0
    for ec in energy:
        ec = Candidate(ec.start, ec.end, ec.category, ec.caption,
                       ec.score / emax, ec.source)
        match = next((i for i, lc in enumerate(llm)
                      if i not in used_llm and _overlaps(ec, lc)), None)
        if match is not None:
            lc = llm[match]
            used_llm.add(match)
            ec.category, ec.caption = lc.category, lc.caption
            ec.score += 1.0
            ec.source = "energy+llm"
        merged.append(ec)
    for i, lc in enumerate(llm):
        if i not in used_llm:
            merged.append(lc)

    merged.sort(key=lambda c: c.score, reverse=True)
    final: list[Candidate] = []
    for c in merged:                       # greedy de-overlap, strongest first
        if not any(_overlaps(c, f) for f in final):
            final.append(c)
    return final


def detect_highlights(video, transcript: Transcript,
                      backend: str | None = None) -> list[Candidate]:
    return rank_candidates(energy_candidates(video),
                           llm_candidates(transcript, backend))
