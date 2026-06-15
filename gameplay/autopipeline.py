"""EXPERIMENTAL — full-auto pipeline: a long gameplay video in, a set of finished
candidate Shorts out.

Ingest -> transcribe + diarize -> detect & categorise highlights -> auto-cut each
candidate -> feed each through the MANUAL backend (reframe -> captions -> effects
-> overlay). The set is returned for the user to keep or discard.

Isolated from the manual path and failure-contained: one bad candidate is skipped,
not fatal. This is compute-heavy on a ~1hr video (acceptable as a batch job) and is
GPU+token gated for the transcribe step.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterator

from modules.assemble import _run, _probe_duration
from orchestrator.errors import FriendlyError, ensure_ffmpeg
from gameplay import autohighlight as ah
from gameplay.manual import ManualOptions, run_manual
from gameplay.state import AutoSession, GameplayClip, slugify
from gameplay.transcribe import transcribe
from gameplay.transcript import Transcript, Word

Progress = Callable[[str], None]


def slice_transcript(transcript: Transcript, start: float, end: float) -> Transcript:
    """The words inside [start, end], rebased so the clip starts at t=0."""
    words = [Word(w.text, round(w.start - start, 3), round(w.end - start, 3),
                  w.speaker)
             for w in transcript.words if w.end > start and w.start < end]
    for w in words:                       # clamp to the window
        w.start = max(0.0, w.start)
    return Transcript(words=words, single_speaker=transcript.single_speaker)


def cut_clip(video: str | Path, cand: ah.Candidate, name: str) -> GameplayClip:
    """Cut [cand.start, cand.end] from `video` into a fresh GameplayClip source."""
    clip = GameplayClip(name)
    dest = clip.dir / "source.mp4"
    if not clip.has_source():
        # -fps_mode cfr: force constant frame rate so burned captions don't drift
        # (long captures are often VFR). -map keeps only the first v+a, dropping
        # any stray data/timecode track.
        _run(["ffmpeg", "-y", "-ss", f"{cand.start:.3f}", "-to", f"{cand.end:.3f}",
              "-i", str(video), "-map", "0:v:0", "-map", "0:a:0?", "-dn",
              "-map_metadata", "-1", "-fps_mode", "cfr",
              "-c:v", "libx264", "-pix_fmt", "yuv420p",
              "-c:a", "aac", "-b:a", "192k", str(dest)])
    return clip


def run_autopipeline(video: str | Path, opts: ManualOptions | None = None,
                     backend: str | None = None, max_clips: int = 8
                     ) -> Iterator[dict]:
    """Yields {"msg": str} progress events, then a final
    {"done": True, "results": [{"candidate": Candidate, "output": Path}, ...]}.
    """
    ensure_ffmpeg()
    video = Path(video)
    opts = opts or ManualOptions()
    if not video.exists():
        raise FriendlyError(f"Video not found: {video}")

    total = _probe_duration(video)
    yield {"msg": f"Ingesting {video.name} ({total/60:.1f} min). "
                  f"Transcribing + diarizing (this is the slow part)..."}
    transcript = transcribe(video, progress=lambda m: None)
    yield {"msg": f"Transcript: {len(transcript.words)} words, "
                  f"{'single speaker' if transcript.single_speaker else f'{len(transcript.speakers)} speakers'}."}

    yield {"msg": "Detecting loud moments (audio-energy pass)..."}
    energy = ah.energy_candidates(video)
    yield {"msg": f"  {len(energy)} loud window(s)."}
    yield {"msg": "Categorising with the LLM (clutch/funny/rage/story)..."}
    llm = ah.llm_candidates(transcript, backend)
    yield {"msg": f"  {len(llm)} LLM-categorised window(s)"
                  f"{' (LLM unavailable — energy-only)' if not llm else ''}."}

    candidates = ah.rank_candidates(energy, llm)[:max_clips]
    if not candidates:
        yield {"msg": "No candidates found.", "done": True, "results": []}
        return
    yield {"msg": f"Building {len(candidates)} candidate Short(s)..."}

    results: list[dict] = []
    for i, c in enumerate(candidates):
        label = f"{i+1}/{len(candidates)} [{c.category}] {c.start:.0f}-{c.end:.0f}s"
        try:
            yield {"msg": f"  Cutting {label}: {c.caption or '(no caption)'}"}
            name = slugify(f"auto_{i:02d}_{c.category}")
            clip = cut_clip(video, c, name)
            sub = slice_transcript(transcript, c.start, c.end)
            sub.save(clip.transcript_path)
            out = None
            for ev in run_manual(clip, sub, opts, force=True):
                if ev.get("done"):
                    out = ev["output"]
            results.append({"candidate": c, "output": out})
            yield {"msg": f"  Done {label} -> {out}"}
        except Exception as e:           # noqa: BLE001 — contain per-candidate failures
            yield {"msg": f"  Candidate {label} failed ({type(e).__name__}: {e}); "
                          f"skipped."}

    yield {"msg": f"Full-auto complete: {len(results)} clip(s) built.",
           "done": True, "results": results}


# ---- review-first flow: detect -> review -> load-into-manual / batch --------
# Splits detection from building so the user reviews candidates (with previews)
# before committing GPU/render time. Building reuses the manual backend.

def _cand_to_dict(c: ah.Candidate) -> dict:
    return {"start": c.start, "end": c.end, "category": c.category,
            "caption": c.caption, "score": c.score, "source": c.source}


def _cand_from_dict(d: dict) -> ah.Candidate:
    return ah.Candidate(float(d["start"]), float(d["end"]), d.get("category", "story"),
                        d.get("caption", ""), float(d.get("score", 0.0)),
                        d.get("source", "energy"))


def candidate_name(session: AutoSession, cand: ah.Candidate) -> str:
    return slugify(f"{session.name}_{int(round(cand.start))}s_{cand.category}")


def make_preview(video: str | Path, cand: ah.Candidate, out: Path) -> Path:
    """A single representative frame (downscaled) from the candidate's midpoint —
    cheap thumbnail for the review gallery."""
    mid = (cand.start + cand.end) / 2.0
    _run(["ffmpeg", "-y", "-ss", f"{mid:.3f}", "-i", str(video), "-frames:v", "1",
          "-vf", "scale=360:-2", str(out)])
    return out


def detect_candidates(video: str | Path, backend: str | None = None,
                      max_clips: int = 8, diarize: bool = True,
                      progress: Progress | None = None
                      ) -> tuple[list[ah.Candidate], AutoSession]:
    """Transcribe + detect + rank highlights for a long video — WITHOUT building.
    Persists the transcript, candidates.json, and a preview thumbnail per
    candidate so the GUI can present a review gallery. Returns (candidates,
    session)."""
    ensure_ffmpeg()
    video = Path(video)
    if not video.exists():
        raise FriendlyError(f"Video not found: {video}")
    emit = (lambda m: progress(m)) if progress else (lambda m: None)
    session = AutoSession(video.stem)

    total = _probe_duration(video)
    emit(f"Ingesting {video.name} ({total/60:.1f} min). Transcribing + diarizing "
         f"(the slow part — progress below)...")
    transcript = transcribe(video, progress=progress, diarize=diarize)
    transcript.save(session.transcript_path)

    emit("Detecting loud moments (audio-energy pass)...")
    energy = ah.energy_candidates(video)
    emit(f"  {len(energy)} loud window(s).")
    emit("Categorising with the LLM (clutch/funny/rage/story)...")
    llm = ah.llm_candidates(transcript, backend)
    emit(f"  {len(llm)} LLM-categorised window(s)"
         f"{' (LLM unavailable — energy-only)' if not llm else ''}.")

    candidates = ah.rank_candidates(energy, llm)[:max_clips]
    session.candidates_path.write_text(
        json.dumps([_cand_to_dict(c) for c in candidates], indent=2),
        encoding="utf-8")
    for i, c in enumerate(candidates):
        try:
            make_preview(video, c, session.preview_path(i))
        except Exception:        # noqa: BLE001 — a missing thumb shouldn't sink detect
            pass
    emit(f"{len(candidates)} candidate(s) ready for review.")
    return candidates, session


def load_candidates(session: AutoSession) -> list[ah.Candidate]:
    if not session.has_candidates():
        return []
    data = json.loads(session.candidates_path.read_text(encoding="utf-8"))
    return [_cand_from_dict(d) for d in data]


def load_candidate(session: AutoSession, video: str | Path, cand: ah.Candidate
                   ) -> tuple[GameplayClip, Transcript]:
    """Cut the candidate's clip (CFR) and slice its transcript, rebased to t=0, for
    handoff into the manual flow. Returns (clip, transcript)."""
    transcript = Transcript.load(session.transcript_path)
    clip = cut_clip(video, cand, candidate_name(session, cand))
    sub = slice_transcript(transcript, cand.start, cand.end)
    sub.save(clip.transcript_path)
    return clip, sub


def build_candidate(session: AutoSession, video: str | Path, cand: ah.Candidate,
                    opts: ManualOptions | None = None) -> Path | None:
    """Build one candidate end-to-end via the manual backend (for batch build)."""
    clip, sub = load_candidate(session, video, cand)
    out = None
    for ev in run_manual(clip, sub, opts or ManualOptions(), force=True):
        if ev.get("done"):
            out = ev["output"]
    return out
