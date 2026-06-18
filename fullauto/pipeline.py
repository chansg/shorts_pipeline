"""EXPERIMENTAL — full-auto pipeline: a long video in, ONE 16:9 YouTube video out.

Ingest -> transcribe + diarize -> detect & categorise highlights -> review ->
auto-cut the chosen highlights -> assemble into a 16:9 YouTube video.

This deliberately does NOT touch the 9:16 Shorts backend (blur-pad reframe,
like/subscribe overlay, karaoke captioner, vertical export) — full-auto exports a
standard landscape YouTube video via fullauto.export. Detection still reuses the
shared, aspect-agnostic infra in gameplay/ (transcription, config, energy envelope).

Failure-contained and GPU+token gated for the transcribe step; compute-heavy on a
~1hr video (acceptable as a batch job).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from modules.assemble import _run, _probe_duration
from orchestrator.errors import FriendlyError, ensure_ffmpeg
from fullauto import highlight as ah
from gameplay import config as gconf
from gameplay.state import AutoSession, slugify
from gameplay.transcribe import transcribe
from gameplay.transcript import Transcript, Word

Progress = Callable[[str], None]


def slice_transcript(transcript: Transcript, start: float, end: float) -> Transcript:
    """The words inside [start, end], rebased so the clip starts at t=0.

    Kept for the future 16:9-captioning hook (see fullauto.export): full-auto does
    not burn captions today, but the per-window transcript is here when it does."""
    words = [Word(w.text, round(w.start - start, 3), round(w.end - start, 3),
                  w.speaker)
             for w in transcript.words if w.end > start and w.start < end]
    for w in words:                       # clamp to the window
        w.start = max(0.0, w.start)
    return Transcript(words=words, single_speaker=transcript.single_speaker)


# ---- review-first flow: detect -> review -> build 16:9 ----------------------
# Detection is split from building so the user reviews candidates (with previews)
# before committing render time. Building assembles a 16:9 YouTube video — it does
# NOT reuse the manual 9:16 backend.

def _cand_to_dict(c: ah.Candidate) -> dict:
    return {"start": c.start, "end": c.end, "category": c.category,
            "caption": c.caption, "score": c.score, "source": c.source,
            "reason": c.reason}


def _cand_from_dict(d: dict) -> ah.Candidate:
    return ah.Candidate(float(d["start"]), float(d["end"]), d.get("category", "story"),
                        d.get("caption", ""), float(d.get("score", 0.0)),
                        d.get("source", "energy"), d.get("reason", ""))


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
                      max_clips: int | None = None, diarize: bool = True,
                      progress: Progress | None = None
                      ) -> tuple[list[ah.Candidate], AutoSession]:
    """Transcribe + fused highlight detection for a long video — WITHOUT building.
    Persists the transcript, candidates.json, and a preview thumbnail per
    candidate so the GUI can present a review gallery. Returns (candidates,
    session). Staged progress with elapsed time across transcribe -> detect ->
    preview so an hour-long job never looks hung."""
    import time
    ensure_ffmpeg()
    video = Path(video)
    if not video.exists():
        raise FriendlyError(f"Video not found: {video}")
    emit = (lambda m: progress(m)) if progress else (lambda m: None)
    session = AutoSession(video.stem)
    top_n = int(max_clips) if max_clips else gconf.AUTO_TOP_N

    total = _probe_duration(video)
    mins = total / 60.0
    if mins > gconf.AUTO_MAX_MINUTES:        # warn, don't block
        emit(f"⚠ Video is {mins:.0f} min (over the ~{gconf.AUTO_MAX_MINUTES} min "
             f"guideline). Proceeding, but transcription will take a while.")
    t0 = time.time()
    emit(f"[1/3] Ingesting {video.name} ({mins:.1f} min). Transcribing + diarizing "
         f"— the slow step (roughly real-time-ish on GPU); progress below...")
    transcript = transcribe(video, progress=progress, diarize=diarize,
                            batch_size=gconf.AUTO_TRANSCRIBE_BATCH)
    transcript.save(session.transcript_path)
    spk = ("single speaker" if transcript.single_speaker
           else f"{len(transcript.speakers)} speakers")
    emit(f"[2/3] Detecting highlights ({len(transcript.words)} words, {spk}; "
         f"{time.time() - t0:.0f}s elapsed)...")
    candidates = ah.detect_highlights(video, transcript, backend=backend,
                                      progress=progress, top_n=top_n)

    session.candidates_path.write_text(
        json.dumps([_cand_to_dict(c) for c in candidates], indent=2),
        encoding="utf-8")
    emit(f"[3/3] Rendering {len(candidates)} preview thumbnail(s)...")
    for i, c in enumerate(candidates):
        try:
            make_preview(video, c, session.preview_path(i))
        except Exception:        # noqa: BLE001 — a missing thumb shouldn't sink detect
            pass
    emit(f"Done in {time.time() - t0:.0f}s — {len(candidates)} candidate(s) ready "
         f"for review.")
    return candidates, session


def load_candidates(session: AutoSession) -> list[ah.Candidate]:
    if not session.has_candidates():
        return []
    data = json.loads(session.candidates_path.read_text(encoding="utf-8"))
    return [_cand_from_dict(d) for d in data]


def build_youtube(session: AutoSession, video: str | Path,
                  candidates: list[ah.Candidate],
                  progress: Progress | None = None) -> Path:
    """Assemble the chosen candidates into ONE 16:9 YouTube video at native
    resolution. No blur-pad reframe, no like/subscribe overlay, no karaoke captions
    — that 9:16 Shorts backend is intentionally not called. Profanity is bleeped from
    the audio (shared gameplay.censor), audio-only since there are no captions here.
    Returns the output path."""
    from fullauto import export as export_mod
    censor_spans = None
    if gconf.CENSOR_ENABLED and gconf.CENSOR_AUDIO and session.transcript_path.exists():
        from gameplay import censor as cmod
        t = Transcript.load(session.transcript_path)
        censor_spans = cmod.merge_spans(t.censor_spans(), gconf.CENSOR_PAD_S)
    out = session.dir / f"{session.name}_youtube.mp4"
    return export_mod.export_youtube(video, candidates, out, progress=progress,
                                     censor_spans=censor_spans)
