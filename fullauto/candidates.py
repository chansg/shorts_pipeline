"""Full-auto CANDIDATE EXPORT — a folder of long OBS recordings -> the top-N raw
60-90s highlight TRIMS per source, for manual finishing later.

This is a SEPARATE stage from fullauto.clips (which reframes to 9:16 for the manual
editor). Here the output is RAW cuts — no reframe, captions, or overlay — that
**preserve BOTH audio tracks** (Track 1 mix + Track 2 voice), so the downstream
clean-voice transcription still has the isolated voice. Two signals drive selection:

  - VOICE ENERGY (primary, robust): the isolated voice track (a:1) folded to an RMS /
    onset-reaction curve -> `banter` candidates (squad reactions, funny/fail moments).
  - HUD OCR (booster + tagging): kill banners ("Pentakill", "Ace", ...) read with
    Tesseract -> timestamped events -> `play` candidates with a score boost.

Fail-safe everywhere: missing Track 2 -> full mix + WARNING; Tesseract missing ->
audio-only + WARNING; per-frame OCR error -> skip + log; no qualifying regions ->
export none and say so; one bad source -> logged, the batch continues.

The scoring / matching / selection / manifest are PURE (no ffmpeg) and unit-tested;
the ffmpeg/OCR seams are injectable.
"""
from __future__ import annotations

import argparse
import difflib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from gameplay import config as gconf
from fullauto import hud as hud_mod
from fullauto import reaction as rx

Progress = Callable[[str], None]

VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".m4v")


# ---- data ------------------------------------------------------------------

@dataclass
class OcrEvent:
    t: float            # time (s) the banner was read
    text: str           # canonical keyword text for the manifest (e.g. "PENTAKILL")
    keyword: str        # the matched OCR_KEYWORDS entry (e.g. "Pentakill")


@dataclass
class Candidate:
    rank: int
    category: str       # "play" (has an OCR kill event) | "banter" (voice only)
    score: float        # 0..1 combined interest at the peak
    start: float
    end: float
    peak: float
    voice_energy_score: float
    ocr_events: list[OcrEvent] = field(default_factory=list)
    output: str = ""
    why: str = ""

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)


# ---- PURE: fuzzy keyword matching ------------------------------------------

def match_keyword(text, keywords=None, threshold=None):
    """PURE. Match noisy OCR `text` against `keywords`, tolerating Tesseract misreads.
    Substring (collapsed/lowercased) is an exact hit; otherwise the best difflib ratio
    over same-length word windows must reach `threshold`. Returns (keyword, ratio) or
    None. Exact-substring beats fuzzy so 'TRIPLE KILL' isn't mis-snapped to 'Double Kill'."""
    keywords = gconf.OCR_KEYWORDS if keywords is None else keywords
    threshold = gconf.OCR_FUZZY_THRESHOLD if threshold is None else threshold
    low = " ".join(str(text or "").lower().split())
    if not low:
        return None
    words = low.split()
    best, best_r = None, 0.0
    for kw in keywords:
        k = " ".join(str(kw).lower().split())
        if not k:
            continue
        if k in low:                                   # exact substring -> strongest
            return (kw, 1.0)
        klen = len(k.split())
        r = difflib.SequenceMatcher(None, k, low).ratio()
        for i in range(0, max(1, len(words) - klen + 1)):   # slide over the noise
            seg = " ".join(words[i:i + klen])
            r = max(r, difflib.SequenceMatcher(None, k, seg).ratio())
        if r > best_r:
            best, best_r = kw, r
    return (best, best_r) if best is not None and best_r >= threshold else None


# ---- PURE: window framing + scoring + selection ----------------------------

def frame_window(peak: float, duration: float, min_s=None, max_s=None
                 ) -> tuple[float, float]:
    """PURE. A 60-90s window anchored on `peak` (slightly before it for setup), clamped
    to [0, duration]. Targets the midpoint of [min_s, max_s]; if the source is shorter
    than the target, returns the whole clip (may be < min_s — the caller notes that)."""
    min_s = gconf.CLIP_MIN_SECONDS if min_s is None else min_s
    max_s = gconf.CLIP_MAX_SECONDS if max_s is None else max_s
    target = min(float(max_s), max(float(min_s), (float(min_s) + float(max_s)) / 2.0))
    if duration <= target:
        return (0.0, round(float(duration), 3))
    start = peak - target * 0.35                        # keep the build-up before the spike
    end = start + target
    if start < 0:
        start, end = 0.0, target
    if end > duration:
        end, start = float(duration), float(duration) - target
    return (round(start, 3), round(end, 3))


def build_interest(times, vscore, ocr_events, *, weight_voice=None, weight_ocr=None
                   ) -> np.ndarray:
    """PURE. Per-bin interest = weight_voice*voice_score + weight_ocr at each bin holding
    an OCR event. The OCR term lets a quiet-but-eventful `play` outscore loud banter."""
    weight_voice = gconf.WEIGHT_VOICE if weight_voice is None else weight_voice
    weight_ocr = gconf.WEIGHT_OCR if weight_ocr is None else weight_ocr
    times = np.asarray(times, dtype=float)
    interest = weight_voice * np.asarray(vscore, dtype=float)
    if interest.size == 0:
        return interest
    for e in ocr_events or []:
        i = int(np.argmin(np.abs(times - e.t)))
        interest[i] += weight_ocr
    return interest


def select_candidates(times, interest, vscore, ocr_events, duration, *,
                      n=None, min_gap=None, min_s=None, max_s=None, floor=None
                      ) -> list[Candidate]:
    """PURE. Greedily take the top-`n` interest peaks as non-overlapping 60-90s windows,
    strongest first, keeping peaks >= `min_gap` apart so the picks spread across the VOD.
    A window with an OCR event in it is `play` (peak = the banner), else `banter` (peak =
    the voice spike). `floor` gates weak peaks (default = the tuned reaction threshold)."""
    n = gconf.CANDIDATES_PER_SOURCE if n is None else n
    min_gap = gconf.MIN_GAP_SECONDS if min_gap is None else min_gap
    floor = (gconf.WEIGHT_VOICE * gconf.REACTION_THRESHOLD) if floor is None else floor
    times = np.asarray(times, dtype=float)
    interest = np.asarray(interest, dtype=float)
    vscore = np.asarray(vscore, dtype=float)
    if times.size == 0:
        return []
    chosen: list[Candidate] = []
    for idx in np.argsort(interest)[::-1]:
        val = float(interest[idx])
        if val < floor:
            break                                       # the rest are weaker still
        t = float(times[idx])
        if any(abs(t - c.peak) < min_gap for c in chosen):
            continue
        start, end = frame_window(t, duration, min_s, max_s)
        if any(not (end <= c.start or start >= c.end) for c in chosen):
            continue                                    # overlaps a stronger pick
        evs = [e for e in (ocr_events or []) if start <= e.t <= end]
        mask = (times >= start) & (times <= end)
        vmax = float(vscore[mask].max()) if mask.any() and vscore.size else 0.0
        if evs:
            peak = min(evs, key=lambda e: abs(e.t - t)).t
            kws = sorted({e.keyword for e in evs})
            why = f"{'/'.join(kws)} banner + voice reaction ({vmax:.2f})"
            category = "play"
        else:
            peak = t
            why = f"sustained squad reaction ({vmax:.2f})"
            category = "banter"
        score = min(1.0, val / (gconf.WEIGHT_VOICE + gconf.WEIGHT_OCR))
        chosen.append(Candidate(
            rank=len(chosen) + 1, category=category, score=round(score, 4),
            start=start, end=end, peak=round(peak, 3),
            voice_energy_score=round(vmax, 4), ocr_events=evs, why=why))
        if len(chosen) >= n:
            break
    return chosen


def candidate_filename(c: Candidate) -> str:
    """e.g. clip_01_play_1m18s.mp4 — rank, category, peak timestamp."""
    m, s = divmod(int(round(c.peak)), 60)
    return f"clip_{c.rank:02d}_{c.category}_{m}m{s:02d}s.mp4"


def manifest_dict(source: str, duration: float, audio_tracks: int,
                  cands: list[Candidate], note: str = "") -> dict:
    """PURE. The candidates.json structure."""
    out = {
        "source": source,
        "duration_s": round(float(duration), 1),
        "audio_tracks": int(audio_tracks),
        "candidates": [{
            "rank": c.rank, "category": c.category, "score": round(c.score, 2),
            "start_s": round(c.start, 1), "end_s": round(c.end, 1),
            "duration_s": round(c.end - c.start, 1), "peak_s": round(c.peak, 1),
            "voice_energy_score": round(c.voice_energy_score, 2),
            "ocr_events": [{"t_s": round(e.t, 1), "text": e.text} for e in c.ocr_events],
            "output": c.output, "why": c.why,
        } for c in cands],
    }
    if note:
        out["note"] = note
    return out


# ---- ffmpeg seams: probe, voice extraction, OCR sampling, export ------------

def count_audio_tracks(src) -> int:
    """Number of audio streams in `src` (an OBS dual-track recording has >=2)."""
    import subprocess
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(src)],
        capture_output=True, text=True).stdout
    return len([ln for ln in out.splitlines() if ln.strip()])


def extract_voice_wav(src, dest, *, track=None, tracks=None, progress=None):
    """Extract the voice track (a:`track`, default Track 2) to a mono 16k WAV for energy
    analysis. Falls back to the full mix (a:0) with a WARNING when the source has a single
    track (game audio present -> noisier energy). Returns (dest, n_audio_tracks)."""
    from modules.assemble import _run
    emit = progress or (lambda m: None)
    track = gconf.VOICE_TRACK_INDEX if track is None else int(track)
    n = count_audio_tracks(src) if tracks is None else tracks
    if n >= track + 1:
        use = track
    else:
        use = 0
        emit(f"  WARNING: source has {n} audio track(s) (no a:{track}) - analysing the "
             "full mix (a:0); energy detection will be noisier (game audio present).")
    _run(["ffmpeg", "-y", "-i", str(src), "-vn", "-map", f"0:a:{use}?",
          "-ac", "1", "-ar", str(gconf.REACTION_SR), "-c:a", "pcm_s16le", str(dest)])
    return Path(dest), n


def voice_score_curve(wav):
    """(times, voice_score 0..1) from a prepared voice WAV — reuses the streamed vocal
    band-pass reaction detector at ENERGY_WINDOW_S resolution. Empty arrays if no audio."""
    times, rms = rx.reaction_envelope(
        wav, window_s=gconf.ENERGY_WINDOW_S, band=gconf.REACTION_BAND_HZ,
        sr=gconf.REACTION_SR)
    if rms.size == 0:
        return np.asarray([]), np.asarray([])
    return times, rx.reaction_score(times, rms, window_s=gconf.ENERGY_WINDOW_S)


def _sample_all(video, duration: float, sample_fps: float, chunk_s: float = 120.0):
    """Yield (t, RGB frame) across the WHOLE clip at `sample_fps`, in bounded chunks so
    the temp JPEG dir never holds the whole video. A chunk that fails to decode is
    skipped (logged by the caller via the missing frames), not fatal."""
    t = 0.0
    while t < duration:
        end = min(duration, t + chunk_s)
        try:
            for item in hud_mod._sample_frames(video, t, end, sample_fps):
                yield item
        except Exception:        # noqa: BLE001 — skip a bad chunk, keep scanning
            pass
        t = end


def ocr_scan(video, duration: float, *, frames=None, recognizer=None, sample_fps=None,
             crop=None, keywords=None, threshold=None, progress=None) -> list[OcrEvent]:
    """Scan the whole clip for kill banners. PER-FRAME guarded: a bad frame / OCR error
    is skipped, never aborts the scan. Returns [] (with a logged reason) when Tesseract
    is unavailable. `frames`/`recognizer` are injectable for testing without ffmpeg/OCR."""
    emit = progress or (lambda m: None)
    sample_fps = gconf.OCR_SAMPLE_FPS if sample_fps is None else sample_fps
    crop = gconf.OCR_CROP if crop is None else crop
    if frames is None and not hud_mod.ocr_available():
        emit("  " + hud_mod.ocr_unavailable_message() + " Continuing audio-only.")
        return []
    recognizer = recognizer or hud_mod._default_recognizer
    frame_iter = (frames if frames is not None
                  else _sample_all(video, duration, sample_fps))
    events: list[OcrEvent] = []
    bad = 0
    for t, frame in frame_iter:
        try:
            roi = hud_mod.roi_crop(frame, crop) if crop else frame
            m = match_keyword(recognizer(roi), keywords, threshold)
        except Exception:        # noqa: BLE001 — per-frame failure: skip, keep going
            bad += 1
            continue
        if m:
            events.append(OcrEvent(round(float(t), 2), str(m[0]).upper(), m[0]))
    if bad:
        emit(f"  OCR: skipped {bad} unreadable frame(s) (non-fatal).")
    emit(f"  OCR: {len(events)} kill-banner event(s).")
    return events


def _video_args(encoder: str, quality) -> list[str]:
    """Encoder args. NVENC quality is -cq; libx264 is -crf. Other encoders pass quality
    through as -crf (sane default)."""
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", str(quality),
                "-pix_fmt", "yuv420p"]
    return ["-c:v", encoder, "-preset", "medium", "-crf", str(quality),
            "-pix_fmt", "yuv420p"]


def export_candidate(src, c: Candidate, out_path, *, tracks: int, encoder=None,
                     quality=None, audio_bitrate=None) -> Path:
    """Frame-accurate trim of [start, end] from the ORIGINAL source, re-encoding video
    (so the cut is exact) and KEEPING BOTH AUDIO TRACKS (-map 0:a:0 -map 0:a:1) so the
    downstream clean-voice transcription still has Track 2. Single-track sources keep a:0."""
    from modules.assemble import _run
    encoder = gconf.CAND_ENCODER if encoder is None else encoder
    quality = gconf.CAND_QUALITY if quality is None else quality
    audio_bitrate = gconf.CAND_AUDIO_BITRATE if audio_bitrate is None else audio_bitrate
    dur = max(0.0, c.end - c.start)
    cmd = ["ffmpeg", "-y", "-ss", f"{c.start:.3f}", "-i", str(src), "-t", f"{dur:.3f}",
           "-map", "0:v:0"]
    cmd += (["-map", "0:a:0", "-map", "0:a:1"] if tracks >= 2 else ["-map", "0:a:0?"])
    cmd += _video_args(encoder, quality) + ["-c:a", "aac", "-b:a", str(audio_bitrate),
                                            str(out_path)]
    _run(cmd)
    return Path(out_path)


# ---- orchestration ---------------------------------------------------------

def collect_inputs(inputs) -> list[Path]:
    """Resolve `inputs` (a folder, a single file, or a list of either) to source videos.
    A directory contributes its top-level video files (sorted, non-recursive)."""
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    out: list[Path] = []
    for item in inputs or []:
        p = Path(item)
        if p.is_dir():
            out += sorted(q for q in p.iterdir()
                          if q.is_file() and q.suffix.lower() in VIDEO_EXTS)
        elif p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            out.append(p)
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def run_source(src, out_root, *, progress=None) -> tuple[Path, dict]:
    """Detect + export the top-N candidates for ONE source. Writes the clips and
    candidates.json under <out_root>/<source_stem>/. Returns (out_dir, manifest)."""
    from modules.assemble import _probe_duration
    emit = progress or (lambda m: None)
    src = Path(src)
    out_dir = Path(out_root) / src.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = float(_probe_duration(src) or 0.0)
    tracks = count_audio_tracks(src)
    times, vscore = np.asarray([]), np.asarray([])
    if tracks >= 1:
        with tempfile.TemporaryDirectory(prefix="cand_") as td:
            wav = Path(td) / "voice.wav"
            extract_voice_wav(src, wav, tracks=tracks, progress=emit)
            emit(f"  {tracks} audio track(s); {duration:.0f}s. Scoring voice energy "
                 f"(a:{gconf.VOICE_TRACK_INDEX if tracks >= 2 else 0})...")
            times, vscore = voice_score_curve(wav)
    else:
        emit(f"  {tracks} audio track(s); {duration:.0f}s. No audio — voice energy "
             "skipped (OCR-only).")

    ocr_events = ocr_scan(src, duration, progress=emit)
    interest = build_interest(times, vscore, ocr_events)
    cands = select_candidates(times, interest, vscore, ocr_events, duration)

    note = ""
    if not cands:
        note = "No qualifying highlight regions (no reaction above threshold, no kill banners)."
        emit("  " + note)
    elif len(cands) < gconf.CANDIDATES_PER_SOURCE:
        note = (f"Only {len(cands)} qualifying region(s) (<{gconf.CANDIDATES_PER_SOURCE}).")
    if duration and duration < gconf.CLIP_MIN_SECONDS:
        note = (note + " " if note else "") + (
            f"Source shorter than CLIP_MIN_SECONDS ({gconf.CLIP_MIN_SECONDS}s); "
            "clip is the whole recording.")

    for c in cands:
        c.output = candidate_filename(c)
        try:
            export_candidate(src, c, out_dir / c.output, tracks=tracks)
            emit(f"  rank {c.rank}: {c.category} {c.start:.0f}-{c.end:.0f}s "
                 f"-> {c.output}")
        except Exception as e:        # noqa: BLE001 — one bad cut shouldn't sink the source
            emit(f"  export failed for rank {c.rank}: {type(e).__name__}: {e}")
            c.output = ""

    manifest = manifest_dict(src.name, duration, tracks, cands, note)
    (out_dir / "candidates.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    emit(f"  wrote {out_dir / 'candidates.json'} ({len(cands)} candidate(s)).")
    return out_dir, manifest


def run_batch(inputs=None, out_root=None, *, progress=None) -> list[tuple[Path, dict]]:
    """Process each source independently. One source failing is logged, not fatal."""
    from orchestrator.errors import ensure_ffmpeg
    emit = progress or print
    ensure_ffmpeg()
    inputs = [gconf.CAND_INPUT_DIR] if inputs is None else inputs
    out_root = gconf.CAND_OUTPUT_DIR if out_root is None else out_root
    srcs = collect_inputs(inputs)
    if not srcs:
        emit(f"No source videos found in {inputs}.")
        return []
    Path(out_root).mkdir(parents=True, exist_ok=True)
    emit(f"{len(srcs)} source(s) -> {out_root}")
    results = []
    for src in srcs:
        emit(f"\n=== {src.name} ===")
        try:
            results.append(run_source(src, out_root, progress=emit))
        except Exception as e:        # noqa: BLE001 — keep the batch alive
            emit(f"  FAILED: {type(e).__name__}: {e}")
    return results


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Export the top-N raw 60-90s highlight candidates per source "
                    "(voice energy + HUD OCR), preserving both audio tracks.")
    ap.add_argument("--input", nargs="*", default=None,
                    help="source video files, or one folder (default: config.CAND_INPUT_DIR)")
    ap.add_argument("--output", default=None,
                    help="output root (default: config.CAND_OUTPUT_DIR)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    def safe_print(msg):
        # never let a legacy-codepage console (cp1252) abort the batch on an odd glyph
        enc = getattr(__import__("sys").stdout, "encoding", None) or "utf-8"
        print(str(msg).encode(enc, "replace").decode(enc, "replace"))

    run_batch(args.input, args.output, progress=safe_print)
    return 0


if __name__ == "__main__":          # pragma: no cover
    raise SystemExit(main())
