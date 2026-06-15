"""WhisperX transcribe + word-align + (optional) pyannote diarization.

One pass yields word-level timestamps and speaker labels for 4-5 people. Device +
model size come from gameplay.device (CUDA when available, else a smaller CPU
model). Diarization needs a free HuggingFace token (HF_TOKEN) AND a one-time
acceptance of the pyannote model licence; without it — or if only one voice is
found — the result falls back gracefully to single-speaker (no per-speaker colour).

WhisperX/torch are heavy and GPU-specific, so they are imported lazily inside the
functions: the rest of the gameplay package (and its tests) import and run without
them installed. Install via `pip install -r requirements-gameplay.txt`.

Diarization is kept self-contained (we extract speaker turns and assign them to
words ourselves) so it survives whisperx/pyannote API churn and is unit-testable
without the heavy deps — v1 broke precisely on such an API move
(`whisperx.DiarizationPipeline` -> `whisperx.diarize.DiarizationPipeline`).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable

from orchestrator.errors import FriendlyError
from gameplay import config as gconf
from gameplay import device as device_mod
from gameplay.state import GameplayClip
from gameplay.transcript import Transcript, from_whisperx

Progress = Callable[[str], None]


# ---- ingest -----------------------------------------------------------------

def _probe_streams(src: Path) -> list[dict]:
    import json
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(src)],
        capture_output=True, text=True,
    ).stdout
    try:
        return json.loads(out).get("streams", [])
    except (ValueError, KeyError):
        return []


def needs_normalization(src: str | Path) -> bool:
    """True if the source has non-A/V streams (stray data/timecode tracks) or looks
    VFR (avg_frame_rate != r_frame_rate) — both can desync burned captions or break
    filtergraphs. Clean CFR A/V clips return False (no needless re-encode)."""
    streams = _probe_streams(Path(src))
    if not streams:
        return False
    for s in streams:
        if s.get("codec_type") not in ("video", "audio"):
            return True            # data / timecode / subtitle track present
        if s.get("codec_type") == "video":
            avg, r = s.get("avg_frame_rate", "0/0"), s.get("r_frame_rate", "0/0")
            if avg not in ("0/0", "") and r not in ("0/0", "") and avg != r:
                return True        # variable frame rate
    return False


def normalize_source(src: str | Path, dest: str | Path) -> Path:
    """Re-encode to a clean, constant-frame-rate mp4 keeping only the first video +
    first audio stream (drops data/timecode tracks). Defensive against OBS/ShadowPlay
    VFR captures where burned captions drift from speech."""
    from modules.assemble import _run
    src, dest = Path(src), Path(dest)
    _run([
        "ffmpeg", "-y", "-i", str(src),
        "-map", "0:v:0", "-map", "0:a:0?", "-dn", "-map_metadata", "-1",
        "-fps_mode", "cfr",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", str(dest),
    ])
    return dest


def import_source(src: str | Path, name: str | None = None) -> GameplayClip:
    """Copy/normalise an uploaded clip into its work dir under
    output/gameplay/<name>/ and return the GameplayClip. Normalises (CFR + strip
    non-A/V) only when needed; otherwise a plain copy. Idempotent."""
    src = Path(src)
    clip = GameplayClip(name or src.stem)
    if not clip.has_source():
        if needs_normalization(src):
            normalize_source(src, clip.dir / "source.mp4")
        else:
            shutil.copy2(src, clip.dir / f"source{src.suffix.lower()}")
    return clip


# ---- diarization (self-contained, version-robust, testable) -----------------

def _resolve_diarization_pipeline():
    """whisperx 3.2+ moved DiarizationPipeline to whisperx.diarize; older versions
    had it at the top level. Try the new location first."""
    try:
        from whisperx.diarize import DiarizationPipeline
        return DiarizationPipeline
    except Exception:       # noqa: BLE001
        from whisperx import DiarizationPipeline   # old API
        return DiarizationPipeline


def diarization_turns(diar) -> list[tuple[float, float, str]]:
    """Normalise a diarization result into (start, end, speaker) turns. Handles:
    a pandas DataFrame (whisperx; columns start/end/speaker), a pyannote Annotation
    (`.itertracks(yield_label=True)`), and iterables of dicts/tuples."""
    turns: list[tuple[float, float, str]] = []
    # pandas DataFrame (whisperx DiarizationPipeline output)
    if hasattr(diar, "iterrows"):
        for _, row in diar.iterrows():
            turns.append((float(row["start"]), float(row["end"]), str(row["speaker"])))
        return turns
    # pyannote Annotation
    if hasattr(diar, "itertracks"):
        for segment, _track, label in diar.itertracks(yield_label=True):
            turns.append((float(segment.start), float(segment.end), str(label)))
        return turns
    # plain iterable of dicts / tuples
    for item in diar or []:
        if isinstance(item, dict):
            turns.append((float(item["start"]), float(item["end"]),
                          str(item.get("speaker", item.get("label")))))
        else:
            s, e, spk = item[0], item[1], item[2]
            turns.append((float(s), float(e), str(spk)))
    return turns


def _best_speaker(turns, start: float, end: float, fill_nearest: bool):
    best, best_ov = None, 0.0
    for ts, te, spk in turns:
        ov = min(end, te) - max(start, ts)
        if ov > best_ov:
            best_ov, best = ov, spk
    if best is None and fill_nearest and turns:
        mid = (start + end) / 2.0
        best = min(turns, key=lambda t: abs((t[0] + t[1]) / 2.0 - mid))[2]
    return best


def assign_speakers(result: dict, turns: list[tuple[float, float, str]],
                    fill_nearest: bool = True) -> dict:
    """Assign a speaker to each word (and segment) by maximum time-overlap with the
    diarization turns, falling back to the nearest turn. Mutates and returns
    `result` (a whisperx-style {"segments": [{"words": [...]}, ...]}). With no turns
    it leaves the result unchanged (single-speaker)."""
    if not turns:
        return result
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            s, e = w.get("start"), w.get("end")
            if s is None or e is None:
                continue
            spk = _best_speaker(turns, float(s), float(e), fill_nearest)
            if spk is not None:
                w["speaker"] = spk
        s, e = seg.get("start"), seg.get("end")
        if s is not None and e is not None:
            spk = _best_speaker(turns, float(s), float(e), fill_nearest)
            if spk is not None:
                seg["speaker"] = spk
    return result


_AUTH_MARKERS = ("401", "403", "gated", "unauthor", "authenticat", "access",
                 "agree", "terms", "licen", "accept", "private", "token",
                 "huggingface", "hf_token", "use_auth_token")


def _is_auth_error(e: Exception) -> bool:
    return any(m in str(e).lower() for m in _AUTH_MARKERS)


def _diar_failure_message(e: Exception) -> str:
    """Classify a diarization failure so the user knows which knob to turn."""
    if _is_auth_error(e):
        return ("Diarization auth/licence error — falling back to single-speaker. "
                "Set HF_TOKEN in .env AND accept the model licence (one-time) on "
                "huggingface.co for BOTH pyannote/speaker-diarization-community-1 and "
                "pyannote/segmentation-3.0. "
                f"Details: {type(e).__name__}: {e}")
    return ("Diarization failed (likely a whisperx/pyannote version mismatch) — "
            "falling back to single-speaker. Pin the tested versions in "
            "requirements-gameplay.txt. "
            f"Details: {type(e).__name__}: {e}")


def _diarize(audio, token: str, dev: str):
    """Run pyannote diarization and return (start,end,speaker) turns. Raises on
    failure (the caller classifies + degrades to single-speaker)."""
    DiarizationPipeline = _resolve_diarization_pipeline()
    try:                                  # whisperx 3.8 uses token=
        diarizer = DiarizationPipeline(token=token, device=dev)
    except TypeError:                     # older whisperx used use_auth_token=
        diarizer = DiarizationPipeline(use_auth_token=token, device=dev)
    diar = diarizer(audio, min_speakers=gconf.DIARIZE_MIN_SPEAKERS,
                    max_speakers=gconf.DIARIZE_MAX_SPEAKERS)
    return diarization_turns(diar)


# ---- main entry points ------------------------------------------------------

def _emit(progress: Progress | None, msg: str) -> None:
    if progress:
        progress(msg)


def transcribe(audio_or_video: str | Path, progress: Progress | None = None,
               diarize: bool = True) -> Transcript:
    """Transcribe + word-align + optionally diarize `audio_or_video`.

    Returns a Transcript. Falls back to single-speaker when there's no HF token,
    pyannote finds <=1 voice, or diarization errors. Raises FriendlyError for the
    real failure modes (missing dep, GPU OOM, no speech)."""
    path = str(Path(audio_or_video))
    try:
        import whisperx
    except ImportError:
        raise FriendlyError(
            "WhisperX is not installed. Install the gameplay extras:\n"
            "    pip install -r requirements-gameplay.txt\n"
            "(GPU build needs the CUDA-matched torch — see the README).")

    plan = device_mod.plan_device()
    if plan.warning:
        _emit(progress, plan.warning)
    _emit(progress, f"Loading WhisperX ({plan.describe()})...")
    try:
        model = whisperx.load_model(plan.model, plan.device,
                                    compute_type=plan.compute_type)
        audio = whisperx.load_audio(path)
        _emit(progress, "Transcribing...")
        result = model.transcribe(audio, batch_size=gconf.WHISPERX_BATCH)
    except Exception as e:                       # noqa: BLE001 — re-raise as friendly
        raise _friendly_transcribe_error(e)

    if not result.get("segments"):
        raise FriendlyError(
            "No speech detected in the clip. Check the clip has an audible voice "
            "track (gameplay music-only clips won't caption).")

    _emit(progress, "Aligning words to the audio...")
    try:
        align_model, metadata = whisperx.load_align_model(
            language_code=result["language"], device=plan.device)
        result = whisperx.align(result["segments"], align_model, metadata, audio,
                                plan.device, return_char_alignments=False)
    except Exception as e:                       # noqa: BLE001
        raise _friendly_transcribe_error(e)

    token = gconf.hf_token()
    if diarize and token:
        _emit(progress, "Diarizing speakers (pyannote)...")
        try:
            turns = _diarize(audio, token, plan.device)
            result = assign_speakers(result, turns)
            n = len({t[2] for t in turns})
            _emit(progress, f"Diarization: {n} speaker(s) over {len(turns)} turn(s).")
        except Exception as e:                   # noqa: BLE001 — degrade gracefully
            _emit(progress, _diar_failure_message(e))
    elif diarize and not token:
        _emit(progress, "No HF_TOKEN set — single-speaker captions "
                        "(set HF_TOKEN in .env + accept the pyannote licence to "
                        "colour per speaker).")

    transcript = from_whisperx(result)
    n = len(transcript.speakers)
    _emit(progress, f"Done: {len(transcript.words)} words, "
                    f"{'single speaker' if transcript.single_speaker else f'{n} speakers'}.")
    return transcript


def transcribe_clip(clip: GameplayClip, progress: Progress | None = None,
                    force: bool = False, diarize: bool = True) -> Transcript:
    """Transcribe a clip's source and cache the result to transcript.json
    (resumable — won't re-run unless forced)."""
    if clip.has_transcript() and not force:
        _emit(progress, "Using cached transcript.")
        return Transcript.load(clip.transcript_path)
    src = clip.source_path()
    if src is None:
        raise FriendlyError("No source clip to transcribe.")
    transcript = transcribe(src, progress=progress, diarize=diarize)
    transcript.save(clip.transcript_path)
    return transcript


def _friendly_transcribe_error(e: Exception) -> FriendlyError:
    msg = str(e)
    low = msg.lower()
    if "out of memory" in low or "cuda oom" in low or "cublas" in low:
        return FriendlyError(
            "GPU ran out of memory during transcription. Try a smaller model "
            "(gameplay/config.py: WHISPERX_MODEL_CUDA='medium') or a lower "
            "WHISPERX_BATCH, or transcribe a shorter clip.\n\nDetails: " + msg)
    if "ffmpeg" in low or "winerror 2" in low:
        return FriendlyError(
            "ffmpeg not found on PATH (WhisperX needs it to read audio). Install "
            "it (winget install Gyan.FFmpeg) and restart.")
    return FriendlyError(f"Transcription failed: {type(e).__name__}: {msg}")
