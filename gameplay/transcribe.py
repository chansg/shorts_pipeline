"""WhisperX transcribe + word-align + (optional) pyannote diarization.

One pass yields word-level timestamps and speaker labels for 4-5 people. Runs on
GPU (CUDA) when available, falls back to CPU. Diarization needs a free HuggingFace
token (HF_TOKEN in .env); without it — or if only one voice is found — the result
falls back gracefully to single-speaker (no per-speaker colour).

WhisperX/torch are heavy and GPU-specific, so they are imported lazily inside the
functions: the rest of the gameplay package (and its tests) import and run without
them installed. Install via `pip install -r requirements-gameplay.txt`.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from orchestrator.errors import FriendlyError
from gameplay import config as gconf
from gameplay.state import GameplayClip
from gameplay.transcript import Transcript, from_whisperx

Progress = Callable[[str], None]


def import_source(src: str | Path, name: str | None = None) -> GameplayClip:
    """Copy an uploaded clip into its work dir under output/gameplay/<name>/ and
    return the GameplayClip. Idempotent (won't recopy if a source already exists)."""
    src = Path(src)
    clip = GameplayClip(name or src.stem)
    if not clip.has_source():
        dest = clip.dir / f"source{src.suffix.lower()}"
        shutil.copy2(src, dest)
    return clip


def _device_and_compute() -> tuple[str, str]:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", gconf.WHISPERX_COMPUTE_CUDA
    except Exception:
        pass
    return "cpu", gconf.WHISPERX_COMPUTE_CPU


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

    device, compute_type = _device_and_compute()
    _emit(progress, f"Loading WhisperX ({gconf.WHISPERX_MODEL}) on {device}...")
    try:
        model = whisperx.load_model(gconf.WHISPERX_MODEL, device,
                                    compute_type=compute_type)
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
            language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], align_model, metadata, audio,
                                device, return_char_alignments=False)
    except Exception as e:                       # noqa: BLE001
        raise _friendly_transcribe_error(e)

    token = gconf.hf_token()
    if diarize and token:
        try:
            _emit(progress, "Diarizing speakers (pyannote)...")
            diarizer = whisperx.DiarizationPipeline(use_auth_token=token,
                                                    device=device)
            diar = diarizer(audio, min_speakers=gconf.DIARIZE_MIN_SPEAKERS,
                            max_speakers=gconf.DIARIZE_MAX_SPEAKERS)
            result = whisperx.assign_word_speakers(diar, result)
        except Exception as e:                   # noqa: BLE001 — degrade gracefully
            _emit(progress, f"Diarization failed ({type(e).__name__}); "
                            f"falling back to single-speaker.")
    elif diarize and not token:
        _emit(progress, "No HF_TOKEN set — single-speaker captions "
                        "(set HF_TOKEN in .env to colour per speaker).")

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
            "(gameplay/config.py: WHISPERX_MODEL='medium') or a lower "
            "WHISPERX_BATCH, or transcribe a shorter clip.\n\nDetails: " + msg)
    if "ffmpeg" in low or "winerror 2" in low:
        return FriendlyError(
            "ffmpeg not found on PATH (WhisperX needs it to read audio). Install "
            "it (winget install Gyan.FFmpeg) and restart.")
    return FriendlyError(f"Transcription failed: {type(e).__name__}: {msg}")
