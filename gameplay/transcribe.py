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

# Fixed name for the cleaned 16k-mono work audio, written into the clip dir. MUST
# NOT match the `source.*` glob in state.GameplayClip.source_path, or reframe/build
# would treat it as the source video. Leading underscore => sorts/reads as a work file.
PREP_AUDIO_NAME = "_audio16k.wav"


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


def probe_audio(path: str | Path) -> tuple[int, int]:
    """(sample_rate, channels) of the first audio stream via ffprobe; (0, 0) if
    unknown. Used to log what actually reaches the model and to assert the prep
    step produced 16k mono."""
    for s in _probe_streams(Path(path)):
        if s.get("codec_type") == "audio":
            try:
                return int(s.get("sample_rate", 0)), int(s.get("channels", 0))
            except (TypeError, ValueError):
                return 0, 0
    return 0, 0


def _audio_filter_chain() -> str:
    """ffmpeg -af chain for the prep step: high-pass (cut explosion/footstep rumble)
    then EBU loudnorm (raise quiet voice chat over loud game audio). Either can be
    disabled in config. Empty string => no -af (plain downmix only)."""
    parts: list[str] = []
    if gconf.WHISPERX_AUDIO_HIGHPASS_HZ and gconf.WHISPERX_AUDIO_HIGHPASS_HZ > 0:
        parts.append(f"highpass=f={int(gconf.WHISPERX_AUDIO_HIGHPASS_HZ)}")
    if gconf.WHISPERX_AUDIO_LOUDNORM:
        parts.append("loudnorm=I=-16:TP=-1.5:LRA=11")
    return ",".join(parts)


def prepare_audio(src: str | Path, dest: str | Path) -> Path:
    """Produce a clean 16k MONO wav for WhisperX.

    whisperx.load_audio already downmixes to 16k mono but applies NO filtering. On
    loud-game-audio clips the voice is buried after a flat channel-average, so VAD
    misses speech (dropout) and ASR collapses. This adds an explicit stereo->mono
    downmix plus a high-pass + loudness-normalise so the voice survives. Feed the
    result to whisperx.load_audio (a no-op resample) so VAD/ASR/diarization all see
    the cleaned audio."""
    from modules.assemble import _run
    src, dest = Path(src), Path(dest)
    cmd = ["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000"]
    chain = _audio_filter_chain()
    if chain:
        cmd += ["-af", chain]
    cmd += ["-c:a", "pcm_s16le", str(dest)]
    _run(cmd)
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


def _is_oom(e: Exception) -> bool:
    low = str(e).lower()
    return ("out of memory" in low or "cuda oom" in low or "cublas" in low
            or "alloc" in low and "cuda" in low)


def _transcribe_with_oom_retry(model, audio, batch: int, oom_batch: int,
                               progress: Progress | None,
                               chunk_size: int | None = None):
    """Run ASR; on a CUDA OOM, free VRAM and retry ONCE at a smaller batch with a
    clear log (rather than crashing). A second OOM propagates to a friendly error.
    `chunk_size` is the VAD-merge window (seconds) — small enough that dense
    continuous speech is decoded in several passes instead of one giant window that
    the model abandons after a few words (defaults to gconf.WHISPERX_CHUNK_SIZE)."""
    cs = chunk_size if chunk_size is not None else gconf.WHISPERX_CHUNK_SIZE
    try:
        return model.transcribe(audio, batch_size=batch, chunk_size=cs)
    except Exception as e:                       # noqa: BLE001
        if _is_oom(e) and oom_batch < batch:
            device_mod.free_vram()
            _emit(progress, f"⚠ CUDA OOM at batch={batch}; freed VRAM, retrying "
                            f"once at batch={oom_batch}...")
            return model.transcribe(audio, batch_size=oom_batch, chunk_size=cs)
        raise


def _load_model(whisperx, plan, asr_options: dict, vad_options: dict):
    """Load the WhisperX model, passing our ASR + VAD options. Older/other whisperx
    builds may not accept every kwarg, so progressively drop the optional ones
    (vad first, then asr_options) rather than failing — the bare load always works."""
    attempts = (
        dict(compute_type=plan.compute_type, asr_options=asr_options,
             vad_method=gconf.WHISPERX_VAD_METHOD, vad_options=vad_options),
        dict(compute_type=plan.compute_type, asr_options=asr_options),
        dict(compute_type=plan.compute_type),
    )
    last_err: Exception | None = None
    for kw in attempts:
        try:
            return whisperx.load_model(plan.model, plan.device, **kw)
        except TypeError as e:       # unsupported kwarg on this whisperx build
            last_err = e
            continue
    raise last_err or RuntimeError("whisperx.load_model failed")


def transcribe(audio_or_video: str | Path, progress: Progress | None = None,
               diarize: bool = True, batch_size: int | None = None) -> Transcript:
    """Transcribe + word-align + optionally diarize `audio_or_video`.

    Long-video safe on a 10GB card: the ASR model's VRAM is released before the
    alignment model loads, and again before diarization, so the card never holds
    two models at once. `batch_size` overrides the default (full-auto passes a
    smaller one); a CUDA OOM retries once at a smaller batch.

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
    batch = int(batch_size or gconf.WHISPERX_BATCH)
    _emit(progress, f"Loading WhisperX ({plan.describe()}, batch={batch})...")
    # Anti-repetition ASR options. WhisperX runs BATCHED inference, whose decode path
    # only honours no_repeat_ngram_size / repetition_penalty (the rest are kept for
    # forward-compat / the non-batched fallback — see gameplay/config.py). VAD is
    # always on; we pass its thresholds so loud non-speech is skipped and quiet
    # speech isn't. Both fall back gracefully if this whisperx build is older.
    asr_options = {
        "condition_on_previous_text": gconf.WHISPERX_CONDITION_ON_PREVIOUS,
        "no_speech_threshold": gconf.WHISPERX_NO_SPEECH_THRESHOLD,
        "compression_ratio_threshold": gconf.WHISPERX_COMPRESSION_RATIO_THRESHOLD,
        "no_repeat_ngram_size": gconf.WHISPERX_NO_REPEAT_NGRAM_SIZE,
        "repetition_penalty": gconf.WHISPERX_REPETITION_PENALTY,
    }
    vad_options = {"vad_onset": gconf.WHISPERX_VAD_ONSET,
                   "vad_offset": gconf.WHISPERX_VAD_OFFSET}
    try:
        model = _load_model(whisperx, plan, asr_options, vad_options)
        # Clean 16k-mono prep (downmix + high-pass + loudnorm) so VAD finds the
        # speech and ASR has SNR — then load_audio (a no-op resample) for the model.
        src_sr, src_ch = probe_audio(path)
        # Write next to the source under a FIXED name (PREP_AUDIO_NAME) that can't
        # match the clip's `source.*` glob (state.GameplayClip.source_path) —
        # otherwise the prepped wav is picked up as the "source" by reframe/build and
        # re-prepped each run (source.16k.16k.wav…). Overwritten each run; no accumulation.
        prepped = prepare_audio(path, Path(path).parent / PREP_AUDIO_NAME)
        out_sr, out_ch = probe_audio(prepped)
        _emit(progress, f"Audio: source {src_sr or '?'}Hz/{src_ch or '?'}ch → "
                        f"{out_sr or 16000}Hz/{'mono' if out_ch == 1 else f'{out_ch}ch'} "
                        f"(high-pass {gconf.WHISPERX_AUDIO_HIGHPASS_HZ}Hz, "
                        f"loudnorm {'on' if gconf.WHISPERX_AUDIO_LOUDNORM else 'off'}).")
        _emit(progress, f"VAD: {gconf.WHISPERX_VAD_METHOD} on "
                        f"(onset={gconf.WHISPERX_VAD_ONSET}, offset={gconf.WHISPERX_VAD_OFFSET}, "
                        f"chunk={gconf.WHISPERX_CHUNK_SIZE}s); "
                        f"repetition guard no_repeat_ngram={gconf.WHISPERX_NO_REPEAT_NGRAM_SIZE}, "
                        f"penalty={gconf.WHISPERX_REPETITION_PENALTY}.")
        audio = whisperx.load_audio(str(prepped))
        _emit(progress, "Transcribing...")
        result = _transcribe_with_oom_retry(model, audio, batch,
                                            gconf.AUTO_TRANSCRIBE_BATCH_OOM, progress,
                                            gconf.WHISPERX_CHUNK_SIZE)
    except Exception as e:                       # noqa: BLE001 — re-raise as friendly
        raise _friendly_transcribe_error(e)
    # release the ASR model's VRAM before the alignment model loads (10GB card)
    del model
    device_mod.free_vram()

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
    # release the alignment model before diarization loads its model
    del align_model, metadata
    device_mod.free_vram()

    token = gconf.hf_token()
    diarized_ran = False
    if diarize and token:
        _emit(progress, "Diarizing speakers (pyannote)...")
        try:
            turns = _diarize(audio, token, plan.device)
            result = assign_speakers(result, turns)
            diarized_ran = True
            n = len({t[2] for t in turns})
            _emit(progress, f"Diarization: {n} speaker(s) over {len(turns)} turn(s).")
        except Exception as e:                   # noqa: BLE001 — degrade gracefully
            _emit(progress, _diar_failure_message(e))
        finally:
            device_mod.free_vram()
    elif diarize and not token:
        _emit(progress, "No HF_TOKEN set — single-speaker captions "
                        "(set HF_TOKEN in .env + accept the pyannote licence to "
                        "colour per speaker).")

    transcript = from_whisperx(result, max_word_s=gconf.WHISPERX_MAX_WORD_S,
                               diarized=diarized_ran,
                               max_word_chars=gconf.WHISPERX_MAX_WORD_CHARS)
    n = len(transcript.speakers)
    if diarized_ran and transcript.single_speaker:
        _emit(progress, f"Done: {len(transcript.words)} words. Diarization ran but "
                        f"collapsed to one dominant speaker.")
    else:
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
