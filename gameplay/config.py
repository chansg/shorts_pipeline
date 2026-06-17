"""Gameplay-pipeline tunables. Kept separate from the lore `config.py` so the two
pipelines never step on each other. Reuses the lore config for the things both
share (output frame size, fps, the bundled caption font dir).
"""
from __future__ import annotations

import os
from pathlib import Path

import config as _lore  # main pipeline config (WIDTH/HEIGHT/FPS/FONTS_DIR/OUTPUT_DIR/ROOT)
from modules.karaoke_captions import DEFAULT_SPEAKER_PALETTE

# --- Paths ---
ROOT = _lore.ROOT
GAMEPLAY_DIR = _lore.OUTPUT_DIR / "gameplay"   # per-clip work + finished Shorts
OVERLAYS_DIR = ROOT / "overlays"               # like/subscribe alpha assets
for _d in (GAMEPLAY_DIR, OVERLAYS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Output format (shared with the lore pipeline) ---
WIDTH = _lore.WIDTH      # 1080
HEIGHT = _lore.HEIGHT    # 1920
FPS = _lore.FPS          # 30
FONTS_DIR = _lore.FONTS_DIR

# --- WhisperX transcribe + diarize ---
# Read at call time so the Settings/env can override without a code edit.
HF_TOKEN_ENV = "HF_TOKEN"          # HuggingFace read token (pyannote diarization)
# Model is chosen by device (see gameplay/device.py): large-v2 needs a GPU; on CPU
# it is unusably slow, so we drop to a small model so the mode stays usable.
WHISPERX_MODEL_CUDA = "large-v2"   # 3080 (10GB) handles large-v2; drop to "medium" if OOM
WHISPERX_MODEL_CPU = "small"       # CPU fallback — large-v2 on CPU is impractical
WHISPERX_MODEL = WHISPERX_MODEL_CUDA   # back-compat alias (the GPU default)
WHISPERX_BATCH = 16                # transcription batch size; lower if GPU OOM
WHISPERX_COMPUTE_CUDA = "float16"  # cuda compute type ("int8" uses less VRAM)
WHISPERX_COMPUTE_CPU = "int8"      # cpu fallback compute type
DIARIZE_MIN_SPEAKERS = 1
DIARIZE_MAX_SPEAKERS = 6           # 4-5 people expected; a little headroom
# ASR anti-hallucination (noisy gameplay audio).
#
# IMPORTANT: WhisperX runs *batched* inference (asr.py:generate_segment_batched).
# The batched decode path consumes ONLY beam_size / patience / length_penalty /
# suppress_* / no_repeat_ngram_size / repetition_penalty. It does NOT use
# temperature fallback, compression_ratio_threshold, log_prob_threshold,
# no_speech_threshold, or condition_on_previous_text — those gate the *sequential*
# whisper path only. So the real levers against a repetition collapse ("Naaaa…"
# screen-wide wall) here are the two below; the others are kept for forward-compat
# and the non-batched fallback but are documented as batched no-ops.
WHISPERX_NO_REPEAT_NGRAM_SIZE = 3   # block any 3-gram from repeating (kills "na na na…")
WHISPERX_REPETITION_PENALTY = 1.15  # >1 penalises repeats; the main anti-loop lever (batched)
WHISPERX_CONDITION_ON_PREVIOUS = False  # batched no-op; correct intent, kept explicit
WHISPERX_NO_SPEECH_THRESHOLD = 0.6      # batched no-op; sequential-path only
WHISPERX_COMPRESSION_RATIO_THRESHOLD = 2.4  # batched no-op; sequential-path only

# VAD (Voice Activity Detection) — WhisperX ALWAYS runs VAD and only transcribes the
# speech regions it finds. If voice is buried under loud game audio, VAD can miss
# whole regions (the "28s gap" dropout). Lower onset = more sensitive (recovers
# missed speech) but risks decoding loud non-speech; tune against your clips.
WHISPERX_VAD_METHOD = "pyannote"   # "pyannote" (default) | "silero"
WHISPERX_VAD_ONSET = 0.50          # speech-start probability threshold (whisperx default 0.500)
WHISPERX_VAD_OFFSET = 0.363        # speech-end probability threshold (whisperx default 0.363)

# Audio prep before WhisperX: force a clean 16k-mono downmix and lift the voice over
# game audio (better VAD + ASR SNR). whisperx.load_audio already downmixes to 16k
# mono, but does NO filtering — these add a high-pass (cut explosion/footstep rumble)
# and EBU loudness-normalise so quiet voice chat isn't swallowed.
WHISPERX_AUDIO_HIGHPASS_HZ = 80    # high-pass cutoff (Hz); 0 disables
WHISPERX_AUDIO_LOUDNORM = True     # EBU R128 loudnorm pass to raise quiet voice

# Hard clamp BEFORE the editable grid: split/clamp any word longer than this (s).
WHISPERX_MAX_WORD_S = 1.2
# Post-guard against runaway tokens: drop/repair any single "word" longer than this
# many characters, or any token that is one character repeated (e.g. "Naaaaaa…").
# Last-line defence so a repetition-collapse token can never reach the grid/captions.
WHISPERX_MAX_WORD_CHARS = 40


def hf_token() -> str | None:
    """The HuggingFace token, or None. None => diarization is skipped and the
    pipeline falls back to single-speaker captions."""
    tok = os.getenv(HF_TOKEN_ENV)
    return tok.strip() if tok and tok.strip() else None


# --- Reframe (9:16 blur-pad) ---
BLUR_RADIUS = 24        # boxblur luma radius for the top/bottom filler
BLUR_BG_BOOST = 1.05    # slightly scale the blurred bg past cover so edges are clean

# --- Captions ---
CAPTION_FONT = _lore.CAPTION_AW_FONT          # bundled Anton by default
CAPTION_FONTSIZE = _lore.CAPTION_AW_FONTSIZE
# Lower than the lore default (0.60): on gameplay footage 0.60 sits on the weapon/
# HUD. 0.78 drops captions into the lower blur band, off the action. GUI slider tunes it.
CAPTION_POS_Y_FRAC = 0.78
SPEAKER_PALETTE = list(DEFAULT_SPEAKER_PALETTE)  # offered in the transcript editor
# Render-side defence in depth (independent of the ASR clamp above):
CAPTION_MAX_EVENT_S = 1.2        # no single caption stays on screen longer than this
CAPTION_MAX_LINE_CHARS = 12      # wrap/hard-split so no line exceeds the frame width

# --- Effects (starter set; the registry in effects.py is built to extend) ---
PUNCH_ZOOM_AMOUNT = 0.08    # 1.0 -> 1.08 push on a beat
PUNCH_ZOOM_SIGMA = 0.12     # seconds; width of each zoom pulse
SHAKE_AMPLITUDE = 8         # pixels of positional jitter at a peak
SHAKE_SIGMA = 0.10          # seconds; width of each shake burst
SHAKE_FREQ = 42.0           # Hz-ish oscillation of the jitter
ENERGY_PEAK_Z = 1.8         # loudness z-score above which a moment counts as a "beat"
ENERGY_WINDOW_S = 0.10      # RMS window for the audio-energy envelope
ENERGY_MAX_PEAKS = 24       # cap so the ffmpeg filter expression stays bounded
ENERGY_MIN_GAP_S = 0.40     # merge peaks closer than this

# --- Overlay defaults ---
OVERLAY_DEFAULT_POSITION = "bottom-center"
OVERLAY_DEFAULT_START = 1.0       # seconds
OVERLAY_DEFAULT_DURATION = 4.0    # seconds (0 / None = whole clip)

# --- Full-auto (experimental) ---
AUTO_LLM_BACKEND = _lore.REWRITE_BACKEND      # reuse "ollama" | "claude" | "none"
AUTO_OLLAMA_MODEL = _lore.OLLAMA_MODEL
AUTO_CATEGORIES = ["clutch", "funny", "rage", "hype", "story"]

# ============================================================================
# Full-auto HIGHLIGHT DETECTION — the calibration surface.
# These are deliberately exposed: the first runs on real VODs are a tuning pass.
# Detection fuses three signals (audio energy + reaction keywords + an LLM judge)
# into ranked candidate windows. Every threshold/weight/lexicon below is tunable.
# ============================================================================

# -- Audio-energy peaks (gameplay/effects.py provides the loudness envelope) --
AUTO_ENERGY_WINDOW_S = 0.5      # RMS window for DETECTION (coarser than effects' 0.1)
AUTO_ENERGY_K = 1.5             # peak threshold = rolling_median + K * rolling_std
AUTO_ENERGY_ROLL_S = 30.0       # seconds; rolling-stats window (local normalisation,
                                # so a loud match isn't drowned by a loud whole-VOD)
AUTO_ENERGY_MIN_PROMINENCE = 0.15  # min (peak − local median), as a fraction of the
                                   # video's max prominence (0..1); rejects small bumps
AUTO_ENERGY_MIN_SPACING_S = 8.0    # peaks must be >= this far apart (no clustering)
AUTO_ENERGY_MAX_ANCHORS = 60       # cap energy anchors before framing

# -- Reaction-keyword lexicon (per category). ADD YOUR GROUP'S SLANG + GAMES. --
# Substring match, case-insensitive, over per-speaker utterances. Multi-word
# phrases are fine. A hit is an anchor carrying this category as a hint.
AUTO_REACTION_LEXICON = {
    "funny": ["lol", "lmao", "lmfao", "haha", "hahaha", "bro", "bruh", "dying",
              "i'm dead", "im dead", "crying", "wheeze", "💀"],
    "hype":  ["let's go", "lets go", "lets gooo", "no way", "no wayyy", "insane",
              "actually insane", "crazy", "sheesh", "poggers", "pog", "w "],
    "clutch": ["clutch", "1v2", "1v3", "1v4", "1v5", "ace", "got him", "got em",
               "defuse", "down", "knifed", "last one"],
    "rage":  ["are you kidding me", "kidding me", "no shot", "what the", "wtf",
              "trash", "broken", "bullshit", "are you serious", "damn it", "rigged"],
}
AUTO_REACTION_DENSITY_CAP = 0.30   # hits/sec that maps to a reaction score of 1.0

# -- Window framing (anchor -> clip window, snapped to transcript boundaries) --
AUTO_CLIP_MIN_S = 15.0          # target clip length floor
AUTO_CLIP_MAX_S = 45.0          # target clip length ceiling
AUTO_LEAD_IN_S = 4.0            # seconds before the anchor (the set-up line)
AUTO_LEAD_OUT_S = 7.0           # seconds after the anchor (the payoff / reaction)

# -- LLM judge (chunked so a 60-min transcript is never sent in one call) --
AUTO_LLM_CHUNK_S = 150.0        # ~2.5 min transcript chunks
AUTO_LLM_CHUNK_OVERLAP_S = 20.0  # overlap so a moment on a boundary isn't split

# -- Fuse + score weights (LLM-led). score is a weighted sum of 0..1 signals. --
AUTO_W_LLM = 0.50               # LLM judge confidence
AUTO_W_ENERGY = 0.25            # normalised audio-energy prominence
AUTO_W_REACTION = 0.15          # reaction-keyword density
AUTO_W_OVERLAP = 0.10           # multi-speaker banter factor (min(speakers,3)/3)
AUTO_TOP_N = 15                 # cap on surviving candidates after dedupe

# -- Long-video hardening (10GB RTX 3080) --
AUTO_TRANSCRIBE_BATCH = 8       # whisperx batch for long videos (safe default)
AUTO_TRANSCRIBE_BATCH_OOM = 4   # retry batch after a CUDA OOM (once)
AUTO_MAX_MINUTES = 60           # warn (don't block) past this length
