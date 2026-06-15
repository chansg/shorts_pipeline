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
AUTO_CLIP_MIN_S = 6.0
AUTO_CLIP_MAX_S = 45.0
AUTO_ENERGY_TOP_N = 30        # candidate windows from the audio-energy pass
AUTO_CATEGORIES = ["clutch", "funny", "rage", "story"]
