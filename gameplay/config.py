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
# Transcription language. WhisperX auto-detects from the first 30s when this is
# None — but on noisy gameplay intros that mis-detects (English audio came out as
# Danish), and a wrong language also loads the wrong alignment model. Pin it so the
# captions are always in the spoken language. Set to a whisper code ("en", "es",
# "de", …) or None to restore auto-detection.
WHISPERX_LANGUAGE = "en"
DIARIZE_MIN_SPEAKERS = 1
DIARIZE_MAX_SPEAKERS = 6           # 4-5 people expected; a little headroom
# Pin the EXACT speaker count (None = auto-detect within [MIN, MAX]). Auto-clustering is
# unreliable on short, cross-talky clips (it mislabels/merges similar voices); when you
# know how many people are talking, telling pyannote the number is the strongest lever.
# Exposed per-transcribe in the GUI ("Speakers" dropdown). NOTE: this cannot separate
# two people talking AT ONCE — single-stream ASR transcribes overlap as one voice; true
# cross-talk separation needs a source-separation stage (see README).
DIARIZE_NUM_SPEAKERS = None
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
# VAD merges contiguous speech into windows of this many seconds, each decoded in
# ONE batched pass. WhisperX's default (30) is too coarse for dense gameplay chatter:
# a long continuous-speech run becomes a single 30s window and the model emits a few
# words then stops (end-of-text), silently dropping the rest — e.g. 26s of talking
# transcribed as 5 words. A smaller window forces several independent decodes and
# recovers the speech (measured on a real clip: 30s->5 words, 8s->41 words). Lower =
# more coverage but more fragmentation; ~6-10 is the sweet spot for noisy game audio.
# Measured on a dense 28.6s ARAM clip: at 8 the batched decoder gave up inside covered
# chunks (only 30 words despite 92% VAD coverage); 6 recovered 64 words (2.1x), 4 -> 49.
WHISPERX_CHUNK_SIZE = 6

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


# --- Output encode quality (shared final-encode helper, gameplay/encode.py) ---
# The FINAL encode is constant-quality (CRF), not a fixed low bitrate. CRF 18 is
# visually near-lossless for 1080x1920 high-motion gameplay and lands well above
# 10 Mbps on its own (16 for more headroom). A slower preset is worth it — these
# are short clips on a fast GPU box. Intermediates are near-lossless (CRF 14) so
# only the final CRF governs quality (no compounding loss across passes).
OUTPUT_CRF = 18              # final-encode constant-quality target (lower = better)
OUTPUT_PRESET = "slow"       # final-encode x264 preset (slow/medium; short clips)
OUTPUT_PROFILE = "high"      # h264 profile — phone/browser safe
INTERMEDIATE_CRF = 14        # near-lossless intermediates (reframe, etc.)
INTERMEDIATE_PRESET = "medium"

# --- Reframe (9:16 layout) ---
# How the 16:9 source is fit into 1080x1920:
#   "fill" (DEFAULT, recommended) — cover + crop so the gameplay fills the whole frame
#                          at full resolution (sharpest, no wasted blur bars). Loses the
#                          far horizontal edges; bias the crop with REFRAME_CROP_X_OFFSET
#                          (for ARAM the fight is centre; nudge to keep the minimap).
#   "fit_crop"           — fill at fraction 1.0, centred (alias of the old behaviour).
#   "blur_pad"           — full frame centred over a blurred fill (no crop, but a ~16:9
#                          strip carries the action; most pixels are blur).
#   "zoom_blur"          — blur-pad with the centred gameplay band scaled up.
#   "tall" (DEFAULT)     — a FULL-WIDTH gameplay band, uniformly scaled (NO stretch), that
#                          fills REFRAME_TALL_HEIGHT_FRAC of the 1920 height with a thin
#                          blurred frame top/bottom. Uses much more vertical space than
#                          blur_pad while keeping MORE horizontal context than full-crop
#                          fill — the sweet spot for ARAM's wide, busy top-down action
#                          (big readable detail without cropping the mayhem to the centre).
REFRAME_MODE = "tall"        # "tall" | "fill" | "fit_crop" | "blur_pad" | "zoom_blur"
# tall: fraction of the 1920 height the gameplay band fills (full width, no stretch).
# Higher = more vertical / more side-crop; lower = more width kept / thicker blur frame.
REFRAME_TALL_HEIGHT_FRAC = 0.82
REFRAME_FILL_FRACTION = 1.0  # fill: zoom past cover (>=1.0; 1.0 = just fills the frame)
REFRAME_CROP_X_OFFSET = 0.5  # fill/tall: horizontal crop bias (0=left, 0.5=centre, 1=right)
REFRAME_CROP_Y_OFFSET = 0.5  # fill/tall: vertical crop bias (0=top, 0.5=centre, 1=bottom)
ZOOM_BLUR_SCALE = 1.4        # zoom_blur: enlarge the centred gameplay band by this factor
BLUR_RADIUS = 24        # boxblur luma radius for the top/bottom filler
BLUR_BG_BOOST = 1.05    # slightly scale the blurred bg past cover so edges are clean

# --- Captions ---
CAPTION_FONT = _lore.CAPTION_AW_FONT          # bundled Anton by default
CAPTION_FONTSIZE = _lore.CAPTION_AW_FONTSIZE
# Lower than the lore default (0.60): the "fill" layout puts gameplay across the whole
# frame, so captions sit on a readable lower band. 0.72 keeps them ABOVE the branded
# like/subscribe banner (centred ~0.84), so caption and banner never overlap. GUI
# slider tunes it per build.
CAPTION_POS_Y_FRAC = 0.72
SPEAKER_PALETTE = list(DEFAULT_SPEAKER_PALETTE)  # offered in the transcript editor
# When diarization finds no/one speaker, seed this many default SPEAKER_NN rows in
# the colour grid (palette hex) so it's never empty and the user has starter colours.
DEFAULT_SPEAKER_ROWS = 5
# Render-side defence in depth (independent of the ASR clamp above):
CAPTION_MAX_EVENT_S = 1.2        # no single caption stays on screen longer than this
CAPTION_MAX_LINE_CHARS = 12      # wrap/hard-split so no line exceeds the frame width

# --- Caption timing/chunking (gameplay/captioning.py) ---
# Strict one-word-at-a-time karaoke magnifies any per-word ASR drift and reads as
# broken when words are sparse. "phrase" mode groups a few words into one cue that
# holds for the span of its words — far more forgiving of drift and easier to read on
# a phone. "word" keeps the one-word karaoke. Gameplay default is "phrase"; lore is
# unaffected.
CAPTION_CHUNK_MODE = "phrase"       # "phrase" (default) | "word"
CAPTION_CHUNK_MAX_WORDS = 4         # max words per phrase cue
CAPTION_CHUNK_MAX_WINDOW_S = 1.2    # max time span a phrase cue covers (s)
CAPTION_CHUNK_MAX_CHARS = 22        # start a new cue past this many chars
CAPTION_MIN_DUR_S = 0.4             # minimum on-screen time per cue (anti-flash)
CAPTION_MAX_GAP_S = 0.8             # bridge gaps smaller than this (don't blink off)
CAPTION_OFFSET_S = 0.0             # global lead(-)/lag(+) nudge for residual drift (s)

# --- Profanity censor (gameplay/censor.py) ---
# Bleep/mute the audio AND mask the caption on curse words, driven off the WhisperX
# word spans (a censored word is the same (text,start,end) the captions use). A word
# is censored (case-insensitive) when its bare token is in CENSOR_WORDLIST OR contains
# a CENSOR_STEM as a substring — UNLESS it's in CENSOR_ALLOWLIST. The stems catch
# variants/compounds automatically (fucking, bullshit, wankers, dipshit, motherf...)
# so far fewer words need a manual tick; the allow-list guards the false positives the
# stems would otherwise hit (Scunthorpe, niggle, fire-retardant, …). Deterministic.
# To KEEP a flagged word uncensored, add it to CENSOR_ALLOWLIST (unticking a profane
# word in the editor no longer keeps it — auto-detection wins; the editor tick only
# ADDS censor to a word the lists don't already catch).
CENSOR_ENABLED = True            # master switch for the whole feature
CENSOR_AUDIO = True              # censor the audio over each hit span
CENSOR_CAPTION = True            # mask the matching word in the burned caption
CENSOR_AUDIO_MODE = "bleep"      # "bleep" (1kHz tone) | "mute" (silence) | "duck" (quieten)
CENSOR_CAPTION_STYLE = "stars"   # "stars" -> f***  |  "block" -> [bleep]
CENSOR_PAD_S = 0.05              # widen each hit span by this (s) so onsets aren't clipped
CENSOR_BLEEP_HZ = 1000           # bleep tone frequency
CENSOR_BLEEP_GAIN = 0.3          # bleep tone amplitude (0..1)
CENSOR_DUCK_GAIN = 0.2           # "duck" mode: span volume multiplier
# Substrings that flag a token wherever they appear (the sensitivity lever). Chosen so
# substring matching is safe; the rare clean word they'd hit is in CENSOR_ALLOWLIST.
CENSOR_STEMS = [
    "fuck", "shit", "bitch", "cunt", "slut", "whore", "wank", "nigg", "fagg",
    "retard", "douche", "jizz", "twat",
]
CENSOR_WORDLIST = [              # whole words that have no safe stem above
    "ass", "asshole", "assholes", "dumbass", "jackass", "smartass", "badass",
    "bastard", "dick", "dickhead", "cock", "piss", "pissed", "prick", "bollocks",
    "arse", "arsehole", "damn", "goddamn", "crap", "bugger", "wtf", "stfu", "fag",
]
CENSOR_ALLOWLIST = [             # whole words that must NEVER be censored
    "shaco", "assassin", "assassins", "cassiopeia", "scunthorpe", "class",
    "pass", "bass", "grass", "dictionary", "cockpit", "shitake",
    "niggle", "niggling", "retardant", "class", "compass", "bypass",
]

# --- Narrated hook (gameplay/hook.py) ---
# Read an opening hook line aloud (ElevenLabs) over the start of the Short — the
# TikTok "story-time" format. Per-build toggle (default off); the game audio ducks
# under the narration and swells back when it ends. Reuses the lore ElevenLabs client.
NARRATED_HOOK_ENABLED = False    # documents the default; the real control is per-build
HOOK_VOICE = _lore.ELEVENLABS_VOICE_ID   # default = the lore pipeline's voice
HOOK_LEAD_IN_S = 0.0             # tiny pad before the narration starts
DUCK_LEVEL = 0.25               # (legacy duck level; superseded by mute+muffle below)
DUCK_RELEASE_S = 0.3            # ramp back to full over this many seconds after the line
NARRATOR_CAPTION_COLOR = (0, 229, 255)   # reserved cyan — the hook caption's colour
# While she speaks the game bed is MUTED + MUFFLED (not merely ducked) so the narration
# is clean, then it swells back over DUCK_RELEASE_S. Game-speech captions are also
# suppressed during the narration so nothing overlaps the hook caption on screen.
HOOK_MUTE_GAIN = 0.05            # near-silent bed gain under the narration (0..1)
HOOK_MUFFLE_HZ = 500             # low-pass cutoff applied to the bed under the narration

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

# --- Overlay defaults (branded like/subscribe bottom banner) ---
# The shipped asset is a wide 3.18:1 call-to-action bar (LIKE + ASHEN CHAN + SUBSCRIBE
# + bell), scaled to OVERLAY_WIDTH_FRAC of the frame width (aspect preserved) and
# centred horizontally at OVERLAY_POS_Y_FRAC (the banner's vertical CENTRE). It sits in
# a low band that clears the caption band — captions were moved up to 0.72 (above) so
# the two never collide. The placeholder remains in overlays/ but is no longer default.
LIKE_SUB_OVERLAY = "like_subscribe_overlay.png"   # default branded banner in overlays/
OVERLAY_WIDTH_FRAC = 0.85         # banner width as a fraction of the 1080px frame (~918px -> ~288px tall)
OVERLAY_POS_Y_FRAC = 0.84         # banner vertical CENTRE (0=top, 1=bottom); ~0.76-0.92 band
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

# ============================================================================
# Full-auto HIGHLIGHT CLIPS — audio-reaction-led detection -> ranked 9:16 raw
# candidates for manual refinement (the 3rd-button flow). No transcript/LLM needed:
# AUDIO REACTION LEADS (robust, streamed) and HUD events CONFIRM (a score booster
# that is allowed to fail). Distinct from the LLM/16:9 YouTube path above.
#
# Detection WILL need calibration on real footage — the run logs the score curve.
# If candidates are sparse: lower REACTION_THRESHOLD. If noisy/teamfights leak in:
# raise REACTION_ONSET_WEIGHT (favour sharp exclamations over sustained roar) or
# REACTION_THRESHOLD. Widen the clips with PRE_ROLL_S / POST_ROLL_S.
# ============================================================================
REACTION_WINDOW_S = 0.1            # per-window analysis resolution (s)
REACTION_BAND_HZ = (300, 3400)     # vocal band-pass applied before scoring
REACTION_BASELINE_WINDOW_S = 8.0   # rolling baseline (fire on SUDDEN vs recent chatter)
REACTION_ONSET_WEIGHT = 0.65       # attack-sharpness vs sustained-energy weight (0..1)
REACTION_THRESHOLD = 0.35          # min reaction score (0..1 vs baseline) for a peak
REACTION_MIN_SPACING_S = 6.0       # one reaction = one peak (min peak spacing)
REACTION_MAX_PEAKS = 80            # cap peaks before windowing
REACTION_SR = 16000                # PCM sample rate for streamed analysis
REACTION_BLOCK_SAMPLES = 1 << 20   # PCM stream block size (bounded memory, ~1M samples)

PRE_ROLL_S = 8.0                   # generous setup BEFORE the spike (anchor before it)
POST_ROLL_S = 10.0                 # payoff AFTER the spike
MERGE_GAP_S = 3.0                  # merge windows within this gap into one candidate
MAX_CANDIDATES = 20                # keep the highest-scoring N (recall-biased)

# Path to the Tesseract OCR binary (HUD/ARAM multikill detection reads the banner with
# it). PATH is used when this resolves to no file, so a machine with tesseract on PATH
# still works. Env var TESSERACT_CMD overrides; default = the common Windows AppData
# install. OCR is optional — when neither resolves, OCR features degrade with a message.
TESSERACT_CMD = os.environ.get(
    "TESSERACT_CMD",
    r"C:\Users\chansg\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
)

# -- HUD event scan (League-specific, brittle; isolated + fail-safe booster) --
HUD_SCAN_ENABLED = True            # master switch; off -> audio-only candidates
HUD_SAMPLE_FPS = 1.0               # frames/sec sampled WITHIN a candidate window only
HUD_BOOST_CAP = 1.5                # max total boost (final = audio_score * (1 + boost))
HUD_EVENT_WEIGHTS = {              # boost contribution per detected HUD event kind
    "pentakill": 1.0, "quadrakill": 0.8, "triplekill": 0.6, "doublekill": 0.4,
    "ace": 0.7, "shutdown": 0.4, "kill": 0.25, "death": 0.3, "executed": 0.3}
# 1080p HUD ROIs as (x, y, w, h) fractions of the frame — kill feed (top-right) and
# the centre multikill/ace banner. Resolution-dependent; tune per capture layout.
HUD_ROIS = {
    "killfeed": (0.78, 0.18, 0.22, 0.30),
    "banner":   (0.30, 0.20, 0.40, 0.18),
}
HUD_EVENT_LEXICON = {              # recognised HUD text (substring) -> canonical event
    "penta kill": "pentakill", "pentakill": "pentakill",
    "quadra kill": "quadrakill", "quadrakill": "quadrakill",
    "triple kill": "triplekill", "triplekill": "triplekill",
    "double kill": "doublekill", "doublekill": "doublekill",
    "ace": "ace", "shut down": "shutdown", "shutdown": "shutdown",
    "has slain": "kill", "slain an enemy": "kill", "you have slain": "kill",
    "executed": "executed", "have been slain": "death", "was slain": "death",
}

# ============================================================================
# Game-mode preset. "aram" (League of Legends ARAM) makes MULTIKILLS the primary
# candidate driver instead of audio reactions: the WHOLE clip is sampled for the centre
# multikill / ace banner, the escalating banners of one fight (Double -> Triple -> ... ->
# Penta) collapse into ONE streak reported at its TOP tier, and each streak at/above
# ARAM_MIN_MULTIKILL (default triple) becomes a candidate. A loud voice reaction inside
# the window breaks ties. "generic" keeps the audio-reaction-led detection unchanged.
# ============================================================================
GAME_MODE = "generic"               # "generic" | "aram"
ARAM_TIERS = ["doublekill", "triplekill", "quadrakill", "pentakill"]   # ascending
ARAM_MIN_MULTIKILL = "triplekill"   # lowest tier that anchors a candidate (infer triple+)
ARAM_INCLUDE_ACE = True             # also anchor on team Aces (an ARAM money moment)
ARAM_STREAK_GAP_S = 8.0             # banners within this gap = one escalating streak
ARAM_PRE_ROLL_S = 10.0             # fight build-up kept BEFORE the streak
ARAM_POST_ROLL_S = 6.0             # celebration kept AFTER the last banner
ARAM_SCAN_FPS = 0.5                # whole-clip banner sample rate (banner persists ~3-4s)
