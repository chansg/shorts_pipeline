"""
Central configuration for the Shorts pipeline.
Everything tunable lives here so the modules stay clean.
"""
from pathlib import Path

# Load variables from a local .env file (e.g. ELEVENLABS_API_KEY) into the
# environment. We point at the .env next to THIS file, so it loads no matter
# what the current working directory is (PyCharm run configs often differ).
# Requires: pip install python-dotenv
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# --- Paths ---
ROOT = Path(__file__).parent
SCRIPTS_DIR = ROOT / "scripts"        # raw story seeds + rewritten scripts
IMAGES_DIR = ROOT / "assets" / "images"
MUSIC_DIR = ROOT / "assets" / "music"
CUTAWAY_DIR = ROOT / "assets" / "cutaways"
OUTPUT_DIR = ROOT / "output"
WORK_DIR = ROOT / "output" / "_work"  # intermediate files (audio, subs)

for _d in (SCRIPTS_DIR, IMAGES_DIR, MUSIC_DIR, CUTAWAY_DIR, OUTPUT_DIR, WORK_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Cutaways (cutscene inserts) ---
# Insert a clip AFTER a given sentence (1-based). The narration PAUSES for the
# clip's length and the clip's OWN audio plays; then narration resumes. The clip
# is letterboxed to 9:16 (full widescreen frame preserved). Files live in
# assets/cutaways/. Example:
#   CUTAWAYS = [{"after_sentence": 5, "clip": "artorias_speech.mp4"}]
CUTAWAYS = [

]

# --- Video format (YouTube Shorts / TikTok / Reels) ---
WIDTH = 1080
HEIGHT = 1920
FPS = 30

# --- TTS (ElevenLabs only — no fallback) ---
# Needs ELEVENLABS_API_KEY in your .env. Find voice IDs:  python -m modules.tts voices
ELEVENLABS_VOICE_ID = "goT3UYdM9bhm0n2lmKQx"  # Oliver Silk - Deep Gravel Narrative
ELEVENLABS_MODEL = "eleven_multilingual_v2"   # best quality; "eleven_flash_v2_5" = cheaper/faster
ELEVENLABS_STABILITY = 0.3   # lower = more expressive/varied, higher = more consistent
ELEVENLABS_SIMILARITY = 0.85  # how closely to match the chosen voice's character
ELEVENLABS_STYLE = 0.45       # 0 = neutral read, higher = more dramatic (good for lore)
ELEVENLABS_USE_SPEAKER_BOOST = True

# --- Captions (reuses Aria's Whisper) ---
WHISPER_MODEL = "small"      # matches your current Aria regression; "base" is faster
CAPTION_MAX_WORDS = 3        # words on screen at once (3 keeps lines short = fits width)
CAPTION_FONT = "Arial"       # any font installed on your system
CAPTION_FONTSIZE = 78        # smaller so long words don't overflow
CAPTION_OUTLINE_W = 5        # black outline thickness for legibility over images
CAPTION_PRIMARY = "&H00FFFFFF"   # white  (ASS uses &HAABBGGRR)
CAPTION_HIGHLIGHT = "&H0000F0FF" # yellow-ish for the active word
CAPTION_OUTLINE = "&H00000000"   # black outline
CAPTION_MARGIN_H = 90            # left/right margin so text wraps instead of clipping
CAPTION_MARGIN_V = 360           # vertical position from bottom

# --- Visuals ---
KEN_BURNS_ZOOM = 1.18        # how far the slow zoom pushes in over a scene
TRANSITION_SEC = 0.4         # crossfade duration between scenes (auto-clamped to short scenes)
MUSIC_VOLUME = 0.16          # background music level under the voice (0-1)
DEFAULT_MUSIC = "music.mp3"  # used if --music not passed; "" to disable

# --- Rewrite step ---
REWRITE_BACKEND = "ollama"   # "ollama" (local Mistral, like Aria) | "claude" | "none"
OLLAMA_MODEL = "mistral"
