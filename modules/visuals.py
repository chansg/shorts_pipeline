"""
Step 4 — Visuals.

AI-generated stills with a slow Ken Burns push. Drop images into
assets/images/ named so they sort in order (01.png, 02.png ...).

Image timing is SENTENCE-ALIGNED: instead of splitting time blindly, images
switch only at sentence boundaries in the narration. Your images map, in order,
to an equal share of the sentences. So if you write one sentence per creature
and provide one image per creature, each creature's image stays on screen for
exactly its sentence — "this creature until the next."

  - 1 image per sentence  -> image i shows during sentence i (most control)
  - fewer images          -> each image covers a contiguous block of sentences
  - no word timings given  -> falls back to an even time split
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import re
import config

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".m4v"}
VALID_EXT = IMAGE_EXT | VIDEO_EXT


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXT


def _split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]


def _script_sentence_ends(script_text: str, words, total_duration: float) -> list[float]:
    """End-time of each SCRIPT sentence. Pass DRIFT-CORRECTED words (aligned to
    the script) so the cumulative word index is exact even when Whisper added or
    dropped a word. Each boundary is the START of the next sentence's first word,
    so a sentence's full audio AND its trailing pause stay on the right side of
    the cut — important for cutaways (the last words finish before the clip)."""
    sents = _split_sentences(script_text)
    if not sents or not words:
        return [total_duration]
    counts = [len(s.split()) for s in sents]
    ends: list[float] = []
    idx = 0
    for si, c in enumerate(counts):
        idx += c
        if si < len(counts) - 1:
            nwi = min(idx, len(words) - 1)          # first word of the NEXT sentence
            # boundary sits just before the next sentence begins (after the pause)
            ends.append(words[nwi].start)
        else:
            ends.append(total_duration)             # last sentence runs to the end
    # enforce monotonic increase (guards against odd timestamps)
    for i in range(1, len(ends)):
        if ends[i] <= ends[i - 1]:
            ends[i] = min(ends[i - 1] + 0.2, total_duration)
    return ends


@dataclass
class Scene:
    image: Path
    duration: float  # seconds on screen
    is_cutaway: bool = False
    focus: str = "center"  # crop anchor: left/center/right[-top/center/bottom]


_FOCUS_WORDS = {"left", "right", "center", "centre", "top", "bottom"}


def parse_focus(path: Path) -> str:
    """Read an optional crop anchor from the filename, after the beat number:
    `02_right.png` -> 'right', `05_top-right.png` -> 'top-right', `01.png` ->
    'center'. Lets you keep an off-centre subject without editing any config."""
    stem = path.stem.lower().strip()
    m = re.match(r"^\d+[ _-]+(.+)$", stem)
    if not m:
        return "center"
    parts = [("center" if p == "centre" else p)
             for p in m.group(1).replace("_", "-").split("-") if p]
    if parts and all(p in _FOCUS_WORDS for p in parts):
        return "-".join(parts)
    return "center"


def _order_key(p: Path):
    m = re.match(r"^(\d+)", p.stem)
    return (int(m.group(1)) if m else 9999, p.stem)


def collect_images(images_dir: str | Path | None = None) -> list[Path]:
    """Collect scene media (images AND/OR video clips) ordered by beat number.
    Mix freely: 01.png, 02.mp4, 03.png ... each maps to a sentence in order.
    An optional suffix sets the crop anchor: 02_right.png, 07_top-left.png."""
    d = Path(images_dir) if images_dir else config.IMAGES_DIR
    imgs = sorted((p for p in d.iterdir() if p.suffix.lower() in VALID_EXT),
                  key=_order_key)
    if not imgs:
        raise FileNotFoundError(f"No images or video clips found in {d}")
    return imgs


def _sentence_end_times(words, total_duration: float) -> list[float]:
    """Group words into sentences and return each sentence's END time."""
    ends: list[float] = []
    for w in words:
        # a word ends a sentence if its text closes with . ! ? (allowing quotes)
        if re.search(r'[.!?]["\')\]]*$', w.text):
            ends.append(w.end)
    # always close the final sentence at the true end of the audio
    if not ends or ends[-1] < total_duration - 0.05:
        ends.append(total_duration)
    return ends


def sentence_end_times(words, total_duration: float, script_text: str | None = None) -> list[float]:
    """Public: end-time of each sentence. Script-driven if script_text given."""
    if not words:
        return [total_duration]
    if script_text:
        return _script_sentence_ends(script_text, words, total_duration)
    return _sentence_end_times(words, total_duration)


def count_sentences(words, total_duration: float, script_text: str | None = None) -> int:
    """How many sentences the narration splits into. Uses the script if given
    (deterministic), else falls back to Whisper's punctuation."""
    if script_text:
        return len(_split_sentences(script_text))
    if not words:
        return 0
    return len(_sentence_end_times(words, total_duration))


def build_scenes(images: list[Path], total_duration: float,
                 words=None, script_text: str | None = None) -> list[Scene]:
    """Sentence-aligned scenes. If the script text is given, sentence boundaries
    come from the SCRIPT (one image per script sentence); otherwise from Whisper;
    otherwise an even time split."""
    if not words:
        per = total_duration / len(images)
        return [Scene(img, per, focus=parse_focus(img)) for img in images]

    if script_text:
        sent_ends = _script_sentence_ends(script_text, words, total_duration)
    else:
        sent_ends = _sentence_end_times(words, total_duration)
    n_sent = len(sent_ends)
    n_img = len(images)

    if n_img > n_sent:
        per = total_duration / n_img
        return [Scene(img, per, focus=parse_focus(img)) for img in images]

    scenes: list[Scene] = []
    prev_time = 0.0
    for i, img in enumerate(images):
        hi_idx = round((i + 1) * n_sent / n_img)
        hi_idx = max(i + 1, min(hi_idx, n_sent))
        boundary = sent_ends[hi_idx - 1] if i < n_img - 1 else total_duration
        duration = max(0.2, boundary - prev_time)
        scenes.append(Scene(img, duration, focus=parse_focus(img)))
        prev_time = boundary

    return scenes