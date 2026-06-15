"""
Step 3 — Captions.

Reuses Aria's Whisper. Transcribes the voiceover with WORD-LEVEL timestamps,
then writes a styled .ass subtitle file with TikTok-style word highlighting.

Design notes (these fix the earlier off-screen / flicker problems):
- WrapStyle 0 + side margins so long words wrap instead of running off-screen.
- One caption visible at a time, CONTINUOUS: each word's line runs until the
  next word starts (no gaps), so the caption never flickers or appears to repeat.
- Small word groups (CAPTION_MAX_WORDS) keep each line short enough to fit.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import config


@dataclass
class Word:
    text: str
    start: float
    end: float


def transcribe_words(audio_path: str | Path) -> list[Word]:
    """faster-whisper with word timestamps — same lib as Aria's STT."""
    from faster_whisper import WhisperModel
    model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(str(audio_path), word_timestamps=True)
    words: list[Word] = []
    for seg in segments:
        for w in seg.words:
            words.append(Word(w.word.strip(), w.start, w.end))
    return words


def _ass_time(t: float) -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600); t -= h * 3600
    m = int(t // 60); t -= m * 60
    s = int(t)
    cs = int(round((t - s) * 100))
    if cs == 100:
        cs = 0; s += 1
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def _escape(text: str) -> str:
    # ASS uses { } for override blocks; escape stray braces in spoken text.
    return text.replace("{", "(").replace("}", ")")


def align_script_words(words: list[Word], script_text: str) -> tuple[list[Word], bool]:
    """Replace Whisper's (possibly misheard) word TEXT with the script's exact
    words, keeping Whisper's timing. Fixes fantasy proper nouns like 'Oolacile'.

    Robust to small drift: if Whisper splits/adds/drops a word, a sequence
    alignment still maps the rest correctly, so spelling stays from the script.
    Returns (aligned_words, exact) where `exact` is True only on a clean 1:1 map.
    """
    import difflib
    script_words = script_text.split()
    if not words or not script_words:
        return words, False

    exact = len(script_words) == len(words)

    w_norm = [w.text.lower().strip(".,!?;:\"'()") for w in words]
    s_norm = [s.lower().strip(".,!?;:\"'()") for s in script_words]
    sm = difflib.SequenceMatcher(a=w_norm, b=s_norm, autojunk=False)

    out: list[Word] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                w = words[i1 + k]
                out.append(Word(script_words[j1 + k], w.start, w.end))
        elif tag in ("replace", "delete", "insert"):
            # Map the script words in this block across the whisper time span.
            w_start = words[i1].start if i1 < len(words) else (out[-1].end if out else 0.0)
            w_end = words[i2 - 1].end if i2 - 1 < len(words) and i2 > i1 else w_start
            block = script_words[j1:j2]
            if not block:
                continue
            span = max(0.01, w_end - w_start)
            step = span / len(block)
            for k, sw in enumerate(block):
                out.append(Word(sw, w_start + k * step, w_start + (k + 1) * step))
    # enforce monotonic, non-zero spans
    for i in range(1, len(out)):
        if out[i].start < out[i - 1].start:
            out[i] = Word(out[i].text, out[i - 1].start, max(out[i].end, out[i - 1].start + 0.05))
    return out, exact


def apply_cutaway_shifts(words: list[Word], cutaways: list[dict]) -> list[Word]:
    """Shift each word later by the total duration of all cutaways inserted at
    or before its (original) start time. Keeps captions in sync after inserts."""
    if not cutaways:
        return words
    cuts = sorted(cutaways, key=lambda c: c["narration_time"])
    out = []
    for w in words:
        add = sum(c["duration"] for c in cuts if c["narration_time"] <= w.start + 1e-6)
        out.append(Word(w.text, w.start + add, w.end + add))
    return out


def shift_words_after(words: list[Word], cut_time: float, amount: float) -> list[Word]:
    """Push every word at or after cut_time later by `amount` seconds. Used so
    captions stay in sync after a cutaway clip is inserted into the timeline."""
    out = []
    for w in words:
        if w.start >= cut_time - 1e-6:
            out.append(Word(w.text, w.start + amount, w.end + amount))
        else:
            out.append(w)
    return out


def _to_tuples(words) -> list[tuple[str, float, float]]:
    """Normalise whatever word-timestamp shape we're handed into (text, start, end)
    tuples for the active-word renderer. Robust to Word objects, dicts (text/word +
    start/end), and (text, start, end) tuples/lists."""
    out: list[tuple[str, float, float]] = []
    for w in words:
        if isinstance(w, dict):
            text = w.get("text", w.get("word", ""))
            start, end = w["start"], w["end"]
        elif isinstance(w, (tuple, list)):
            text, start, end = w[0], w[1], w[2]
        else:  # Word dataclass or any object with .text/.start/.end
            text, start, end = w.text, w.start, w.end
        out.append((str(text).strip(), float(start), float(end)))
    return out


def _write_ass_active_word(words, out_path: Path,
                           script_text: str | None) -> Path:
    """Active-word style: one big bold yellow word at a time, popping in synced to
    the voice. Delegates the actual ASS to the vendored karaoke_captions module;
    we only adapt our word data and map config -> CaptionStyle."""
    from modules.karaoke_captions import CaptionStyle, build_ass
    if script_text:
        words, _ = align_script_words(words, script_text)
    style = CaptionStyle(
        font=config.CAPTION_AW_FONT,
        fontsize=config.CAPTION_AW_FONTSIZE,
        fill=tuple(config.CAPTION_AW_FILL),
        outline_rgb=tuple(config.CAPTION_AW_OUTLINE_RGB),
        outline=config.CAPTION_AW_OUTLINE,
        shadow=config.CAPTION_AW_SHADOW,
        play_w=config.WIDTH,
        play_h=config.HEIGHT,
        pos_y_frac=config.CAPTION_AW_POS_Y_FRAC,
        words_per_cue=config.CAPTION_AW_WORDS_PER_CUE,
        gap_fill=True,
        max_gap=0.8,   # don't let a word linger across a long pause / cutaway gap
        hold=0.4,
    )
    out_path.write_text(build_ass(_to_tuples(words), style), encoding="utf-8")
    return out_path


def write_ass(words: list[Word], out_path: str | Path,
              script_text: str | None = None,
              break_times: list[float] | None = None) -> Path:
    """Render captions to an .ass file. Dispatches on config.CAPTION_STYLE:
    "active_word" (one big yellow word that pops in) or "classic" (the older
    3-words-per-line style with the active word highlighted).

    Classic: one short caption group at a time, lines continuous (no gaps) so
    nothing flickers or repeats. Groups never straddle a `break_time` (cutaway
    boundary), and a caption won't linger across a large gap (clears during a
    cutscene). `break_times` only applies to the classic path; the active-word
    path clears long gaps via its own max_gap cap."""
    out_path = Path(out_path)
    if not words:
        out_path.write_text("", encoding="utf-8")
        return out_path

    if getattr(config, "CAPTION_STYLE", "classic") == "active_word":
        return _write_ass_active_word(words, out_path, script_text)

    if script_text:
        words, _ = align_script_words(words, script_text)

    n = config.CAPTION_MAX_WORDS
    breaks = sorted(break_times or [])
    GAP_CAP = 0.8   # if the next word is >0.8s away, don't stretch the caption
    HOLD = 0.4      # how long a capped caption lingers before clearing

    # Build groups of up to n words, starting a new group at any cutaway boundary.
    groups: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        crosses = bool(cur) and any(cur[-1].start < bt <= w.start for bt in breaks)
        if cur and (len(cur) >= n or crosses):
            groups.append(cur)
            cur = []
        cur.append(w)
    if cur:
        groups.append(cur)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {config.WIDTH}
PlayResY: {config.HEIGHT}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV
Style: Base,{config.CAPTION_FONT},{config.CAPTION_FONTSIZE},{config.CAPTION_PRIMARY},{config.CAPTION_OUTLINE},&H64000000,-1,1,{config.CAPTION_OUTLINE_W},2,2,{config.CAPTION_MARGIN_H},{config.CAPTION_MARGIN_H},{config.CAPTION_MARGIN_V}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    hi = config.CAPTION_HIGHLIGHT
    base = config.CAPTION_PRIMARY

    for gi, group in enumerate(groups):
        next_group_start = (groups[gi + 1][0].start
                            if gi + 1 < len(groups) else group[-1].end + 0.3)
        for wi, w in enumerate(group):
            if wi + 1 < len(group):
                end = group[wi + 1].start
            else:
                end = next_group_start
            # Don't let a caption linger across a big gap (e.g. a cutaway):
            # clear it shortly after the word is spoken.
            if end - w.end > GAP_CAP:
                end = w.end + HOLD
            start = w.start
            if end <= start:
                end = start + 0.08  # guard against zero/negative spans
            parts = []
            for ww in group:
                col = hi if ww is w else base
                parts.append(f"{{\\c{col}}}{_escape(ww.text)}")
            text = " ".join(parts)
            lines.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Base,,0,0,0,,{text}"
            )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path