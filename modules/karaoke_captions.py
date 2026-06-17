"""Active-word caption generator (ASS) — the one-word-at-a-time, big-yellow-pop style.

Takes Whisper word-level timestamps (which the pipeline already produces) and emits
a styled .ass subtitle file. Burn it with ffmpeg's subtitles filter:

    ffmpeg -i in.mp4 -vf "subtitles=captions.ass:fontsdir=./fonts" -c:a copy out.mp4

Style match for the reference:
  - one word at a time (active word), uppercase
  - heavy bold font, bright yellow fill, thick black outline + soft shadow
  - centred in the lower-middle of the frame
  - each word "pops" in: scales 70 -> 112 -> 100 with a quick fade

Vector text via libass means crisp captions and no per-frame Python compositing.

Vendored into the shorts pipeline from a standalone module. The only local change is
the `max_gap`/`hold` pair on CaptionStyle (and its use in build_ass), which lets a held
word clear during a long pause or a cutaway instead of stretching across it — matching
the classic renderer's GAP_CAP/HOLD behaviour. With `max_gap=None` (the default) the
original gap-fill behaviour is preserved exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# RGB palette auto-assigned to speakers (in order of first appearance) when no
# explicit colour is set. Module-level so callers (e.g. the gameplay transcript
# editor) can show the same swatches the renderer will use.
DEFAULT_SPEAKER_PALETTE: list[tuple[int, int, int]] = [
    (255, 235, 0),    # yellow
    (0, 229, 255),    # cyan
    (118, 255, 3),    # lime
    (255, 64, 129),   # hot pink
    (255, 145, 0),    # orange
    (179, 136, 255),  # violet
]


def _ass_time(t: float) -> str:
    """Seconds -> H:MM:SS.cc (centiseconds), ASS format."""
    if t < 0:
        t = 0.0
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_color(r: int, g: int, b: int) -> str:
    """RGB -> ASS &HBBGGRR&."""
    return f"&H00{b:02X}{g:02X}{r:02X}"


@dataclass
class CaptionStyle:
    font: str = "Poppins"            # production: Montserrat Black / Anton / Theboldfont
    fontsize: int = 150              # ~8% of 1920 height
    fill: tuple[int, int, int] = (255, 255, 0)     # yellow
    outline_rgb: tuple[int, int, int] = (0, 0, 0)  # black
    outline: float = 7.0             # thick stroke
    shadow: float = 3.0
    play_w: int = 1080
    play_h: int = 1920
    pos_y_frac: float = 0.60         # vertical centre of the text (0=top, 1=bottom)
    uppercase: bool = True
    pop_in_ms: int = 120             # scale-up duration
    settle_ms: int = 90              # overshoot -> settle
    fade_ms: int = 40
    min_hold_s: float = 0.18         # never flash a word shorter than this
    gap_fill: bool = True            # extend each word until the next begins (no flicker)
    words_per_cue: int = 1           # 1 = pure active-word; 2 works too
    max_gap: float | None = None     # if next word is >max_gap away, don't stretch; hold instead
    hold: float = 0.4                # how long a word lingers when a big gap is capped
    # Per-speaker colour (gameplay pipeline). Explicit map wins; otherwise speakers
    # are auto-assigned from the palette in order of first appearance. With no
    # speaker on a cue (the 3-tuple lore path) the style default fill is used.
    speaker_colors: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    speaker_palette: list[tuple[int, int, int]] = field(
        default_factory=lambda: list(DEFAULT_SPEAKER_PALETTE))
    # Render-side defence in depth (all default OFF -> lore output is unchanged):
    max_event_s: float | None = None   # cap how long any one cue stays on screen
    max_line_chars: int = 0            # wrap/hard-split so no line exceeds frame width
    prevent_overlap: bool = False      # snap a cue's end to the next cue's start


def _header(st: CaptionStyle) -> str:
    primary = _ass_color(*st.fill)
    outline = _ass_color(*st.outline_rgb)
    # Alignment 5 = middle-centre; we place precisely with \pos per line.
    return (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {st.play_w}\nPlayResY: {st.play_h}\n"
        "WrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Pop,{st.font},{st.fontsize},{primary},{primary},{outline},&H64000000,"
        f"-1,0,0,0,100,100,0,0,1,{st.outline},{st.shadow},5,0,0,0,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )


def _resolve_speaker_color(st: CaptionStyle, speaker, assigned: dict) -> str | None:
    """Return an ASS colour override for a speaker, or None to use the style default.

    An explicit `speaker_colors[speaker]` wins; otherwise the speaker is assigned
    the next palette colour in order of first appearance (cached in `assigned`)."""
    if speaker is None:
        return None
    if speaker in st.speaker_colors:
        return _ass_color(*st.speaker_colors[speaker])
    if speaker not in assigned:
        idx = len(assigned) % len(st.speaker_palette)
        assigned[speaker] = st.speaker_palette[idx]
    return _ass_color(*assigned[speaker])


def _wrap_text(text: str, max_chars: int) -> str:
    """Wrap to <=max_chars per line at word boundaries, hard-splitting any single
    token longer than the limit, so no line can exceed the frame width. Uses ASS
    line breaks (\\N). No-op when max_chars<=0 (the lore path)."""
    if max_chars <= 0:
        return text
    tokens: list[str] = []
    for word in text.split():
        while len(word) > max_chars:           # hard-split an over-long token
            tokens.append(word[:max_chars])
            word = word[max_chars:]
        tokens.append(word)
    lines: list[str] = []
    cur = ""
    for tok in tokens:
        if not cur:
            cur = tok
        elif len(cur) + 1 + len(tok) <= max_chars:
            cur += " " + tok
        else:
            lines.append(cur)
            cur = tok
    if cur:
        lines.append(cur)
    return "\\N".join(lines)


def _cue_text(st: CaptionStyle, text: str, color: str | None = None) -> str:
    if st.uppercase:
        text = text.upper()
    text = _wrap_text(text, st.max_line_chars)
    x = st.play_w // 2
    y = int(st.play_h * st.pos_y_frac)
    color_tag = f"\\c{color}&" if color else ""
    # pop: small -> overshoot -> settle, with a quick fade-in/out
    anim = (
        f"\\pos({x},{y})\\an5\\fad({st.fade_ms},{st.fade_ms}){color_tag}"
        f"\\fscx70\\fscy70"
        f"\\t(0,{st.pop_in_ms},\\fscx112\\fscy112)"
        f"\\t({st.pop_in_ms},{st.pop_in_ms + st.settle_ms},\\fscx100\\fscy100)"
    )
    return "{" + anim + "}" + text.replace("\n", " ")


def build_ass(words, style: CaptionStyle | None = None) -> str:
    """words: list of (text, start_s, end_s) OR (text, start_s, end_s, speaker).

    The 4-tuple form drives per-speaker colour (gameplay pipeline). The editable
    transcript the review step produces is exactly this list of tuples, so a
    corrected transcript flows straight in with no other changes. The 3-tuple form
    (lore path) leaves `speaker` as None and uses the style's default fill.
    """
    st = style or CaptionStyle()
    lines = [_header(st)]
    assigned: dict = {}  # speaker -> colour, in order of first appearance

    # group words per cue (default 1)
    cues: list[tuple[str, float, float, object]] = []
    n = st.words_per_cue
    for i in range(0, len(words), n):
        chunk = words[i:i + n]
        text = " ".join(w[0] for w in chunk)
        start = chunk[0][1]
        end = chunk[-1][2]
        speaker = chunk[0][3] if len(chunk[0]) > 3 else None
        cues.append((text, start, end, speaker))

    for i, (text, start, end, speaker) in enumerate(cues):
        # extend to next cue's start so a single word stays up with no flicker,
        # unless the gap is too big (a long pause / cutaway) — then just hold briefly.
        next_start = cues[i + 1][1] if i + 1 < len(cues) else None
        if st.gap_fill and next_start is not None:
            if st.max_gap is not None and next_start - end > st.max_gap:
                end = end + st.hold
            elif st.prevent_overlap:
                end = next_start            # active-word: show exactly until the next
            else:
                end = max(end, next_start)
        if end - start < st.min_hold_s:
            end = start + st.min_hold_s
        if st.max_event_s and end - start > st.max_event_s:
            end = start + st.max_event_s    # no screen-wide "AAAA…" wall
        if st.prevent_overlap and next_start is not None and next_start > start:
            end = min(end, next_start)      # never overlap the next cue
        color = _resolve_speaker_color(st, speaker, assigned)
        lines.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Pop,,0,0,0,,"
            f"{_cue_text(st, text, color)}"
        )
    return "\n".join(lines) + "\n"


def write_ass(words, path: str,
              style: CaptionStyle | None = None) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(build_ass(words, style))
    return path
