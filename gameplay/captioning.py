"""Transform the word-level transcript into caption cues for the gameplay renderer.

Strict one-word-at-a-time karaoke magnifies any per-word ASR drift and reads as broken
when words are sparse. "phrase" mode groups a few consecutive words into one cue that
holds for the span of its words — far more forgiving of drift and easier to read on a
phone. The renderer (`modules.karaoke_captions.build_ass`) is untouched; this is a pure
transform over the word list that preserves the `(text, start, end, speaker)` contract.

Min-duration and gap-bridging are applied by the renderer via `CaptionStyle`
(`min_hold_s` = CAPTION_MIN_DUR_S, `max_gap` = CAPTION_MAX_GAP_S) so there's no
duplicate timing logic here — this module only groups words and applies the global
`offset`.
"""
from __future__ import annotations

from gameplay import config as gconf


def _word_text(w, mask: bool, mask_style) -> str:
    text = w.text
    if mask and getattr(w, "censor", False):
        from gameplay import censor
        text = censor.mask_text(text, mask_style)
    return text


def chunk_words(words, *, max_words: int | None = None,
                max_window_s: float | None = None, max_chars: int | None = None,
                offset: float = 0.0, mask: bool = False, mask_style=None) -> list[tuple]:
    """Group consecutive SAME-SPEAKER words into phrase cues (≤max_words, ≤max_window_s
    span, ≤max_chars of text), applying a global time `offset`. A cue spans
    [first.start, last.end]. Returns `(text, start, end, speaker)` tuples."""
    max_words = gconf.CAPTION_CHUNK_MAX_WORDS if max_words is None else max_words
    max_window_s = gconf.CAPTION_CHUNK_MAX_WINDOW_S if max_window_s is None else max_window_s
    max_chars = gconf.CAPTION_CHUNK_MAX_CHARS if max_chars is None else max_chars

    cues: list[tuple] = []
    group: list = []

    def flush():
        if not group:
            return
        text = " ".join(_word_text(w, mask, mask_style) for w in group)
        cues.append((text, round(group[0].start + offset, 3),
                     round(group[-1].end + offset, 3), group[0].speaker))
        group.clear()

    for w in words:
        if group:
            cand_len = len(" ".join(x.text for x in group)) + 1 + len(w.text)
            window = w.end - group[0].start
            if (w.speaker != group[0].speaker or len(group) >= max_words
                    or cand_len > max_chars or window > max_window_s):
                flush()
        group.append(w)
    flush()
    return cues


def word_tuples(words, *, offset: float = 0.0, mask: bool = False,
                mask_style=None) -> list[tuple]:
    """One-word-at-a-time cues (the classic karaoke), with a global `offset` applied.
    Speaker may be None (single-speaker) — the renderer then uses its default fill."""
    return [(_word_text(w, mask, mask_style), round(w.start + offset, 3),
             round(w.end + offset, 3), w.speaker) for w in words]


def caption_cues(words, *, mode: str | None = None, offset: float | None = None,
                 mask: bool = False, mask_style=None) -> list[tuple]:
    """Build caption tuples from the transcript words for the given mode
    ("phrase" | "word"), applying the global caption offset."""
    mode = gconf.CAPTION_CHUNK_MODE if mode is None else mode
    offset = gconf.CAPTION_OFFSET_S if offset is None else offset
    if mode == "word":
        return word_tuples(words, offset=offset, mask=mask, mask_style=mask_style)
    return chunk_words(words, offset=offset, mask=mask, mask_style=mask_style)
