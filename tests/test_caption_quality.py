"""Render-side caption defence: no over-long events, no on-screen overlap, line
wrap to frame width — and the lore path stays byte-identical (defaults off)."""
import re

from modules.karaoke_captions import CaptionStyle, build_ass, _wrap_text
from gameplay.manual import ManualOptions, caption_style


def _events(ass: str):
    """(start, end) seconds for each Dialogue line."""
    out = []
    for m in re.finditer(r"Dialogue: 0,(\d):(\d\d):(\d\d\.\d\d),(\d):(\d\d):(\d\d\.\d\d),", ass):
        h1, m1, s1, h2, m2, s2 = m.groups()
        out.append((int(h1) * 3600 + int(m1) * 60 + float(s1),
                    int(h2) * 3600 + int(m2) * 60 + float(s2)))
    return out


def _gameplay_style():
    return caption_style(ManualOptions())


def test_mega_token_event_is_clamped():
    # a 28s "word" must not produce a 28s caption event
    words = [("hi", 0.0, 0.4, "S0"), ("NAAAA", 3.1, 31.08, "S0")]
    ass = build_ass(words, _gameplay_style())
    evs = _events(ass)
    assert evs, "no events emitted"
    assert max(e - s for s, e in evs) <= 1.2 + 1e-6


def test_no_consecutive_overlap_gameplay():
    words = [("a", 0.0, 5.0, "S0"), ("b", 0.5, 6.0, "S0"), ("c", 1.0, 7.0, "S0")]
    evs = _events(build_ass(words, _gameplay_style()))
    for i in range(len(evs) - 1):
        assert evs[i][1] <= evs[i + 1][0] + 1e-6, f"event {i} overlaps {i+1}: {evs}"


def test_long_token_is_wrapped_to_frame_width():
    wrapped = _wrap_text("SUPERCALIFRAGILISTIC", 12)
    assert "\\N" in wrapped
    assert all(len(line) <= 12 for line in wrapped.split("\\N"))
    # multi-word wraps at boundaries
    w2 = _wrap_text("LETS GO BABY THATS INSANE", 12)
    assert all(len(line) <= 12 for line in w2.split("\\N"))


def test_build_ass_wraps_in_output():
    ass = build_ass([("ABSOLUTELYRIDICULOUS", 0.0, 0.4, "S0")], _gameplay_style())
    assert "\\N" in ass


def test_lore_path_unchanged_by_new_fields():
    # Default CaptionStyle (lore) must behave exactly as before: no wrap, gap-fill
    # extends to the next word (max), no event clamp.
    words = [("hello", 0.0, 0.5), ("world", 2.0, 2.5)]
    ass = build_ass(words, CaptionStyle())
    evs = _events(ass)
    # first cue extends to the second's start (2.0) — the original max() behaviour
    assert evs[0] == (0.0, 2.0)
    assert "\\N" not in ass                      # no wrapping on the lore path


def test_lore_long_event_not_clamped():
    # without max_event_s, a long gap-filled hold is preserved (lore unchanged)
    words = [("a", 0.0, 0.3), ("b", 10.0, 10.3)]
    # default max_gap None -> gap_fill extends a to 10.0
    evs = _events(build_ass(words, CaptionStyle()))
    assert evs[0] == (0.0, 10.0)
