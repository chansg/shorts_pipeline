"""karaoke_captions speaker-colour mapping + 3-tuple back-compat. Guards that the
gameplay 4-tuple path colours per speaker while the lore 3-tuple path is unchanged."""
import re

from modules.karaoke_captions import build_ass, CaptionStyle, DEFAULT_SPEAKER_PALETTE
from gameplay.transcript import Transcript, Word


def _color_overrides(ass: str) -> list[str]:
    return re.findall(r"\\c(&H[0-9A-F]+)&", ass)


def test_3tuple_lore_path_has_no_colour_override():
    ass = build_ass([("hello", 0.0, 0.5), ("world", 0.5, 1.0)])
    assert _color_overrides(ass) == []


def test_4tuple_two_speakers_get_distinct_palette_colours():
    ass = build_ass([("hey", 0.0, 0.4, "SPEAKER_00"),
                     ("nice", 0.4, 0.9, "SPEAKER_01")])
    cols = _color_overrides(ass)
    assert len(cols) == 2 and len(set(cols)) == 2


def test_same_speaker_keeps_one_colour():
    ass = build_ass([("a", 0.0, 0.3, "S0"), ("b", 0.3, 0.6, "S0"),
                     ("c", 0.6, 0.9, "S0")])
    cols = _color_overrides(ass)
    assert len(set(cols)) == 1


def test_explicit_speaker_colors_win():
    st = CaptionStyle(speaker_colors={"Chan": (255, 0, 0)})   # red
    ass = build_ass([("go", 0.0, 0.4, "Chan")], st)
    # ASS colour is &H00BBGGRR -> red = &H000000FF
    assert "&H000000FF" in _color_overrides(ass)


def test_palette_first_colour_matches_module_default():
    ass = build_ass([("x", 0.0, 0.4, "ONLY")])
    r, g, b = DEFAULT_SPEAKER_PALETTE[0]
    assert f"&H00{b:02X}{g:02X}{r:02X}" in _color_overrides(ass)


def test_single_speaker_transcript_renders_without_colour():
    # End-to-end through the gameplay adapter: single speaker -> no override.
    t = Transcript([Word("solo", 0.0, 0.5, "S0")], single_speaker=True)
    ass = build_ass(t.to_tuples())
    assert _color_overrides(ass) == []
