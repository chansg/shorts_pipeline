"""The transcript editor's speaker→colour grid seeding (no GPU)."""
from gameplay import gui as g
from gameplay import config as gconf
from gameplay.transcript import Transcript, Word


def test_speaker_rows_seeds_default_hex_when_no_speakers():
    # Single-speaker / collapsed diarization -> the grid must still offer default
    # SPEAKER_NN rows with valid hex colours (so it's never empty to populate).
    t = Transcript([Word("hi", 0.0, 0.4)], single_speaker=True)
    rows = g._speaker_rows(t)
    assert len(rows) == gconf.DEFAULT_SPEAKER_ROWS
    for name, hexv in rows:
        assert name.startswith("SPEAKER_")
        assert hexv.startswith("#") and len(hexv) == 7      # #RRGGBB
    # the seeded hex must round-trip back to a palette rgb
    assert g._hex_to_rgb(rows[0][1]) == gconf.SPEAKER_PALETTE[0]


def test_speaker_rows_detected_first_then_padded_to_default():
    # detected speakers lead; the grid is PADDED with SPEAKER_NN up to DEFAULT_SPEAKER_ROWS
    # so there are always at least that many speakers to assign to.
    t = Transcript([Word("a", 0.0, 0.4, "Chan"), Word("b", 0.4, 0.8, "Sam")])
    rows = g._speaker_rows(t)
    assert [r[0] for r in rows][:2] == ["Chan", "Sam"]      # detected first
    assert len(rows) == gconf.DEFAULT_SPEAKER_ROWS == 5     # padded to 5
    assert len({r[0] for r in rows}) == len(rows)           # unique names
    assert all(r[1].startswith("#") and len(r[1]) == 7 for r in rows)


def test_palette_has_enough_distinct_colours_for_default_speakers():
    assert len(gconf.SPEAKER_PALETTE) >= gconf.DEFAULT_SPEAKER_ROWS
    first5 = gconf.SPEAKER_PALETTE[:5]
    assert len(set(first5)) == 5                            # 5 distinct default colours
