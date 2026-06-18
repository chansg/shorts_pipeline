"""Transcript-grid bulk-edit operations. Pure, no GPU/ffmpeg. Asserts the
(text, speaker, start, end) row contract is preserved through every op."""
from gameplay.editing import (assign_speaker, find_replace, merge_rows,
                              parse_row_span, split_row)
from gameplay.transcript import Transcript


def _rows():
    return [["hey", "S0", 0.0, 0.4],
            ["jet", "S0", 0.4, 0.8],
            ["clutch", "S1", 0.8, 1.4],
            ["gg", "S1", 1.4, 1.8]]


def test_parse_row_span_forms():
    assert parse_row_span("1-3,4", 4) == [0, 1, 2, 3]
    assert parse_row_span("2 4", 4) == [1, 3]
    assert parse_row_span("3-1", 4) == [0, 1, 2]      # reversed range ok
    assert parse_row_span("0,5,9", 4) == []           # out of range dropped
    assert parse_row_span("", 4) == []


def test_assign_speaker_multi_row():
    out = assign_speaker(_rows(), "1-2", "Chan")
    assert [r[1] for r in out] == ["Chan", "Chan", "S1", "S1"]
    # contract intact: 5 cols (text,speaker,start,end,censor), times unchanged
    assert out[0] == ["hey", "Chan", 0.0, 0.4, False]


def test_find_replace_counts_and_case():
    rows = [["Jet peek", "S0", 0, 1], ["jet again", "S0", 1, 2]]
    out, n = find_replace(rows, "jet", "Jett", case_sensitive=False)
    assert n == 2 and out[0][0] == "Jett peek" and out[1][0] == "Jett again"
    out2, n2 = find_replace(rows, "jet", "Jett", case_sensitive=True)
    assert n2 == 1                                  # only the lowercase one


def test_find_replace_whole_word():
    rows = [["jetpack jet", "S0", 0, 1]]
    out, n = find_replace(rows, "jet", "X", whole_word=True)
    assert n == 1 and out[0][0] == "jetpack X"


def test_merge_rows():
    out = merge_rows(_rows(), "1-2")
    assert len(out) == 3
    assert out[0] == ["hey jet", "S0", 0.0, 0.8, False]   # joined, span min..max
    assert out[1][0] == "clutch"


def test_split_row_prorates_time():
    out = split_row([["nice shot man", "S0", 0.0, 3.0]], 1)   # 3 words -> k=1
    assert len(out) == 2
    assert out[0] == ["nice", "S0", 0.0, 1.0, False]
    assert out[1] == ["shot man", "S0", 1.0, 3.0, False]


def test_split_single_word_is_noop():
    assert split_row([["gg", "S0", 0.0, 1.0]], 1) == [["gg", "S0", 0.0, 1.0, False]]


def test_edits_preserve_censor_flag():
    # the censor flag (col 5) must survive speaker-assign, merge and split.
    rows = [["damn", "S0", 0.0, 0.4, True], ["it", "S0", 0.4, 0.8, False]]
    assert assign_speaker(rows, "1", "Chan")[0][4] is True
    assert merge_rows(rows, "1-2")[0][4] is True             # censor if any part was
    split = split_row([["damn it", "S0", 0.0, 1.0, True]], 1)
    assert split[0][4] is True and split[1][4] is True       # both halves inherit


def test_edits_feed_build_ass_contract():
    # rows -> Transcript.from_rows -> tuples, the burn input, still valid after edits
    rows = assign_speaker(_rows(), "3-4", "Sam")
    rows = split_row(rows, 1)
    t = Transcript.from_rows(rows)
    tuples = t.to_tuples()
    assert all(len(x) == 4 for x in tuples)         # 4-tuples (speakers present)
    assert {x[3] for x in tuples} == {"S0", "Sam"}
