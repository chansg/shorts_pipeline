"""Workstream A: the edited grid is the source of truth for the build.

Proves from_rows survives every Gradio Dataframe value shape and that a
programmatic row edit changes the tuples reaching build_ass (and the burned .ass)."""
import pytest

from gameplay.transcript import Transcript, Word


_ROWS = [["BRO", "SPEAKER_01", 6.0, 6.4], ["six", "SPEAKER_00", 6.4, 6.9]]


def test_from_rows_list_of_lists():
    t = Transcript.from_rows(_ROWS)
    assert [w.text for w in t.words] == ["BRO", "six"]
    assert t.to_tuples()[0] == ("BRO", 6.0, 6.4, "SPEAKER_01")


def test_from_rows_pandas_does_not_raise():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(_ROWS, columns=Transcript.HEADERS)
    t = Transcript.from_rows(df)                       # must NOT raise
    assert [w.text for w in t.words] == ["BRO", "six"]


def test_from_rows_dict_payload():
    t = Transcript.from_rows({"headers": Transcript.HEADERS, "data": _ROWS})
    assert [w.text for w in t.words] == ["BRO", "six"]


def test_from_rows_numpy():
    np = pytest.importorskip("numpy")
    t = Transcript.from_rows(np.array(_ROWS, dtype=object))
    assert [w.text for w in t.words] == ["BRO", "six"]


def test_from_rows_none_and_garbage_are_empty():
    assert Transcript.from_rows(None).words == []
    assert Transcript.from_rows(123).words == []       # not iterable -> []


def test_edit_changes_tuples_reaching_build_ass():
    original = Transcript.from_rows(_ROWS).to_tuples()
    edited_rows = [["DUDE", "SPEAKER_01", 6.0, 6.4], ["six", "SPEAKER_00", 6.4, 6.9]]
    edited = Transcript.from_rows(edited_rows).to_tuples()
    assert original != edited
    assert edited[0][0] == "DUDE"      # the edit is what build_ass would burn


def test_edit_changes_burned_ass(tmp_path):
    # End-to-end through the manual caption writer: edited rows -> different .ass.
    from gameplay.manual import ManualOptions, write_captions
    opts = ManualOptions()
    a = write_captions(Transcript.from_rows(_ROWS), opts, tmp_path / "a.ass")
    b = write_captions(Transcript.from_rows(
        [["DUDE", "SPEAKER_01", 6.0, 6.4], ["six", "SPEAKER_00", 6.4, 6.9]]),
        opts, tmp_path / "b.ass")
    at = a.read_text(encoding="utf-8")
    bt = b.read_text(encoding="utf-8")
    assert at != bt
    assert "BRO" in at and "DUDE" in bt and "DUDE" not in at
