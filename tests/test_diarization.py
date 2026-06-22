"""Regression lock for the diarization AttributeError class of bug (v1 broke on a
whisperx API move). Mocks the pyannote/whisperx diarization result in its several
shapes and asserts turn-extraction + segment->word speaker mapping. No GPU/token."""
import types

import pytest

from gameplay.transcribe import (assign_speakers, diarization_turns,
                                 _diar_failure_message, _is_auth_error)
from gameplay.transcript import from_whisperx


# ---- diarization_turns: every result shape -> (start, end, speaker) ----

def test_turns_from_list_of_tuples():
    turns = diarization_turns([(0.0, 1.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")])
    assert turns == [(0.0, 1.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")]


def test_turns_from_list_of_dicts():
    turns = diarization_turns([{"start": 0.0, "end": 1.0, "speaker": "S0"},
                               {"start": 1.0, "end": 2.0, "label": "S1"}])
    assert turns == [(0.0, 1.0, "S0"), (1.0, 2.0, "S1")]


def test_turns_from_pyannote_annotation_like():
    # pyannote Annotation: .itertracks(yield_label=True) -> (segment, track, label)
    seg = lambda s, e: types.SimpleNamespace(start=s, end=e)

    class Annotation:
        def itertracks(self, yield_label=False):
            yield seg(0.0, 1.2), "A", "SPEAKER_00"
            yield seg(1.2, 2.0), "B", "SPEAKER_01"

    turns = diarization_turns(Annotation())
    assert turns == [(0.0, 1.2, "SPEAKER_00"), (1.2, 2.0, "SPEAKER_01")]


def test_turns_from_pandas_dataframe():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"segment": [None, None], "label": ["S0", "S1"],
                       "speaker": ["S0", "S1"], "start": [0.0, 1.0],
                       "end": [1.0, 2.0]})
    assert diarization_turns(df) == [(0.0, 1.0, "S0"), (1.0, 2.0, "S1")]


# ---- assign_speakers: segment -> word mapping ----

def _result():
    return {"segments": [
        {"start": 0.0, "end": 1.0, "words": [
            {"word": "hey", "start": 0.0, "end": 0.4},
            {"word": "there", "start": 0.5, "end": 0.9}]},
        {"start": 1.0, "end": 2.0, "words": [
            {"word": "nice", "start": 1.1, "end": 1.5}]}]}


def test_assign_speakers_by_overlap():
    turns = [(0.0, 1.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")]
    out = assign_speakers(_result(), turns)
    words = [w for seg in out["segments"] for w in seg["words"]]
    assert [w["speaker"] for w in words] == ["SPEAKER_00", "SPEAKER_00", "SPEAKER_01"]
    # end-to-end through the adapter -> multi-speaker transcript
    t = from_whisperx(out)
    assert not t.single_speaker and t.speakers == ["SPEAKER_00", "SPEAKER_01"]


def test_assign_speakers_fill_nearest_for_gap():
    # a word in a diarization gap still gets the nearest speaker
    result = {"segments": [{"start": 5.0, "end": 5.4, "words": [
        {"word": "late", "start": 5.0, "end": 5.4}]}]}
    turns = [(0.0, 1.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")]
    out = assign_speakers(result, turns, fill_nearest=True)
    assert out["segments"][0]["words"][0]["speaker"] == "SPEAKER_01"  # nearest


def test_assign_speakers_no_turns_is_noop():
    out = assign_speakers(_result(), [])
    assert all("speaker" not in w for seg in out["segments"] for w in seg["words"])


def test_assign_speaker_sums_overlap_across_fragmented_turns():
    # Cross-talk: pyannote emits many tiny alternating turns. A word spanning the whole
    # span must go to the speaker who DOMINATES it (summed), not to a sliver turn.
    from gameplay.transcribe import _best_speaker
    turns = [(12.6, 14.4, "S1"), (13.99, 14.10, "S0"),   # S1 dominates 12.6-15
             (14.64, 14.70, "S1"), (14.70, 14.73, "S0"),
             (14.75, 14.90, "S0"), (14.90, 15.9, "S1")]
    assert _best_speaker(turns, 12.67, 14.94, True) == "S1"   # summed S1 >> S0
    # but a word sitting mostly inside the S0 sliver region goes to S0
    assert _best_speaker(turns, 14.70, 14.90, True) == "S0"


def test_diarize_pins_num_speakers_when_set(monkeypatch):
    # _diarize must pass num_speakers=N (pinned) OR min/max (auto), per the argument.
    from gameplay import transcribe as tx
    calls = []

    class FakePipe:
        def __init__(self, *a, **k):
            pass

        def __call__(self, audio, **kw):
            calls.append(kw)
            return [(0.0, 1.0, "S0")]

    monkeypatch.setattr(tx, "_resolve_diarization_pipeline", lambda: FakePipe)
    tx._diarize("audio", "tok", "cpu", num_speakers=2)
    assert calls[-1] == {"num_speakers": 2}
    tx._diarize("audio", "tok", "cpu", num_speakers=None)
    assert "min_speakers" in calls[-1] and "max_speakers" in calls[-1]
    assert "num_speakers" not in calls[-1]


def test_word_without_timestamp_is_skipped():
    result = {"segments": [{"start": 0.0, "end": 1.0, "words": [
        {"word": "punct"}]}]}    # no start/end
    out = assign_speakers(result, [(0.0, 1.0, "S0")])
    assert "speaker" not in out["segments"][0]["words"][0]


# ---- failure classification ----

def test_auth_error_classified():
    e = RuntimeError("401 Client Error: you must accept the licence / gated repo")
    assert _is_auth_error(e)
    msg = _diar_failure_message(e)
    assert "licence" in msg.lower() and "single-speaker" in msg.lower()
    assert "segmentation-3.0" in msg


def test_code_error_classified():
    e = AttributeError("module 'whisperx' has no attribute 'DiarizationPipeline'")
    assert not _is_auth_error(e)
    msg = _diar_failure_message(e)
    assert "version mismatch" in msg.lower()
    assert "AttributeError" in msg
