"""The transcript model + WhisperX adapter + single-speaker fallback. No GPU:
these validate everything downstream of the GPU transcribe step."""
from gameplay.transcript import (Transcript, Word, from_whisperx,
                                 speaker_color_map)


def _wx(segments):
    return {"language": "en", "segments": segments}


def test_from_whisperx_multi_speaker_yields_4tuples():
    result = _wx([
        {"words": [{"word": "hey", "start": 0.0, "end": 0.4, "speaker": "SPEAKER_00"},
                   {"word": "nice", "start": 0.4, "end": 0.9, "speaker": "SPEAKER_01"}]},
    ])
    t = from_whisperx(result)
    assert not t.single_speaker
    assert t.speakers == ["SPEAKER_00", "SPEAKER_01"]
    tuples = t.to_tuples()
    assert all(len(x) == 4 for x in tuples)
    assert tuples[0] == ("hey", 0.0, 0.4, "SPEAKER_00")


def test_single_speaker_fallback_yields_3tuples():
    # Only one voice -> single-speaker; speakers normalised to None; 3-tuples out.
    result = _wx([
        {"words": [{"word": "alone", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
                   {"word": "here", "start": 0.5, "end": 1.0, "speaker": "SPEAKER_00"}]},
    ])
    t = from_whisperx(result)
    assert t.single_speaker
    assert t.speakers == []
    assert t.to_tuples() == [("alone", 0.0, 0.5), ("here", 0.5, 1.0)]


def test_no_speaker_labels_is_single_speaker():
    # No diarization at all (no "speaker" key) -> single-speaker fallback.
    result = _wx([{"words": [{"word": "solo", "start": 0.0, "end": 0.5}]}])
    t = from_whisperx(result)
    assert t.single_speaker
    assert t.to_tuples() == [("solo", 0.0, 0.5)]


def test_missing_word_timestamp_inherits_neighbour():
    result = _wx([{"words": [
        {"word": "a", "start": 0.0, "end": 0.5, "speaker": "S0"},
        {"word": ".", "speaker": "S0"},                       # punctuation, no times
        {"word": "b", "start": 0.6, "end": 1.0, "speaker": "S1"}]}])
    t = from_whisperx(result)
    assert len(t.words) == 3
    assert t.words[1].start == 0.5 and t.words[1].end == 0.5


def test_rows_roundtrip_and_rename():
    t = Transcript([Word("hi", 0.0, 0.4, "SPEAKER_00"),
                    Word("yo", 0.4, 0.8, "SPEAKER_01")])
    rows = t.to_rows()
    assert rows[0] == ["hi", "SPEAKER_00", 0.0, 0.4]
    t2 = Transcript.from_rows(rows)
    assert t2.to_tuples() == t.to_tuples()
    t2.rename_speaker("SPEAKER_00", "Chan")
    assert "Chan" in t2.speakers and "SPEAKER_00" not in t2.speakers


def test_from_rows_is_tolerant():
    # blank trailing row, non-numeric timing, swapped start/end
    rows = [["good", "Chan", "0.9", "0.3"],   # swapped -> reordered
            ["", "", "", ""],                 # blank -> dropped
            ["bad", "Sam", "x", "y"]]         # bad timing -> dropped
    t = Transcript.from_rows(rows)
    assert len(t.words) == 1
    w = t.words[0]
    assert w.text == "good" and w.start == 0.3 and w.end == 0.9


def test_save_load_roundtrip(tmp_path):
    t = Transcript([Word("a", 0.0, 0.4, "S0"), Word("b", 0.4, 0.9, "S1")])
    p = t.save(tmp_path / "t.json")
    t2 = Transcript.load(p)
    assert t2.to_tuples() == t.to_tuples()
    assert t2.single_speaker == t.single_speaker


def test_speaker_color_map_normalises():
    m = speaker_color_map({"Chan": [255, 0, 0], "Sam": (0, 255, 0), "X": None})
    assert m == {"Chan": (255, 0, 0), "Sam": (0, 255, 0)}
