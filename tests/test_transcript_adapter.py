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
    assert rows[0] == ["hi", "SPEAKER_00", 0.0, 0.4, False]   # +censor flag column
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


def test_clamp_long_words_kills_mega_token():
    from gameplay.transcript import clamp_long_words
    ws = [Word("hi", 0.0, 0.4, "S0"), Word("Naaaa", 3.1, 31.08, "S0")]
    clamp_long_words(ws, 1.2)
    assert ws[1].end == round(3.1 + 1.2, 3)          # 28s token clamped to 1.2s
    assert ws[0].end == 0.4                            # short word untouched


def test_from_whisperx_clamps_before_grid():
    result = {"language": "en", "segments": [{"words": [
        {"word": "Naaaa", "start": 3.1, "end": 31.08, "speaker": "S0"}]}]}
    t = from_whisperx(result, max_word_s=1.2)
    assert t.words[0].end - t.words[0].start <= 1.2 + 1e-6


def test_from_whisperx_default_no_clamp_and_diarized_flag():
    result = {"language": "en", "segments": [{"words": [
        {"word": "x", "start": 0.0, "end": 5.0, "speaker": "S0"}]}]}
    assert from_whisperx(result).words[0].end == 5.0   # default: no clamp
    assert from_whisperx(result, diarized=True).diarized is True


def test_diarized_persists_roundtrip(tmp_path):
    t = Transcript([Word("a", 0.0, 0.4, "S0")], single_speaker=True, diarized=True)
    t2 = Transcript.load(t.save(tmp_path / "t.json"))
    assert t2.diarized is True


# ---- repetition-collapse / runaway-token post-guard -------------------------

def test_sanitize_collapses_char_runs_keeps_normal_words():
    from gameplay.transcript import sanitize_runaway_tokens
    ws = [Word("Naaaaaaaaaaaa", 3.1, 4.3, "S0"), Word("cool", 0.0, 0.4, "S0")]
    out = sanitize_runaway_tokens(ws, max_chars=40)
    assert out[0].text == "Na"        # run of a's collapsed to one
    assert out[1].text == "cool"      # 'oo' (run of 2) untouched
    assert out[1] is ws[1]            # well-formed word passed through unchanged (identity)


def test_repetition_collapse_token_repaired_and_gap_survives():
    # The reported failure: a 300-char "Naaaa…" token at 3.1–4.3s, a well-formed
    # word before it, and a long gap to the next word at 32.23s. Through the adapter
    # + post-guard the wall must NOT reach captions, real words pass unchanged, and
    # the gap (real timing) is preserved.
    runaway = "N" + "a" * 300
    result = {"language": "en", "segments": [{"words": [
        {"word": "go", "start": 0.0, "end": 0.4, "speaker": "S0"},
        {"word": runaway, "start": 3.1, "end": 4.3, "speaker": "S0"},
        {"word": "clutch", "start": 32.23, "end": 32.8, "speaker": "S0"},
    ]}]}
    t = from_whisperx(result, max_word_s=1.2, max_word_chars=40)
    texts = [w.text for w in t.words]
    assert all(len(x) <= 40 for x in texts)          # no 300-char wall reaches captions
    assert "go" in texts and "clutch" in texts        # well-formed words survive
    naa = next(w for w in t.words if w.text.startswith("N"))
    assert naa.text == "Na"                            # collapsed, still editable
    assert naa.end - naa.start <= 1.2 + 1e-6           # duration also clamped
    # the real gap is untouched (3.1→4.3 word, then 32.23 word)
    assert t.words[-1].start == 32.23


def test_garbage_token_with_no_char_run_is_dropped():
    # A long token with no 4+ char run can't be repaired -> dropped (never captions).
    garbage = "abcdefghij" * 5                        # 50 chars, no repeats to collapse
    result = {"language": "en", "segments": [{"words": [
        {"word": "ok", "start": 0.0, "end": 0.3},
        {"word": garbage, "start": 1.0, "end": 1.2}]}]}
    t = from_whisperx(result, max_word_chars=40)
    assert [w.text for w in t.words] == ["ok"]


def test_post_guard_off_by_default_leaves_long_words():
    # max_word_chars defaults to 0 (disabled) -> the lore path is unaffected.
    runaway = "N" + "a" * 300
    result = {"language": "en", "segments": [{"words": [
        {"word": runaway, "start": 0.0, "end": 0.5}]}]}
    assert from_whisperx(result).words[0].text == runaway
