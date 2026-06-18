"""Profanity censor — pure matching / masking / span / filter logic (no GPU/ffmpeg)."""
from gameplay import censor


# ---- whole-word, case-insensitive matcher + allow-list ----------------------

def test_matcher_basic_and_case_insensitive():
    assert censor.is_censored("fuck")
    assert censor.is_censored("Fuck")
    assert censor.is_censored("that was FUCKING great")     # token within a phrase
    assert censor.is_censored("oh shit!")                   # trailing punctuation
    assert not censor.is_censored("that was great")
    assert not censor.is_censored("")


def test_matcher_is_whole_word_not_substring():
    # the classic false positives — must NOT trip (substring "ass"/"shit"/"cock")
    for clean in ("Shaco", "assassin", "Cassiopeia", "class", "passing",
                  "grass", "cockpit", "shitake", "Scunthorpe"):
        assert not censor.is_censored(clean), clean


def test_matcher_respects_custom_lists():
    assert censor.is_censored("noob", wordlist=["noob"], allowlist=[])
    assert not censor.is_censored("noob", wordlist=["noob"], allowlist=["noob"])


# ---- caption mask -----------------------------------------------------------

def test_mask_text_stars_and_block():
    assert censor.mask_text("fucking great", style="stars") == "f****** great"
    assert censor.mask_text("oh fuck", style="block") == "oh [bleep]"
    # punctuation preserved around the masked core
    assert censor.mask_text("fuck!", style="stars") == "f***!"
    # clean words untouched
    assert censor.mask_text("nice shaco play") == "nice shaco play"


# ---- span merge -------------------------------------------------------------

def test_merge_spans_pads_clamps_and_merges():
    merged = censor.merge_spans([(1.0, 1.2), (1.25, 1.4), (5.0, 5.3)],
                                pad=0.05, dur=5.2)
    # first two overlap after padding -> one span; third clamped to dur
    assert merged[0] == (0.95, 1.45)
    assert merged[1][1] == 5.2
    assert len(merged) == 2


def test_merge_spans_empty():
    assert censor.merge_spans([], pad=0.1) == []


# ---- audio filtergraph ------------------------------------------------------

def test_audio_graph_modes():
    spans = [(0.5, 0.9), (2.1, 2.4)]
    mute = censor.audio_graph(spans, dur=3.0, mode="mute")
    assert "volume=0:enable='between(t,0.500,0.900)+between(t,2.100,2.400)'" in mute

    duck = censor.audio_graph(spans, dur=3.0, mode="duck", duck_gain=0.2)
    assert "volume=0.2:enable=" in duck

    bleep = censor.audio_graph(spans, dur=3.0, mode="bleep", hz=1000)
    assert "sine=frequency=1000" in bleep and "amix=inputs=2" in bleep
    assert "not(between(t," in bleep            # tone gated to the spans


def test_audio_graph_no_spans_is_noop():
    assert censor.audio_graph([], dur=3.0, mode="bleep") is None


# ---- transcript integration (flag column + spans + caption mask) ------------

def test_from_whisperx_auto_flags_profanity():
    from gameplay.transcript import from_whisperx
    res = {"language": "en", "segments": [{"words": [
        {"word": "nice", "start": 0.0, "end": 0.4},
        {"word": "shit", "start": 0.4, "end": 0.8},
        {"word": "shaco", "start": 0.8, "end": 1.2}]}]}
    t = from_whisperx(res)
    flags = {w.text: w.censor for w in t.words}
    assert flags == {"nice": False, "shit": True, "shaco": False}   # allow-list clean


def test_censor_column_round_trips_and_drives_spans_and_mask():
    from gameplay.transcript import Transcript, Word
    t = Transcript([Word("oh", 0.0, 0.3), Word("fuck", 0.3, 0.7, censor=True)])
    rows = t.to_rows()
    assert rows[1] == ["fuck", "", 0.3, 0.7, True]          # censor flag column
    t2 = Transcript.from_rows(rows)
    assert t2.words[1].censor is True and t2.censor_spans() == [(0.3, 0.7)]
    # caption tuples mask only when asked
    assert t2.to_tuples()[1][0] == "fuck"
    assert t2.to_tuples(mask=True)[1][0] == "f***"


def test_untick_censor_flag_clears_the_hit():
    from gameplay.transcript import Transcript
    # user un-ticks the censor checkbox on a flagged row -> no audio span, no mask
    t = Transcript.from_rows([["fuck", "", 0.3, 0.7, False]])
    assert t.censor_spans() == [] and t.to_tuples(mask=True)[0][0] == "fuck"
