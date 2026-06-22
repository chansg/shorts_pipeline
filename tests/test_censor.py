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


def test_matcher_stems_catch_variants_and_compounds():
    # the sensitivity lever: stems flag variants/compounds without listing each
    for w in ("fucking", "fucked", "motherfucker", "bullshit", "dipshit",
              "wankers", "bitches", "shitty", "clusterfuck"):
        assert censor.is_censored(w), w


def test_matcher_allowlist_guards_stem_false_positives():
    # stems must NOT trip these clean words (some contain a stem as a substring)
    for clean in ("Shaco", "assassin", "Cassiopeia", "class", "passing",
                  "grass", "cockpit", "shitake", "Scunthorpe", "niggle",
                  "retardant", "compass"):
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


def test_profane_text_auto_censors_even_if_unticked():
    from gameplay.transcript import Transcript
    # Auto-detection wins: a profane word is censored even if the box is unticked
    # (the tick can only ADD censor). To KEEP a profane word, allow-list it.
    t = Transcript.from_rows([["fuck", "", 0.3, 0.7, False]])
    assert t.censor_spans() == [(0.3, 0.7)] and t.to_tuples(mask=True)[0][0] == "f***"


def test_manual_tick_adds_censor_to_a_non_listed_word():
    from gameplay.transcript import Transcript
    # a word the lists don't catch is censored when the user ticks the box (e.g. a
    # manually-added row), and stays clean when not.
    t = Transcript.from_rows([["noob", "", 1.0, 1.4, True]])
    assert t.censor_spans() == [(1.0, 1.4)]
    assert Transcript.from_rows([["noob", "", 1.0, 1.4, False]]).censor_spans() == []


def test_added_row_with_blank_timing_keeps_censor():
    from gameplay.transcript import Transcript
    # the reported bug: a right-click-added row (blank timing) must survive AND its
    # profane text be censored — timing inferred, not dropped.
    t = Transcript.from_rows([["aim", "S0", 0.0, 0.4, False],
                              ["bullshit", "", "", "", False]])   # added, no timing
    assert [w.text for w in t.words] == ["aim", "bullshit"]
    assert t.words[1].censor is True                              # stem-flagged
    assert t.censor_spans() and t.censor_spans()[-1][0] == 0.4    # inferred after "aim"


def test_added_rows_with_default_zero_timing_infer_in_place():
    from gameplay.transcript import Transcript
    # Gradio fills a new row's number cells with 0/0 (not blank). A real word never has
    # end==start==0, so 0/0 means "added row" -> infer timing after the previous word
    # instead of pinning it (and its censor) to t=0.
    t = Transcript.from_rows([["spray", "S0", 7.4, 7.7, False],
                              ["what", "", 0, 0, False],
                              ["the", "", 0, 0, False],
                              ["fuck?", "", 0, 0, False]])     # right-click-added, 0/0
    assert [w.text for w in t.words] == ["spray", "what", "the", "fuck?"]
    assert t.words[1].start == 7.7                             # inferred after "spray"
    assert t.words[3].censor is True                          # 'fuck?' auto-flagged
    assert t.censor_spans() and t.censor_spans()[0][0] >= 7.7  # bleep not at t=0


def test_editor_censor_toggle_and_autocensor_at_build():
    # the editor's 🔇 toggle + Transcript.from_rows auto-censor replace the old grid
    # censor-cell coercion: a profane word auto-censors and a manual tick is honoured.
    from gameplay.transcript import Transcript
    t = Transcript.from_rows([["nice", "", 0.0, 0.4, False],     # clean -> not censored
                              ["fuck?", "", 0.5, 0.9, False],    # profane -> auto
                              ["noob", "", 1.0, 1.4, True]])     # manual tick honoured
    flags = {w.text: w.censor for w in t.words}
    assert flags == {"nice": False, "fuck?": True, "noob": True}
