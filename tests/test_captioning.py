"""Gameplay caption chunking: word->phrase grouping, offset, word mode, and proof
that the cues use the real word-aligned times (not evenly distributed). No GPU."""
from gameplay import captioning as cap
from gameplay import config as gconf
from gameplay.transcript import Word


def _words(spec):
    # spec: list of (text, start, end[, speaker])
    return [Word(t, s, e, (sp[0] if sp else None)) for t, s, e, *sp in spec]


def test_chunk_groups_by_word_window_and_char_limits():
    w = _words([("the", 0.0, 0.2), ("enemy", 0.25, 0.5), ("is", 0.55, 0.7),
                ("very", 0.75, 0.9), ("low", 0.95, 1.1)])
    cues = cap.chunk_words(w, max_words=4, max_window_s=5, max_chars=99)
    assert cues[0] == ("the enemy is very", 0.0, 0.9, None)   # capped at 4 words
    assert cues[1] == ("low", 0.95, 1.1, None)


def test_chunk_window_limit_splits():
    w = _words([("hold", 0.0, 0.3), ("on", 2.0, 2.3)])         # 2.3s span > window
    cues = cap.chunk_words(w, max_words=9, max_window_s=1.2, max_chars=99)
    assert len(cues) == 2


def test_chunk_char_limit_splits():
    w = _words([("absolutely", 0.0, 0.4), ("incredible", 0.5, 0.9)])
    cues = cap.chunk_words(w, max_words=9, max_window_s=9, max_chars=12)
    assert [c[0] for c in cues] == ["absolutely", "incredible"]


def test_chunk_splits_on_speaker_change():
    w = _words([("nice", 0.0, 0.3, "S0"), ("one", 0.35, 0.6, "S1")])
    cues = cap.chunk_words(w, max_words=9, max_window_s=9, max_chars=99)
    assert len(cues) == 2 and cues[0][3] == "S0" and cues[1][3] == "S1"


def test_cues_are_non_overlapping_and_cover_the_span():
    w = _words([("a", 0.0, 0.2), ("b", 0.3, 0.5), ("c", 0.6, 0.8), ("d", 0.9, 1.1)])
    cues = cap.chunk_words(w, max_words=2, max_window_s=9, max_chars=99)
    for i in range(len(cues) - 1):
        assert cues[i][2] <= cues[i + 1][1]               # no overlap
    assert cues[0][1] == 0.0 and cues[-1][2] == 1.1       # covers first..last


def test_global_offset_shifts_every_cue():
    w = _words([("go", 1.0, 1.4), ("now", 1.5, 1.9)])
    cues = cap.chunk_words(w, offset=0.25, max_words=1)
    assert cues[0][1] == 1.25 and cues[0][2] == 1.65
    assert cues[1][1] == 1.75


def test_word_mode_is_one_per_word():
    w = _words([("a", 0.0, 0.2), ("b", 0.3, 0.5)])
    cues = cap.word_tuples(w)
    assert len(cues) == 2 and cues[0] == ("a", 0.0, 0.2, None)


def test_caption_cues_dispatches_by_mode():
    w = _words([("x", 0.0, 0.2), ("y", 0.25, 0.4)])
    assert len(cap.caption_cues(w, mode="word")) == 2
    assert len(cap.caption_cues(w, mode="phrase", offset=0)) == 1   # grouped


def test_cues_preserve_word_aligned_times_not_even_distribution():
    # Real wav2vec2 alignment is non-uniform. If a regression evenly-distributed words
    # across the span, the per-word boundaries below would become uniform — this guards
    # that the cue times come straight from the words.
    w = _words([("quick", 0.10, 0.18), ("pause", 0.18, 0.22), ("then", 1.90, 2.30)])
    cues = cap.word_tuples(w)
    assert [c[1] for c in cues] == [0.1, 0.18, 1.9]       # exact word starts, uneven
    assert [c[2] for c in cues] == [0.18, 0.22, 2.3]


def test_phrase_mode_masks_censored_words():
    w = [Word("oh", 0.0, 0.3), Word("shit", 0.35, 0.6, censor=True)]
    cues = cap.chunk_words(w, max_words=9, max_window_s=9, max_chars=99,
                           mask=True, mask_style="stars")
    assert cues[0][0] == "oh s***"


def test_caption_style_wires_min_dur_and_gap():
    from gameplay.manual import caption_style, ManualOptions
    st = caption_style(ManualOptions())
    assert st.min_hold_s == gconf.CAPTION_MIN_DUR_S
    assert st.max_gap == gconf.CAPTION_MAX_GAP_S
