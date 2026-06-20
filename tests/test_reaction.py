"""Audio-reaction detector — PURE scoring / peak / window logic + the streaming fold.
No ffmpeg, no GPU: all on synthetic arrays. This is the robust core of full-auto."""
import numpy as np

from fullauto import reaction as rx


# ---- streaming fold (memory-bounded, block-boundary-safe) -------------------

def test_fold_blocks_to_window_rms_crosses_block_boundaries():
    # a steady signal of |0.5| folded in oddly-sized blocks must give RMS ~0.5 per
    # window regardless of where the block edges fall.
    win = 10
    sig = np.full(100, 0.5)
    blocks = [sig[0:7], sig[7:23], sig[23:55], sig[55:100]]   # unaligned to `win`
    rms = rx.fold_blocks_to_window_rms(iter(blocks), win)
    assert rms.size == 10                                  # 100 / 10 windows
    assert np.allclose(rms, 0.5)


def test_fold_drops_trailing_partial_window():
    rms = rx.fold_blocks_to_window_rms([np.ones(25)], 10)   # 2 full + 5 leftover
    assert rms.size == 2


def test_fold_is_streaming_not_full_decode():
    # The fold must never materialise the whole signal: feed a generator that would
    # blow up if fully realised, and assert only bounded blocks are pulled.
    pulled = {"blocks": 0, "samples": 0}

    def gen():
        for _ in range(1000):                              # "hours" of blocks
            pulled["blocks"] += 1
            pulled["samples"] += 1024
            yield np.full(1024, 0.1)

    rms = rx.fold_blocks_to_window_rms(gen(), 256)
    assert rms.size == (1000 * 1024) // 256
    # processed block-by-block (1000 pulls), not concatenated into one giant array
    assert pulled["blocks"] == 1000


# ---- the discriminator: sharp bursts picked, sustained swell rejected -------

def _synth_curve():
    """0.1s windows over 60s: calm vocal baseline + 3 sharp exclamation bursts +
    one long broadband teamfight swell (gradual onset)."""
    ws = rx.gconf.REACTION_WINDOW_S
    n = int(60 / ws)
    rng = np.random.default_rng(0)
    rms = 0.05 + 0.01 * rng.standard_normal(n)
    rms = np.abs(rms)
    # 3 sharp bursts (one-window attack to a high value) at 8s, 25s, 50s
    for t in (8.0, 25.0, 50.0):
        i = int(t / ws)
        rms[i:i + 2] = 0.6                                  # fast attack, brief
    # a sustained swell 35-45s: ramps up slowly and stays loud (no sharp onset)
    a, b = int(35 / ws), int(45 / ws)
    ramp = np.linspace(0.05, 0.45, b - a)
    rms[a:b] = ramp
    times = (np.arange(n) + 0.5) * ws
    return times, rms


def test_three_sharp_bursts_are_peaks_and_swell_is_not():
    times, rms = _synth_curve()
    score = rx.reaction_score(times, rms)
    peaks = rx.pick_reaction_peaks(times, score)
    pts = [t for t, _ in peaks]
    # the three exclamations are detected (within a window of their injected time)
    for want in (8.0, 25.0, 50.0):
        assert any(abs(p - want) < 0.5 for p in pts), (want, pts)
    # the sustained swell (35-45s) is NOT picked — suddenness beats sustained energy
    assert not any(35.0 <= p <= 45.0 for p in pts), pts


def test_score_is_normalised_and_nonnegative():
    times, rms = _synth_curve()
    score = rx.reaction_score(times, rms)
    assert score.min() >= 0.0 and abs(score.max() - 1.0) < 1e-9


def test_peak_spacing_collapses_a_cluster_to_one():
    ws = rx.gconf.REACTION_WINDOW_S
    n = 600
    rms = np.full(n, 0.05)
    for i in (100, 103, 106):                              # 3 spikes within 0.6s
        rms[i] = 0.8
    times = (np.arange(n) + 0.5) * ws
    score = rx.reaction_score(times, rms)
    peaks = rx.pick_reaction_peaks(times, score, min_spacing_s=6.0)
    assert len(peaks) == 1                                 # one reaction, one peak


# ---- generous windows: anchor before + merge -------------------------------

def test_windows_anchor_before_the_spike():
    wins = rx.candidate_windows([(20.0, 1.0)], total=100.0,
                                pre_roll=8.0, post_roll=10.0, merge_gap=3.0)
    assert len(wins) == 1
    w = wins[0]
    assert w.start == 12.0 and w.end == 30.0              # 8s before, 10s after
    assert w.start < 20.0 < w.end                         # spike sits inside, setup ahead
    assert w.peaks == [20.0]


def test_overlapping_windows_merge_into_one_candidate():
    # two peaks 5s apart -> their windows overlap -> one merged candidate
    wins = rx.candidate_windows([(20.0, 0.8), (25.0, 1.0)], total=100.0,
                                pre_roll=8.0, post_roll=10.0, merge_gap=3.0)
    assert len(wins) == 1
    assert wins[0].start == 12.0 and wins[0].end == 35.0
    assert wins[0].audio_score == 1.0                     # carries the max score
    assert wins[0].peaks == [20.0, 25.0]


def test_far_apart_peaks_stay_separate_and_clamp_to_bounds():
    wins = rx.candidate_windows([(5.0, 0.5), (90.0, 0.9)], total=95.0,
                                pre_roll=8.0, post_roll=10.0, merge_gap=3.0)
    assert len(wins) == 2
    assert wins[0].start == 0.0                            # clamped at 0
    assert wins[1].end == 95.0                             # clamped at total


def test_candidate_cap_keeps_highest_scoring():
    peaks = [(float(10 * i), 0.1 * (i + 1)) for i in range(10)]   # spaced, rising score
    wins = rx.candidate_windows(peaks, total=200.0, pre_roll=1.0, post_roll=1.0,
                                merge_gap=0.0, max_candidates=3)
    assert len(wins) == 3
    kept_scores = sorted(w.audio_score for w in wins)
    assert kept_scores == [0.8, 0.9, 1.0]                 # the top 3 survived


def test_no_peaks_yields_no_windows():
    assert rx.candidate_windows([], total=100.0) == []
