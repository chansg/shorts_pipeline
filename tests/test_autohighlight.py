"""Fused highlight detection — pure functions on synthetic inputs (no GPU/ffmpeg/
LLM), plus a mocked-LLM end-to-end that prints the dry-run candidate shape."""
import numpy as np
import pytest

from fullauto import highlight as ah
from gameplay import config as gconf
from fullauto.highlight import (Anchor, Candidate, LLMMoment, chunk_transcript,
                                fuse_and_rank, frame_window, merge_windows,
                                parse_llm_moments, pick_energy_peaks,
                                scan_reactions)
from gameplay.transcript import Transcript, Word


# ---- A. energy peaks ----

def test_pick_energy_peaks_spacing_and_prominence():
    # flat baseline with three spikes; window 0.5s -> times 0.25,0.75,...
    rms = np.full(120, 0.1)
    for idx in (20, 40, 60):           # 10s, 20s, 30s
        rms[idx] = 1.0
    times = (np.arange(120) + 0.5) * 0.5
    peaks = pick_energy_peaks(times, rms, k=1.0, roll_s=10.0, min_prominence=0.1,
                              min_spacing_s=8.0, window_s=0.5)
    pt = [round(t, 2) for t, _ in peaks]
    assert pt == [10.25, 20.25, 30.25]
    assert all(0.0 <= p <= 1.0 for _, p in peaks)
    assert max(p for _, p in peaks) == 1.0      # normalised to the max


def test_pick_energy_peaks_enforces_min_spacing():
    rms = np.full(60, 0.1)
    rms[20] = 1.0
    rms[22] = 0.9                       # 1s later -> within 8s spacing, must drop
    times = (np.arange(60) + 0.5) * 0.5
    peaks = pick_energy_peaks(times, rms, k=1.0, roll_s=10.0, min_prominence=0.05,
                              min_spacing_s=8.0, window_s=0.5)
    assert len(peaks) == 1 and round(peaks[0][0], 2) == 10.25


def test_pick_energy_peaks_empty_on_flat():
    rms = np.full(50, 0.2)
    times = (np.arange(50) + 0.5) * 0.5
    assert pick_energy_peaks(times, rms) == []


# ---- A. reaction scan ----

def test_scan_reactions_category_hints():
    t = Transcript([
        Word("let's", 1.0, 1.2, "S0"), Word("go", 1.2, 1.4, "S0"),
        Word("that", 1.4, 1.6, "S0"), Word("was", 1.6, 1.8, "S0"),
        Word("insane", 1.8, 2.2, "S0"),
        Word("bro", 5.0, 5.3, "S1"), Word("lol", 5.3, 5.6, "S1"),
    ], single_speaker=False)
    lex = {"hype": ["let's go", "insane"], "funny": ["bro", "lol"]}
    anchors = ah.scan_reactions(t, lex)
    hype = [a for a in anchors if a.category_hint == "hype"]
    funny = [a for a in anchors if a.category_hint == "funny"]
    assert hype and hype[0].time == 1.0 and hype[0].strength == 2.0   # 2 phrases
    assert funny and funny[0].time == 5.0 and funny[0].strength == 2.0


# ---- B. framing ----

def _ramp_transcript(n=40, step=1.0):
    # word i spans [i, i+0.5]; every 5th word ends a sentence
    words = []
    for i in range(n):
        txt = "word." if (i + 1) % 5 == 0 else "word"
        words.append(Word(txt, i * step, i * step + 0.5, "S0"))
    return Transcript(words, single_speaker=True)


def test_frame_window_snaps_and_clamps():
    t = _ramp_transcript(40)
    s, e = frame_window(20.0, t, total=40.0, lead_in=4.0, lead_out=7.0,
                        cmin=15.0, cmax=45.0)
    assert s <= 16.0 and s == 16.0          # snapped to a word start <= 16
    assert e >= 27.0                         # snapped to a sentence end >= 27
    assert 15.0 <= (e - s) <= 45.0           # within clip bounds


def test_frame_window_min_duration_enforced():
    t = _ramp_transcript(40)
    s, e = frame_window(2.0, t, total=40.0, lead_in=1.0, lead_out=1.0,
                        cmin=15.0, cmax=45.0)
    assert (e - s) >= 15.0


def test_merge_windows_unions_overlaps():
    assert merge_windows([(0, 10), (5, 15), (20, 25)]) == [(0, 15), (20, 25)]


# ---- C. LLM chunking + parse ----

def test_chunk_transcript_overlap_and_last_partial():
    words = [Word("w", float(i), float(i) + 0.5, "S0") for i in range(0, 300, 1)]
    t = Transcript(words, single_speaker=True)
    chunks = chunk_transcript(t, chunk_s=150.0, overlap_s=20.0)
    starts = [c[0] for c in chunks]
    assert starts[0] == 0.0
    assert starts[1] == 130.0               # step = chunk - overlap
    assert chunks[-1][1] <= 300.0           # last chunk clamped to total
    # consecutive chunks overlap by ~20s
    assert chunks[0][1] - chunks[1][0] == pytest.approx(20.0, abs=0.5)


def test_parse_llm_moments_messy_and_malformed():
    raw = ('```json\n[{"start": 12, "end": 28.5, "category": "FUNNY", '
           '"hook_caption": "he fell off", "confidence": 0.9},'
           '{"start": 40, "end": 50, "category": "banana", "confidence": 2.0},'
           '{"start": 60}]\n```')
    moments = parse_llm_moments(raw)
    assert len(moments) == 2                       # the third (no end) is skipped
    assert moments[0].category == "funny" and moments[0].caption == "he fell off"
    assert moments[1].category == "story"          # invalid -> default
    assert moments[1].confidence == 1.0            # clamped 2.0 -> 1.0


def test_parse_llm_moments_garbage_is_empty():
    assert parse_llm_moments("no json here at all") == []
    assert parse_llm_moments("") == []


# ---- D. fuse + rank ----

def test_fuse_scores_and_dedupes():
    t = _ramp_transcript(80)              # 0..80s, one speaker
    energy = [Anchor(20.0, "energy", strength=1.0)]
    reactions = [Anchor(20.0, "reaction", strength=2.0, category_hint="hype")]
    llm = [LLMMoment(16.0, 30.0, "clutch", "huge clutch", "1v3", 0.9)]
    out = fuse_and_rank(energy, reactions, llm, t, total=80.0, top_n=15)
    assert out, "expected at least one candidate"
    top = out[0]
    # overlapping energy+reaction+llm windows dedupe to one, LLM wins category/caption
    assert top.category == "clutch" and top.caption == "huge clutch"
    assert "llm" in top.source and "energy" in top.source
    assert top.score > 0.5                # LLM-led weight dominates
    # no two survivors overlap
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            assert out[i].end <= out[j].start or out[i].start >= out[j].end


def test_fuse_caps_to_top_n():
    t = _ramp_transcript(400)
    energy = [Anchor(float(x), "energy", strength=0.5) for x in range(10, 360, 20)]
    out = fuse_and_rank(energy, [], [], t, total=400.0, top_n=5)
    assert len(out) == 5


def test_fuse_is_deterministic():
    t = _ramp_transcript(120)
    energy = [Anchor(20.0, "energy", 0.8), Anchor(60.0, "energy", 0.6)]
    a = fuse_and_rank(energy, [], [], t, total=120.0)
    b = fuse_and_rank(energy, [], [], t, total=120.0)
    assert [(c.start, c.end, c.score) for c in a] == \
           [(c.start, c.end, c.score) for c in b]


# ---- mocked-LLM end-to-end (dry run) ----

def test_detect_highlights_end_to_end_mocked(monkeypatch, capsys):
    t = _ramp_transcript(200)
    # energy comes from the video; bypass ffmpeg by stubbing energy_anchors
    monkeypatch.setattr(ah, "energy_anchors",
                        lambda v: [Anchor(40.0, "energy", 0.9),
                                   Anchor(120.0, "energy", 0.5)])

    def fake_llm(system, user, backend):
        return ('[{"start": 38, "end": 60, "category": "clutch", '
                '"hook_caption": "clutch 1v3", "reason": "won the round", '
                '"confidence": 0.95}]')
    monkeypatch.setattr(ah, "_llm_raw", fake_llm)

    cands = ah.detect_highlights("dummy.mp4", t, backend="ollama", top_n=10)
    assert cands and isinstance(cands[0], Candidate)
    assert cands == sorted(cands, key=lambda c: -c.score)        # sorted by score
    top = cands[0]
    assert top.category == "clutch" and top.caption == "clutch 1v3"
    assert {"start", "end", "category", "caption", "score", "source"} <= set(vars(top))
    # dry-run shape for the summary
    print("\nDRY-RUN CANDIDATES:")
    for c in cands:
        print(f"  [{c.category:8}] {c.start:6.1f}-{c.end:6.1f}s  "
              f"score={c.score:.3f}  src={c.source:18}  {c.caption!r}")
