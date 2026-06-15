"""EXPERIMENTAL full-auto logic — the parts that don't need a GPU or an LLM:
candidate parsing, energy+LLM merge/rank, transcript slicing for auto-cuts."""
from gameplay import config as gconf
from gameplay.autohighlight import (Candidate, _parse_candidates,
                                    rank_candidates, llm_candidates)
from gameplay.autopipeline import slice_transcript
from gameplay.transcript import Transcript, Word


def test_parse_candidates_from_messy_llm_output():
    raw = ('Sure! Here are the highlights:\n'
           '[{"start": 12, "end": 28.5, "category": "FUNNY", "caption": "he fell off"},'
           ' {"start": 40, "end": 50, "category": "banana", "caption": "clutch 1v3"}]\n'
           'Hope that helps!')
    cands = _parse_candidates(raw)
    assert len(cands) == 2
    assert cands[0].category == "funny"           # lowercased
    assert cands[1].category == "story"           # invalid -> default
    assert cands[0].start == 12.0 and cands[0].end == 28.5


def test_parse_candidates_handles_garbage():
    assert _parse_candidates("no json here") == []
    assert _parse_candidates('[{"start": 5}]') == []   # missing end -> skipped


def test_llm_candidates_none_backend_is_empty():
    t = Transcript([Word("hi", 0.0, 0.4, "S0")])
    assert llm_candidates(t, backend="none") == []


def test_rank_merges_overlapping_energy_and_llm():
    energy = [Candidate(10, 18, "highlight", "", 1.0, "energy"),   # overlaps llm
              Candidate(100, 108, "highlight", "", 0.5, "energy")] # standalone
    llm = [Candidate(12, 20, "clutch", "huge clutch", 1.0, "llm")]
    ranked = rank_candidates(energy, llm)
    top = ranked[0]
    assert top.source == "energy+llm"
    assert top.category == "clutch" and top.caption == "huge clutch"
    assert top.score > 1.0                         # boosted (loud AND interesting)
    # the standalone energy window survives, de-overlapped
    assert any(c.start == 100 for c in ranked)


def test_rank_keeps_unmatched_llm():
    ranked = rank_candidates([], [Candidate(5, 15, "rage", "tilted", 1.0, "llm")])
    assert len(ranked) == 1 and ranked[0].category == "rage"


def test_rank_deoverlaps_keeping_strongest():
    a = Candidate(0, 10, "story", "a", 2.0, "llm")
    b = Candidate(5, 15, "funny", "b", 1.0, "llm")   # overlaps a, weaker
    ranked = rank_candidates([], [a, b])
    assert len(ranked) == 1 and ranked[0].caption == "a"


def test_slice_transcript_rebases_to_zero():
    t = Transcript([Word("before", 2.0, 2.5, "S0"),
                    Word("inside", 11.0, 11.5, "S0"),
                    Word("edge", 19.5, 20.5, "S1"),     # straddles end -> included
                    Word("after", 30.0, 30.5, "S0")])
    sub = slice_transcript(t, 10.0, 20.0)
    texts = [w.text for w in sub.words]
    assert texts == ["inside", "edge"]
    assert sub.words[0].start == 1.0                   # 11.0 - 10.0, rebased
    assert sub.words[0].start >= 0.0
