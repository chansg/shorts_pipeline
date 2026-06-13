"""SFX timing + placement math, and the deterministic filter graph. All pure —
no ffmpeg is invoked (looped placements avoid the ffprobe duration lookup)."""
import math

import pytest

from modules.audio_mix import (Placement, build_filter, db_to_linear,
                               find_word_time, pan_gains, resolve_cue_time,
                               resolve_placements)
from modules.captions import Word
from orchestrator.audio_spec import AudioSpec, Cue, load_sfx_map, resolve_source

SFX_MAP = load_sfx_map()
KNOCK = resolve_source("knock_wood", SFX_MAP)
WIND = resolve_source("wind_hall", SFX_MAP)

SCENES = [
    {"file": "01.png", "start": 0.0, "duration": 3.0},
    {"file": "02.png", "start": 3.0, "duration": 4.0},
    {"file": "03.png", "start": 7.0, "duration": 3.0},
]
WORDS = [
    Word("the", 0.5, 0.7), Word("door", 1.0, 1.4),
    Word("keeps", 1.5, 1.8), Word("knocking", 2.0, 2.6),
    Word("knocking", 8.0, 8.6),  # second occurrence in scene 3
]
TOTAL = 10.0


# ---- scalar helpers -------------------------------------------------------

def test_db_to_linear():
    assert db_to_linear(0) == pytest.approx(1.0)
    assert db_to_linear(-6) == pytest.approx(0.5012, abs=1e-3)
    assert db_to_linear(6) == pytest.approx(1.995, abs=1e-3)


def test_pan_gains_constant_power():
    l, r = pan_gains(0.0)
    assert l == pytest.approx(r) == pytest.approx(math.sqrt(0.5), abs=1e-3)
    assert pan_gains(-1.0) == pytest.approx((1.0, 0.0), abs=1e-6)
    assert pan_gains(1.0) == pytest.approx((0.0, 1.0), abs=1e-6)


# ---- word anchoring -------------------------------------------------------

def test_find_word_time_first_and_second_occurrence():
    assert find_word_time(WORDS, "knocking", 1) == 2.0
    assert find_word_time(WORDS, "knocking", 2) == 8.0
    assert find_word_time(WORDS, "Knocking!", 1) == 2.0  # case/punct-insensitive
    assert find_word_time(WORDS, "missing") is None


# ---- single-cue resolution ------------------------------------------------

def _cue(layer, anchor=None, **kw):
    return Cue(source="knock_wood", path=KNOCK, layer=layer,
              label=f"test-{layer}", anchor=anchor, **kw)


def test_resolve_scene_anchor_uses_scene_start_plus_offset():
    c = _cue("oneshot", {"scene": 2, "offset": 0.5})
    assert resolve_cue_time(c, SCENES, WORDS) == pytest.approx(3.5)


def test_resolve_word_anchor():
    c = _cue("oneshot", {"word": "knocking", "occurrence": 1, "offset": 0.1})
    assert resolve_cue_time(c, SCENES, WORDS) == pytest.approx(2.1)


def test_resolve_time_anchor():
    c = _cue("motif", {"time": 4.2})
    assert resolve_cue_time(c, SCENES, WORDS) == pytest.approx(4.2)


def test_beds_resolve_to_zero():
    c = _cue("ambient_bed")
    assert resolve_cue_time(c, SCENES, WORDS) == 0.0


def test_scene_out_of_range_raises():
    c = _cue("oneshot", {"scene": 9})
    with pytest.raises(ValueError) as e:
        resolve_cue_time(c, SCENES, WORDS)
    assert "out of range" in str(e.value)


def test_word_not_found_raises():
    c = _cue("oneshot", {"word": "nonexistent"})
    with pytest.raises(ValueError) as e:
        resolve_cue_time(c, SCENES, WORDS)
    assert "not found" in str(e.value)


# ---- placement resolution (ordering, bounds) ------------------------------

def test_resolve_placements_stable_order_bed_motif_oneshot():
    spec = AudioSpec(
        ambient_bed=Cue("wind_hall", WIND, "ambient_bed", "bed", loop=True),
        motifs=[_cue("motif", {"time": 5.0})],
        oneshots=[_cue("oneshot", {"word": "knocking"})],
    )
    pl = resolve_placements(spec, SCENES, WORDS, TOTAL)
    assert [p.label for p in pl] == ["bed", "test-motif", "test-oneshot"]
    assert [round(p.start, 1) for p in pl] == [0.0, 5.0, 2.0]


def test_placement_start_after_end_raises():
    spec = AudioSpec(motifs=[_cue("motif", {"time": 99.0})])
    with pytest.raises(ValueError) as e:
        resolve_placements(spec, SCENES, WORDS, TOTAL)
    assert "after the end" in str(e.value)


def test_resolve_placements_reports_all_errors():
    spec = AudioSpec(
        motifs=[_cue("motif", {"scene": 99}), _cue("motif", {"time": 99.0})])
    with pytest.raises(ValueError) as e:
        resolve_placements(spec, SCENES, WORDS, TOTAL)
    msg = str(e.value)
    assert "out of range" in msg and "after the end" in msg


# ---- filter graph (deterministic, no ffmpeg) ------------------------------

def _looped(label, start, **kw):
    return Placement(path=WIND, start=start, gain_db=kw.get("gain_db", -12),
                    pan=kw.get("pan"), fade_in=kw.get("fade_in", 0.0),
                    fade_out=kw.get("fade_out", 0.0), loop=True, label=label)


def test_build_filter_uses_normalize0_and_adelay():
    pl = [_looped("a", 0.0), _looped("b", 2.0)]
    filt = build_filter(pl, TOTAL, duck_enabled=False, duck_amount_db=8,
                        duck_threshold=0.05)
    assert "amix=inputs=2:normalize=0" in filt   # never auto-normalize
    assert "adelay=2000:all=1" in filt           # 2.0s → 2000ms delay
    assert "[vo][sfxmix]amix=inputs=2:normalize=0" in filt


def test_build_filter_single_cue_uses_anull_not_amix():
    filt = build_filter([_looped("solo", 1.0)], TOTAL, False, 8, 0.05)
    assert "[s0]anull[sfxmix]" in filt


def test_build_filter_ducking_adds_sidechain():
    filt = build_filter([_looped("a", 0.0)], TOTAL, duck_enabled=True,
                        duck_amount_db=8, duck_threshold=0.05)
    assert "sidechaincompress" in filt
    assert "asplit=2[vo][vokey]" in filt
    assert "[vo][sfxduck]amix=inputs=2:normalize=0" in filt


def test_build_filter_applies_pan_and_fades():
    pl = [_looped("a", 0.0, pan=-0.5, fade_in=1.0, fade_out=2.0)]
    filt = build_filter(pl, TOTAL, False, 8, 0.05)
    assert "pan=stereo" in filt
    assert "afade=t=in:st=0:d=1.000" in filt
    assert "afade=t=out" in filt
