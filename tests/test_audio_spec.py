"""Timeline parsing + validation. These are the guards against the silent-fail
problem: malformed audio data must raise AudioSpecError listing every issue."""
import pytest

from orchestrator.audio_spec import (AudioSpecError, SFX_DIR, parse_audio_spec,
                                      resolve_source, load_sfx_map)

SFX_MAP = load_sfx_map()


def test_empty_manifest_parses_to_empty_spec():
    assert parse_audio_spec({}).is_empty()


def test_legacy_manifest_without_audio_is_empty():
    # A pre-SFX manifest (clips only) must parse to an empty spec, untouched.
    m = {"defaults": {}, "clips": [{"image": "01.png", "name": "s1"}]}
    spec = parse_audio_spec(m)
    assert spec.is_empty()
    assert spec.ambient_bed is None and not spec.oneshots


def test_valid_full_spec():
    m = {
        "audio": {
            "ambient_bed": {"source": "wind_hall", "gain_db": -22, "loop": True,
                            "fade_in": 2, "fade_out": 3},
            "motifs": [{"source": "rot_shimmer",
                        "at": {"scene": 2, "offset": 0.2},
                        "gain_db": -14, "pan": -0.4}],
            "ducking": {"enabled": True, "amount_db": 8, "threshold": 0.05},
        },
        "clips": [
            {"image": "01.png", "name": "s1",
             "sfx": [{"source": "knock_wood", "at": {"word": "knocking"},
                      "gain_db": -6, "pan": 0.3}]},
            {"image": "02.png", "name": "s2"},
        ],
    }
    spec = parse_audio_spec(m)
    assert spec.ambient_bed.source == "wind_hall"
    assert spec.ambient_bed.loop is True
    assert len(spec.motifs) == 1 and spec.motifs[0].pan == -0.4
    assert len(spec.oneshots) == 1
    assert spec.oneshots[0].anchor["word"] == "knocking"
    assert spec.oneshots[0].scene_index == 1
    assert spec.duck_enabled and spec.duck_amount_db == 8
    assert len(spec.all_cues()) == 3


def test_bed_loop_defaults_true():
    spec = parse_audio_spec({"audio": {"music_bed": {"source": "wind_hall"}}})
    assert spec.music_bed.loop is True


def test_unknown_source_raises_with_known_tags_listed():
    m = {"audio": {"motifs": [{"source": "nope", "at": {"time": 1.0}}]}}
    with pytest.raises(AudioSpecError) as e:
        parse_audio_spec(m)
    assert "not a known tag" in str(e.value)
    assert "knock_wood" in str(e.value)  # suggests available tags


def test_conflicting_anchor_keys_raises():
    m = {"audio": {"motifs": [{"source": "knock_wood",
                               "at": {"scene": 1, "word": "x"}}]}}
    with pytest.raises(AudioSpecError) as e:
        parse_audio_spec(m)
    assert "conflicting keys" in str(e.value)


def test_oneshot_missing_anchor_raises():
    m = {"clips": [{"name": "s1", "sfx": [{"source": "knock_wood"}]}]}
    with pytest.raises(AudioSpecError) as e:
        parse_audio_spec(m)
    assert "needs an 'at' anchor" in str(e.value)


def test_pan_out_of_range_raises():
    m = {"audio": {"motifs": [{"source": "knock_wood", "at": {"time": 1},
                               "pan": 5}]}}
    with pytest.raises(AudioSpecError) as e:
        parse_audio_spec(m)
    assert "pan" in str(e.value) and "maximum" in str(e.value)


def test_non_numeric_gain_raises():
    m = {"audio": {"ambient_bed": {"source": "wind_hall", "gain_db": "loud"}}}
    with pytest.raises(AudioSpecError) as e:
        parse_audio_spec(m)
    assert "gain_db" in str(e.value)


def test_multiple_errors_collected_at_once():
    # Two independent problems should both appear in the single raised message.
    m = {"audio": {"motifs": [{"source": "ghost", "at": {"time": 1}, "pan": 9}]}}
    with pytest.raises(AudioSpecError) as e:
        parse_audio_spec(m)
    msg = str(e.value)
    assert "not a known tag" in msg and "pan" in msg


def test_resolve_source_tag_and_raw_path():
    # tag resolves to a real bundled file
    p = resolve_source("knock_wood", SFX_MAP)
    assert p.exists() and p.name == "knock_wood.wav"
    # an absolute path to that same file also resolves
    assert resolve_source(str(p), SFX_MAP) == p


def test_resolve_source_unknown_import_raises():
    with pytest.raises(AudioSpecError) as e:
        resolve_source("@import/not_imported_yet", SFX_MAP)
    assert "import" in str(e.value).lower()
