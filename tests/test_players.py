"""Player model (gameplay/players.py) — pure: colour normalisation, rename/merge,
recolour, sync, line counts and the GUI card view. No Gradio, no ffmpeg."""
from gameplay import players as P

ROWS = [
    ["no way", "SPEAKER_00", 0.0, 0.5, False],
    ["what", "SPEAKER_01", 0.5, 0.9, False],
    ["nice", "SPEAKER_00", 1.0, 1.3, False],
    ["(unassigned)", "", 1.4, 1.6, False],
]
PLAYERS = [["SPEAKER_00", "#FF0000"], ["SPEAKER_01", "#00FF00"],
           ["SPEAKER_02", "#0000FF"]]


# ---- colours ---------------------------------------------------------------

def test_to_hex_accepts_every_picker_format():
    assert P.to_hex("#ff8800") == "#FF8800"
    assert P.to_hex("ff8800") == "#FF8800"
    assert P.to_hex("#f80") == "#FF8800"
    assert P.to_hex("#ff8800cc") == "#FF8800"            # alpha dropped
    assert P.to_hex("rgb(255, 136, 0)") == "#FF8800"
    assert P.to_hex("rgba(255,136,0,0.5)") == "#FF8800"
    assert P.to_hex("rgb(300, 5, 0.4)") == "#FF0500"     # out-of-range clamps
    assert P.to_hex("rgb(-5, 5, 5)") is None             # malformed -> rejected
    assert P.to_hex("not a colour", "#123456") == "#123456"
    assert P.to_rgb("#FF8800") == (255, 136, 0)


def test_normalize_players_cleans_and_dedupes():
    out = P.normalize_players([[" Chan ", "rgb(1,2,3)"], ["Chan", "#FFFFFF"],
                               ["", "#000000"], ["Jett", "garbage"]])
    assert out[0] == ["Chan", "#010203"]                 # first wins, trimmed
    assert [p[0] for p in out] == ["Chan", "Jett"]       # dup + blank dropped
    assert out[1][1].startswith("#") and len(out[1][1]) == 7   # palette fallback


# ---- rename / merge --------------------------------------------------------

def test_rename_rewrites_every_row_and_keeps_colour():
    rows, players, msg = P.rename_player(ROWS, PLAYERS, "SPEAKER_00", "Chan")
    assert [r[1] for r in rows] == ["Chan", "SPEAKER_01", "Chan", ""]
    assert players[0] == ["Chan", "#FF0000"]             # colour travels with the name
    assert "Renamed" in msg and "2 lines" in msg
    assert ROWS[0][1] == "SPEAKER_00"                    # input not mutated


def test_rename_onto_existing_player_merges():
    # one person split into two speakers: name both "Chan" -> one player
    rows, players, _ = P.rename_player(ROWS, PLAYERS, "SPEAKER_00", "Chan")
    rows, players, msg = P.rename_player(rows, players, "SPEAKER_01", "Chan")
    assert [r[1] for r in rows] == ["Chan", "Chan", "Chan", ""]
    assert [p[0] for p in players] == ["Chan", "SPEAKER_02"]
    assert players[0][1] == "#FF0000"                    # the surviving colour
    assert "Merged" in msg


def test_rename_noops():
    for old, new in [("SPEAKER_00", ""), ("SPEAKER_00", "SPEAKER_00"),
                     ("GHOST", "Chan"), ("", "Chan")]:
        rows, players, msg = P.rename_player(ROWS, PLAYERS, old, new)
        assert msg == "" and [r[1] for r in rows] == [r[1] for r in ROWS]
        assert [p[0] for p in players] == [p[0] for p in PLAYERS]


# ---- recolour / sync / counts / cards --------------------------------------

def test_recolour_normalises_and_ignores_junk():
    out = P.recolour_player(PLAYERS, "SPEAKER_01", "rgba(10, 20, 30, 1)")
    assert out[1] == ["SPEAKER_01", "#0A141E"]
    assert P.recolour_player(PLAYERS, "SPEAKER_01", "junk")[1][1] == "#00FF00"


def test_sync_adds_new_row_speakers_keeps_ghosts():
    rows = ROWS + [["gg", "Jett", 2.0, 2.2, False]]
    out = P.sync_players(PLAYERS, rows)
    assert [p[0] for p in out] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "Jett"]
    assert out[2][0] == "SPEAKER_02"                     # a player with no lines stays


def test_line_counts_ignores_unassigned():
    assert P.line_counts(ROWS) == {"SPEAKER_00": 2, "SPEAKER_01": 1}


def test_colour_map_for_renderer():
    assert P.colour_map(PLAYERS)["SPEAKER_01"] == (0, 255, 0)


def test_card_view_fixed_slots():
    view = P.card_view(PLAYERS, ROWS, 5)
    assert len(view) == 5
    assert [v["visible"] for v in view] == [True, True, True, False, False]
    assert view[0] == {"visible": True, "name": "SPEAKER_00", "colour": "#FF0000",
                       "lines": 2}
    assert view[2]["lines"] == 0                         # ghost player
