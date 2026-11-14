"""Player model for the gameplay transcript — PURE (no Gradio, no ffmpeg).

In a gameplay clip the diarised speakers ARE the players. Transcript rows carry a
speaker label per word (`[text, speaker, start, end, censor]`); the player list is the
ordered `[[name, "#RRGGBB"], ...]` that drives the editor's speaker buttons, the
per-player caption colours, and the player cards in the GUI.

* Renaming a player rewrites every row that carries the old label, so a name typed
  once shows up everywhere (editor buttons, rows, caption colours).
* Renaming a player to a name another player already has MERGES them — the fix for the
  classic diarisation error of one person being split into two speakers.
* Colours are normalised to `#RRGGBB` on the way in, because Gradio's ColorPicker can
  hand back `rgb(...)`/`rgba(...)` as well as hex.
"""
from __future__ import annotations

import re
from collections import Counter

from modules.karaoke_captions import DEFAULT_SPEAKER_PALETTE

_HEX6 = re.compile(r"^#?([0-9a-fA-F]{6})(?:[0-9a-fA-F]{2})?$")
_HEX3 = re.compile(r"^#?([0-9a-fA-F]{3})$")
_RGB = re.compile(r"^rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)")


# ---- colours ---------------------------------------------------------------

def to_hex(value, default: str | None = None) -> str | None:
    """Normalise a colour to `#RRGGBB`. Accepts `#rgb`, `#rrggbb`, `#rrggbbaa`,
    `rgb(r,g,b)` and `rgba(r,g,b,a)`. Returns `default` when unparseable."""
    s = str(value or "").strip()
    m = _HEX6.match(s)
    if m:
        return "#" + m.group(1).upper()
    m = _HEX3.match(s)
    if m:
        return "#" + "".join(c * 2 for c in m.group(1)).upper()
    m = _RGB.match(s.lower())
    if m:
        r, g, b = (max(0, min(255, int(round(float(x))))) for x in m.groups())
        return f"#{r:02X}{g:02X}{b:02X}"
    return default


def to_rgb(value) -> tuple[int, int, int] | None:
    h = to_hex(value)
    return None if h is None else tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def palette_hex(i: int) -> str:
    r, g, b = DEFAULT_SPEAKER_PALETTE[i % len(DEFAULT_SPEAKER_PALETTE)]
    return f"#{r:02X}{g:02X}{b:02X}"


# ---- the player list -------------------------------------------------------

def _row_speaker(row) -> str:
    return str((list(row) + ["", ""])[1] or "").strip()


def normalize_players(players) -> list[list[str]]:
    """Clean `[[name, colour], ...]`: strip names, drop blanks and duplicates (first
    wins), normalise colours (unparseable -> palette colour by position)."""
    out: list[list[str]] = []
    seen: set[str] = set()
    for p in players or []:
        p = list(p) + ["", ""]
        name = str(p[0] or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append([name, to_hex(p[1]) or palette_hex(len(out))])
    return out


def sync_players(players, rows) -> list[list[str]]:
    """The player list plus any speaker that appears in the rows but isn't listed yet
    (e.g. a new name typed into a bulk 'assign speaker'). Order is kept; newcomers get
    the next palette colour. Players with no lines are kept, so a player the diariser
    missed can still be named first and have lines assigned to them afterwards."""
    out = normalize_players(players)
    names = {p[0] for p in out}
    for r in rows or []:
        s = _row_speaker(r)
        if s and s not in names:
            names.add(s)
            out.append([s, palette_hex(len(out))])
    return out


def line_counts(rows) -> Counter:
    """{player: number of transcript rows} (unassigned rows are not counted)."""
    return Counter(s for s in (_row_speaker(r) for r in rows or []) if s)


def rename_player(rows, players, old: str, new: str):
    """Rename player `old` -> `new` everywhere. Returns `(rows, players, message)`.

    If `new` is already another player's name the two are MERGED (their rows become
    one player, keeping the existing player's colour). Blank or unchanged names, or an
    `old` that is no longer a player (a stale UI card), are a no-op with message ""."""
    old = str(old or "").strip()
    new = str(new or "").strip()
    players = normalize_players(players)
    rows = [list(r) for r in rows or []]
    names = [p[0] for p in players]
    if not old or not new or new == old or old not in names:
        return rows, players, ""
    moved = 0
    for r in rows:
        if len(r) > 1 and str(r[1] or "").strip() == old:
            r[1] = new
            moved += 1
    lines = f"{moved} line{'s' if moved != 1 else ''}"
    if new in names:
        players = [p for p in players if p[0] != old]
        msg = f"Merged **{old}** into **{new}** ({lines} moved)."
    else:
        players = [[new, p[1]] if p[0] == old else p for p in players]
        msg = f"Renamed **{old}** → **{new}** ({lines})."
    return rows, players, msg


def recolour_player(players, name: str, colour) -> list[list[str]]:
    """Set a player's colour (ignored if the colour can't be parsed)."""
    hexv = to_hex(colour)
    players = normalize_players(players)
    if hexv is None:
        return players
    return [[p[0], hexv] if p[0] == name else p for p in players]


def colour_map(players) -> dict[str, tuple[int, int, int]]:
    """`{name: (r, g, b)}` for the caption renderer."""
    return {p[0]: to_rgb(p[1]) for p in normalize_players(players)}


def card_view(players, rows, n_slots: int) -> list[dict]:
    """What each of the GUI's fixed player-card slots shows (length == n_slots):
    `{visible, name, colour, lines}`. Players beyond `n_slots` still caption normally
    (auto colour); they just don't get a card."""
    players = normalize_players(players)
    counts = line_counts(rows)
    out = []
    for i in range(n_slots):
        if i < len(players):
            name, hexv = players[i]
            out.append({"visible": True, "name": name, "colour": hexv,
                        "lines": counts.get(name, 0)})
        else:
            out.append({"visible": False, "name": "", "colour": "#888888", "lines": 0})
    return out
