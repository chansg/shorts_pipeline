"""Gameplay tab UX handlers (gameplay/gui.py): the player-captions switch, caption
presets, player cards (rename / merge / recolour / stale-card safety), clip
resolution for captions-off builds, and the build gate. Handlers are called
directly — no Gradio server, no ffmpeg."""
from pathlib import Path

import gradio as gr
import pytest

from gameplay import config as gconf
from gameplay import gui as g

ROWS = [["no way", "SPEAKER_00", 0.0, 0.5, False],
        ["what", "SPEAKER_01", 0.5, 0.9, False],
        ["nice", "SPEAKER_00", 1.0, 1.3, False]]
PLAYERS = [["SPEAKER_00", "#FF0000"], ["SPEAKER_01", "#00FF00"]]
N = gconf.MAX_PLAYER_CARDS


def _u(update, key):
    return update.get(key) if isinstance(update, dict) else getattr(update, key, None)


def _card(cards, i):
    """(column, head, key, name, colour) updates for slot i."""
    return cards[i * 5:(i + 1) * 5]


def _shown(update):
    """Is this section shown? Sections are hidden by the gp-gone class, not `visible`."""
    assert _u(update, "visible") is None, "use classes, not visible (see _GP_STYLE)"
    return g._GONE not in (_u(update, "elem_classes") or [])


# ---- captions on/off -------------------------------------------------------

def test_toggle_captions_shows_and_hides_the_right_sections():
    on = [_shown(x) for x in g._toggle_captions("on")]
    off = [_shown(x) for x in g._toggle_captions("off")]
    # [ASR controls, off-note, step ②, caption style, style-off note]
    assert on == [True, False, True, True, False]
    assert off == [False, True, False, False, True]


def _gameplay_blocks():
    with gr.Blocks() as demo:
        with gr.Tabs():
            g.build_gameplay_tab()
    return demo


def test_nothing_in_the_tab_is_hidden_with_visible():
    # Regressions (review): visible=False unmounted the transcript editor (turning
    # captions back on restored stale rows over the user's edits); visible="hidden"
    # doesn't hide a Column at all; and opening the Gaming page pushes visible=True
    # to every child, re-showing anything hidden that way. Everything conditional is
    # hidden with the gp-gone class instead, so nothing may be built with visible=False.
    demo = _gameplay_blocks()
    hidden = [b for b in demo.blocks.values() if getattr(b, "visible", True) is not True]
    assert hidden == []
    assert any(g._GONE in (getattr(b, "elem_classes", None) or [])
               for b in demo.blocks.values())          # unused cards start hidden


def test_colour_picker_only_when_not_colouring_by_player():
    assert not _shown(g._colour_picker_classes(True))
    assert _shown(g._colour_picker_classes(False))


# ---- presets ---------------------------------------------------------------

def test_apply_preset_sets_every_look_control():
    p = gconf.CAPTION_PRESETS["Minimal"]
    assert g._apply_preset("Minimal") == tuple(p[k] for k in g._PRESET_KEYS)


def test_custom_preset_leaves_controls_alone():
    out = g._apply_preset(g._CUSTOM)
    assert len(out) == len(g._PRESET_KEYS)
    assert all(not isinstance(x, (str, int, float)) for x in out)   # gr.skip()s


def _look(name, **over):
    p = dict(gconf.CAPTION_PRESETS[name])
    p.update(over)
    return [p[k] for k in g._PRESET_KEYS]


def test_preset_label_stays_while_controls_match():
    # applying a preset sets controls that match it -> label must NOT flip
    assert not isinstance(g._preset_label("Clean", *_look("Clean")), str)
    # picker/slider formats still count as a match
    assert not isinstance(g._preset_label("Minimal", *_look(
        "Minimal", color="rgb(255,255,255)", size=float(
            gconf.CAPTION_PRESETS["Minimal"]["size"]))), str)


def test_preset_label_flips_to_custom_on_any_change():
    assert g._preset_label("Clean", *_look("Clean", uppercase=True)) == g._CUSTOM
    assert g._preset_label("Clean", *_look("Clean", size=200)) == g._CUSTOM
    # single-colour preset: changing the text colour is a real change
    assert g._preset_label("Minimal", *_look("Minimal", color="#FF0000")) == g._CUSTOM


def test_preset_label_ignores_hidden_colour_when_colouring_by_player():
    # "Bold pop" colours by player, so the (hidden) text colour is irrelevant
    assert not isinstance(g._preset_label("Bold pop", *_look("Bold pop",
                                                             color="#123456")), str)


def test_preset_label_custom_is_left_alone():
    assert not isinstance(g._preset_label(g._CUSTOM, *_look("Clean")), str)


# ---- player cards ----------------------------------------------------------

def test_card_updates_shape_and_content():
    cards = g._card_updates(PLAYERS, ROWS)
    assert len(cards) == 5 * N
    col, head, key, name, colour = _card(cards, 0)
    assert _shown(col) and "gp-player" in _u(col, "elem_classes")
    assert _u(key, "value") == _u(name, "value") == "SPEAKER_00"
    assert _u(colour, "value") == "#FF0000"
    assert "2 lines" in _u(head, "value")
    assert not _shown(_card(cards, 2)[0])                      # unused slot hidden


def _is_skip(x):
    """gr.skip() — an update that changes nothing (not a value)."""
    return isinstance(x, dict) and set(x) <= {"__type__"}


def test_rename_card_updates_state_but_not_the_editor_or_other_cards():
    # the browser relabels the editor in place (window.__txeRename), so a plain rename
    # must NOT re-render it (that stole focus from the row the user clicked into) and
    # must leave every other card alone (it clobbered names being typed there)
    players, rows, html, msg, *cards = g._rename_card(0, "SPEAKER_00", "Chan",
                                                      PLAYERS, ROWS)
    assert [r[1] for r in rows] == ["Chan", "SPEAKER_01", "Chan"]
    assert players[0] == ["Chan", "#FF0000"]
    assert _is_skip(html) and "Renamed" in msg
    col, head, key, name, colour = _card(cards, 0)
    assert _u(key, "value") == _u(name, "value") == "Chan"     # this card re-keyed
    assert _is_skip(col) and _is_skip(head) and _is_skip(colour)
    assert all(_is_skip(x) for x in cards[5:])                 # other cards untouched
    assert len(cards) == 5 * N


def test_rename_card_merge_collapses_a_split_speaker():
    players, rows, html, msg, *cards = g._rename_card(1, "SPEAKER_01", "SPEAKER_00",
                                                      PLAYERS, ROWS)
    assert {r[1] for r in rows} == {"SPEAKER_00"}
    assert [p[0] for p in players] == ["SPEAKER_00"]
    assert "Merged" in msg and isinstance(html, str)           # full refresh on merge
    assert "data-rev=" in html                                 # ...an authoritative one
    assert not _shown(_card(cards, 1)[0])                      # merged card gone


def test_stale_card_key_is_a_noop():
    # a card that still shows a player who was already merged away must not rename
    # whoever now sits in that slot — and must not touch anything
    out = g._rename_card(0, "GONE", "Chan", PLAYERS, ROWS)
    assert all(_is_skip(x) for x in out)


def test_unchanged_name_blur_touches_nothing():
    # just clicking in and out of a name field used to re-render the editor + cards
    out = g._rename_card(0, "SPEAKER_00", "SPEAKER_00", PLAYERS, ROWS)
    assert all(_is_skip(x) for x in out)


def test_blank_name_restores_only_its_own_card():
    out = g._rename_card(1, "SPEAKER_01", "   ", PLAYERS, ROWS)
    players, rows, html, msg, *cards = out
    assert all(_is_skip(x) for x in (players, rows, html, msg))
    assert _u(_card(cards, 1)[3], "value") == "SPEAKER_01"     # name put back
    assert all(_is_skip(x) for i, x in enumerate(cards) if i != 1 * 5 + 3)


def test_plain_rename_keeps_the_overflow_warning():
    # Regression (review): with more players than cards, renaming one replaced the
    # 'more players than cards' warning — though the extra players still had no card.
    many = [[f"SPEAKER_{i:02d}", "#FFFFFF"] for i in range(N + 2)]
    rows = [["x", p[0], float(i), i + 0.5, False] for i, p in enumerate(many)]
    _players, _rows, _html, msg, *_ = g._rename_card(0, "SPEAKER_00", "Chan", many, rows)
    assert "Renamed" in msg and "2 more players" in msg


def test_rename_handler_binds_the_slot():
    out = g._rename_handler(1)("SPEAKER_01", "Jett", PLAYERS, ROWS)
    cards = out[4:]
    assert _u(_card(cards, 1)[3], "value") == "Jett" and _is_skip(_card(cards, 0)[3])


def test_overflow_note_only_when_players_exceed_cards():
    few = [[f"P{i}", "#FFFFFF"] for i in range(N)]
    many = [[f"P{i}", "#FFFFFF"] for i in range(N + 2)]
    assert g._overflow_note(few) == ""
    note = g._overflow_note(many)
    assert "2 more players" in note and "merge" in note


def test_recolour_card_normalises_rgba():
    players, html = g._recolour_card("SPEAKER_01", "rgba(0, 0, 255, 1)", PLAYERS, ROWS)
    assert players[1] == ["SPEAKER_01", "#0000FF"]
    assert "#0000FF" in html or "#0000ff" in html


def test_recolour_card_unchanged_colour_is_a_noop():
    # the picker's .change also fires when a card refresh re-sets the same colour —
    # that must not write state or re-render the editor (it would steal focus)
    out = g._recolour_card("SPEAKER_01", "#00ff00", PLAYERS, ROWS)
    assert all(not isinstance(x, (list, str)) for x in out)          # gr.skip()s
    out = g._recolour_card("", "#888888", PLAYERS, ROWS)              # hidden slot
    assert all(not isinstance(x, (list, str)) for x in out)


def test_sync_cards_adds_a_newly_typed_player():
    rows = ROWS + [["gg", "Jett", 1.4, 1.6, False]]
    players, _html, status, *cards = g._sync_cards(PLAYERS, rows)
    assert players[-1][0] == "Jett" and status == ""
    assert _u(_card(cards, 2)[3], "value") == "Jett"


def test_sync_cards_reports_overflow():
    rows = [["x", f"P{i}", float(i), i + 0.5, False] for i in range(N + 1)]
    _players, _html, status, *_ = g._sync_cards([], rows)
    assert "1 more player" in status


def test_card_visibility_reasserts_each_slot():
    vis = [_shown(x) for x in g._card_visibility(PLAYERS, ROWS)]
    assert vis == [True, True] + [False] * (N - 2)


def test_bulk_edits_render_the_editor_authoritatively():
    # Regression (review): a bulk edit whose result happened to equal the rows the
    # editor last built from (e.g. re-assigning rows a speaker button had changed) was
    # ignored by the editor, and the next commit wrote the old rows back. Row-changing
    # renders carry a fresh data-rev so the editor always rebuilds from them.
    assert "data-rev=" in g._edit_assign(ROWS, "1", "SPEAKER_01", PLAYERS)[1]
    assert "data-rev=" in g._edit_find_replace(ROWS, "nice", "nice", False, PLAYERS)[1]
    assert "data-rev=" in g._edit_merge(ROWS, "1-2", PLAYERS)[1]
    assert "data-rev=" in g._edit_split(ROWS, 1, PLAYERS)[1]
    # colour/players-only refreshes keep the live rows (no rev)
    assert "data-rev=" not in g._recolour_card("SPEAKER_01", "#0000ff", PLAYERS, ROWS)[1]
    assert "data-rev=" not in g._sync_cards(PLAYERS, ROWS)[1]


def _deps(demo, name):
    return [d for d in demo.fns.values() if getattr(d.fn, "__name__", "") == name]


def test_a_failed_build_unlocks_the_button():
    # Regression (review): the unlock was a .then after the build, but Gradio 6's
    # frontend runs no .then step after an error — a refused build (not transcribed
    # yet, a different clip loaded, …) left Build stuck on 'Building…' until a reload,
    # which loses the transcript edits. The unlock must hang off every failure path.
    demo = _gameplay_blocks()
    fns = demo.fns                                          # dependency id -> function
    run = _deps(demo, "_run_build")[0]
    flush = fns[run.trigger_after]
    idle = _deps(demo, "_build_idle")
    after = {(fns[d.trigger_after].fn.__name__, d.trigger_only_on_failure) for d in idle}
    assert (flush.fn.__name__, True) in after              # the flush failing
    assert ("_run_build", True) in after                    # the build failing
    assert ("_run_build", False) in after                   # the build finishing
    assert run.trigger_only_on_success                      # no build after a failed flush


def test_bulk_edits_queue_behind_row_commits():
    # the click that triggers a bulk edit blurs the row being typed in; sharing the
    # commit's concurrency group makes that commit land first
    demo = _gameplay_blocks()
    for name in ("_edit_assign", "_edit_find_replace", "_edit_merge", "_edit_split",
                 "_rows_from_dom"):
        assert all(d.concurrency_id == "gp-players" for d in _deps(demo, name)), name


def test_card_heads_only_touch_the_counts():
    heads = g._card_heads(PLAYERS, ROWS)
    assert len(heads) == N and "2 lines" in _u(heads[0], "value")


def test_parse_speaker_rows_tolerates_picker_formats():
    assert g._parse_speaker_rows([["Chan", "rgb(255,0,0)"]]) == {"Chan": (255, 0, 0)}


# ---- clip resolution -------------------------------------------------------

def _fake_importer(tmp_path, monkeypatch):
    monkeypatch.setattr(gconf, "GAMEPLAY_DIR", tmp_path)
    imported = []

    def fake_import(src, name=None):
        imported.append((src, name))
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "source.mp4").write_bytes(b"")

    monkeypatch.setattr(g.transcribe_mod, "import_source", fake_import)
    return imported


def test_prepare_build_captions_off_imports_with_visible_progress(tmp_path,
                                                                  monkeypatch):
    imported = _fake_importer(tmp_path, monkeypatch)
    frames = list(g._prepare_build(str(tmp_path / "My Clip.mp4"), "", "off"))
    name = g.slugify("My Clip")
    assert "Importing" in frames[0][1] and frames[0][0] is None     # visible feedback
    assert frames[-1][0] == name and imported[0][1] == name
    imported.clear()
    again = list(g._prepare_build(str(tmp_path / "My Clip.mp4"), "", "off"))
    assert imported == [] and len(again) == 1                        # already imported


def test_resolve_build_clip_captions_on_needs_a_transcribed_clip():
    with pytest.raises(gr.Error, match="Transcribe"):
        g._resolve_build_clip("C:/x/clip.mp4", "", "on")


def test_resolve_build_clip_captions_on_refuses_a_stale_clip():
    with pytest.raises(gr.Error, match="different clip"):
        g._resolve_build_clip("C:/x/new_clip.mp4", "old_clip", "on")
    assert g._resolve_build_clip("C:/x/old_clip.mp4", "old_clip", "on") == \
        ("old_clip", False)


def test_captions_off_build_of_another_clip_cant_block_the_transcribed_one(tmp_path,
                                                                         monkeypatch):
    # Regression (review): transcribe A, captions-off build of B, then captions-on
    # build of A again was refused ("loaded a different clip") because the stale check
    # read the LAST BUILT clip. It must compare against the clip the rows belong to.
    _fake_importer(tmp_path, monkeypatch)
    list(g._prepare_build(str(tmp_path / "b.mp4"), "a", "off"))      # build B, no caps
    assert g._resolve_build_clip(str(tmp_path / "a.mp4"), "a", "on") == ("a", False)


def test_preview_uses_the_transcripts_own_clip():
    # the preview takes tx_clip (whose rows these are), so it can't put clip A's
    # captions on clip B's video; with no transcript it refuses
    with pytest.raises(gr.Error, match="Transcribe a clip first"):
        g._do_preview_captions("", ROWS, PLAYERS, "on", "Anton", 150, 7.0, True,
                               True, "phrase", True, "#FFFFFF", 0.72, 0.0)


# ---- build gate ------------------------------------------------------------

def _build_args(clip_name, tx_clip, rows, captions):
    look = gconf.CAPTION_PRESETS[gconf.CAPTION_DEFAULT_PRESET]
    return [clip_name, tx_clip, rows, PLAYERS, captions, [], g._NONE,
            gconf.OVERLAY_DEFAULT_POSITION, 0, 0, gconf.REFRAME_MODE, 0.5, 0.5, 1.0,
            "both", "bleep", False, "", "",
            look["font"], look["size"], look["outline"], look["uppercase"],
            look["animate"], look["mode"], look["by_player"], look["color"],
            gconf.CAPTION_POS_Y_FRAC, 0.0]


def _setup_clip(tmp_path, monkeypatch, name="clip"):
    monkeypatch.setattr(gconf, "GAMEPLAY_DIR", tmp_path)
    (tmp_path / name).mkdir(parents=True, exist_ok=True)
    (tmp_path / name / "source.mp4").write_bytes(b"")
    seen = {}

    def fake_run(clip, transcript, opts, force=False):
        seen.update(transcript=transcript, opts=opts)
        yield {"msg": "built", "done": True}

    monkeypatch.setattr(g, "run_manual", fake_run)
    return seen


def test_build_captions_off_without_a_transcript_warns_the_bleep_is_skipped(
        tmp_path, monkeypatch):
    seen = _setup_clip(tmp_path, monkeypatch)
    log = list(g._do_build(*_build_args("clip", "", [], "off")))   # censor = "both"
    assert "straight from the clip" in log[0]
    assert "NOT censored" in log[0]                  # the bleep can't run: say so
    assert seen["opts"].captions_enabled is False and not seen["transcript"].words


def test_build_intro_is_truthful_about_captions_and_censor():
    assert "edited transcript (3 rows)" in g._build_intro(True, True, "both", 3)[0]
    assert "bleep profanity" in g._build_intro(False, True, "audio", 3)[0]
    assert "censor off" in g._build_intro(False, True, "off", 3)[0]
    assert "nothing to mask" in g._build_intro(False, True, "caption", 3)[0]
    assert "NOT censored" in " ".join(g._build_intro(False, False, "both", 0))
    plain = g._build_intro(False, False, "off", 0)
    assert len(plain) == 1 and "no transcript" in plain[0]


def test_build_captions_on_with_empty_transcript_is_refused(tmp_path, monkeypatch):
    _setup_clip(tmp_path, monkeypatch)
    with pytest.raises(gr.Error, match="transcript is empty"):
        list(g._do_build(*_build_args("clip", "clip", [], "on")))


def test_build_never_applies_another_clips_rows(tmp_path, monkeypatch):
    seen = _setup_clip(tmp_path, monkeypatch)
    # rows belong to "other" -> a captions-off build of "clip" must ignore them
    list(g._do_build(*_build_args("clip", "other", ROWS, "off")))
    assert not seen["transcript"].words


def test_run_build_is_one_event_from_clip_to_result(tmp_path, monkeypatch):
    # Regression (review): Build was a chain of events sharing a build-clip state, so
    # a second click mid-build swapped the clip and the first build's result step read
    # the wrong one (while the second build was silently dropped). The whole build is
    # now one generator that clears the old result first and shows ITS OWN output.
    _fake_importer(tmp_path, monkeypatch)
    monkeypatch.setattr(g, "run_manual", lambda clip, t, o, force=False: iter(
        [{"msg": "built", "done": True}]))
    out = g.GameplayClip("My Clip").final_path
    args = _build_args("unused", "", [], "off")
    frames = list(g._run_build(str(tmp_path / "My Clip.mp4"), *args[1:]))
    assert frames[0] == ("Importing the clip (normalising it if needed)…", None)
    assert frames[-1][1] is None                                # nothing built yet
    out.write_bytes(b"")
    frames = list(g._run_build(str(tmp_path / "My Clip.mp4"), *args[1:]))
    assert frames[0][1] is None and "built" in frames[-1][0]    # old result cleared
    assert frames[-1][1] == str(out)                            # this build's output


def test_build_button_locks_while_building():
    busy, idle = g._build_busy(), g._build_idle()
    assert _u(busy, "interactive") is False and _u(idle, "interactive") is True


def test_build_passes_the_caption_look_through(tmp_path, monkeypatch):
    seen = _setup_clip(tmp_path, monkeypatch)
    args = _build_args("clip", "clip", ROWS, "on")
    args[20], args[21], args[22], args[23] = 120, 2.5, False, False   # size..animate
    args[25], args[26] = False, "rgb(0,255,0)"                        # by_player, colour
    list(g._do_build(*args))
    o = seen["opts"]
    assert (o.caption_size, o.caption_outline, o.caption_uppercase,
            o.caption_animate, o.caption_by_player, o.caption_color) == \
        (120, 2.5, False, False, False, "#00FF00")
    assert o.speaker_colors["SPEAKER_00"] == (255, 0, 0)
