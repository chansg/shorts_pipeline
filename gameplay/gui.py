"""The Gameplay tab for the unified GUI — a self-contained module so app.py only
needs an import + one call inside its `gr.Tabs()` block (the lore wizard is
untouched).

Manual mode only, laid out as four steps:

  ① Clip & captions   upload a pre-trimmed clip; choose whether to caption the
                      players. "Off" skips transcription entirely.
  ② Transcript & players  (captions on) fix the words in the fast keyboard editor and
                      name each player once on their card — the name and colour flow to
                      every row, the editor's speaker buttons and the captions.
  ③ Style             sub-tabs: Captions (presets + size/outline/case/animation/colour
                      + preview) · Layout · Overlay · Audio & extras.
  ④ Build             streams the build log; shows the 9:16 result.

The experimental full-auto long-video processor lives in its own landing entry
(fullauto/) and exports a 16:9 YouTube video — it is NOT part of this page.

Streaming handlers update only their status Textbox while running (yielding
gr.update() for other outputs), then a chained step populates the editor / preview
from disk — mirroring how app.py's run_assemble streams.
"""
from __future__ import annotations

import time
from pathlib import Path

import gradio as gr

from orchestrator.errors import FriendlyError, friendly
from gameplay import config as gconf
from gameplay import editing as edit_mod
from gameplay import editor as editor_mod
from gameplay import manual as manual_mod
from gameplay import overlay as ov_mod
from gameplay import players as players_mod
from gameplay import transcribe as transcribe_mod
from gameplay.manual import ManualOptions, run_manual
from gameplay.state import GameplayClip, slugify
from gameplay.transcript import Transcript

_NONE = "(none)"
_CUSTOM = "Custom"
# Order of the caption-look controls a preset sets (and _apply_preset returns).
_PRESET_KEYS = ("font", "size", "outline", "uppercase", "animate", "mode",
                "by_player", "color")
_CARD_FIELDS = 5          # per player card: column, head, hidden key, name, colour

# Scoped styling for this tab (gr.HTML injects it; class names are gp-prefixed).
_GP_STYLE = """<style>
.gp-step { margin-top: 12px !important; }
.gp-step h4 { margin: 0 !important; font-size: 1.05em; letter-spacing: .01em;
  padding-bottom: 4px; border-bottom: 1px solid var(--border-color-primary); }
.gp-hint, .gp-hint p { font-size: .86em; opacity: .8; }
.gp-hidden { display: none !important; }
.gp-players { gap: 10px !important; }
.gp-player { border: 1px solid var(--border-color-primary); border-radius: 10px;
  padding: 8px 10px !important; background: var(--block-background-fill); }
.gp-player p { margin: 0 0 4px !important; font-size: .86em; }
/* Conditional sections are hidden with this class, NOT Gradio's `visible`:
   visible=False unmounts a section (the editor then comes back from a stale server
   value), visible="hidden" doesn't hide a Column at all, and showing the Gaming page
   pushes visible=True down to every child. A class sidesteps all three — every
   section stays mounted and nothing outside this tab can un-hide it. */
.gp-gone { display: none !important; }
</style>"""


# ---- colour helpers --------------------------------------------------------

def _rgb_to_hex(rgb) -> str:
    r, g, b = rgb
    return f"#{r:02X}{g:02X}{b:02X}"


def _hex_to_rgb(h):
    h = str(h or "").strip().lstrip("#")
    if len(h) != 6:
        return None
    try:
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None


def _speaker_rows(t: Transcript):
    """Seed the player list (and the editor's speaker buttons) with palette hex
    colours. Detected speakers come first (each the colour the renderer auto-assigns),
    then we PAD with default `SPEAKER_NN` players up to DEFAULT_SPEAKER_ROWS so there
    are always at least that many to assign to — handy when the diariser found fewer
    than were really talking. A player with no lines is harmless (a colour only
    applies to a speaker that appears in a cue)."""
    rows, seen = [], set()
    for s in t.speakers:
        rgb = gconf.SPEAKER_PALETTE[len(rows) % len(gconf.SPEAKER_PALETTE)]
        rows.append([s, _rgb_to_hex(rgb)])
        seen.add(s)
    i = 0
    while len(rows) < gconf.DEFAULT_SPEAKER_ROWS:
        name = f"SPEAKER_{i:02d}"
        if name not in seen:
            rgb = gconf.SPEAKER_PALETTE[len(rows) % len(gconf.SPEAKER_PALETTE)]
            rows.append([name, _rgb_to_hex(rgb)])
            seen.add(name)
        i += 1
    return rows


def _parse_speaker_rows(rows) -> dict:
    """{player: (r, g, b)} for the renderer. Tolerates any colour format the picker
    returns (hex / rgb() / rgba())."""
    return players_mod.colour_map(rows)


# ---- player cards ----------------------------------------------------------

def _card_head(i: int, lines: int) -> str:
    count = f"{lines} line{'s' if lines != 1 else ''}" if lines else "no lines yet"
    return f"**Player {i + 1}** · {count}"


_GONE = "gp-gone"
_CARD_CLASSES = ["gp-player"]


def _shown(flag: bool, base=()) -> list[str]:
    """elem_classes that show/hide a section with the gp-gone class (see _GP_STYLE for
    why this isn't Gradio's `visible`)."""
    return list(base) + ([] if flag else [_GONE])


def _card_classes(shown: bool) -> list[str]:
    """A card slot is always mounted; an unused one is hidden."""
    return _shown(shown, _CARD_CLASSES)


def _card_updates(players, rows) -> list:
    """Flat updates for every card slot: [column, head, key, name, colour] x N. The
    hidden key records which player the card shows, so a rename/recolour always
    targets the player the user SAW even if slots shifted (e.g. after a merge)."""
    out = []
    for i, v in enumerate(players_mod.card_view(players, rows, gconf.MAX_PLAYER_CARDS)):
        out += [gr.update(elem_classes=_card_classes(v["visible"])),
                gr.update(value=_card_head(i, v["lines"]) if v["visible"] else ""),
                gr.update(value=v["name"]),
                gr.update(value=v["name"]),
                gr.update(value=v["colour"])]
    return out


def _card_visibility(players, rows) -> list:
    """Re-send just which card slots are shown (cheap; chained after every step that
    can add or remove a player)."""
    return [gr.update(elem_classes=_card_classes(v["visible"]))
            for v in players_mod.card_view(players, rows, gconf.MAX_PLAYER_CARDS)]


def _card_heads(players, rows) -> list:
    """Just the line counts — cheap refresh after each editor commit. Deliberately
    leaves the name/colour fields alone so it can never clobber a name being typed."""
    return [gr.update(value=_card_head(i, v["lines"]) if v["visible"] else "")
            for i, v in enumerate(players_mod.card_view(players, rows,
                                                        gconf.MAX_PLAYER_CARDS))]


def _overflow_note(players) -> str:
    """Say so when there are more players than cards (they still caption, with an auto
    colour, but have no card to rename/recolour/merge them from)."""
    extra = len(players_mod.normalize_players(players)) - gconf.MAX_PLAYER_CARDS
    if extra <= 0:
        return ""
    return (f"⚠ {extra} more player{'s' if extra != 1 else ''} than there are cards — "
            f"merge some (give a card another player's name), or use Bulk edits → "
            f"Assign player to rows.")


def _rename_card(slot, key, new_name, players, rows):
    """Commit a player-card rename for card `slot` (fires on blur; Enter blurs the
    field). Outputs: players, rows, editor_html, player_status, then every card.

    * no-op (unchanged / stale card / blank name): touch nothing — a blank name just
      puts the old name back in this card;
    * plain rename: the browser has already relabelled the editor in place
      (window.__txeRename), so the editor is NOT re-rendered (that would steal focus
      from a row the user just clicked into) and only this card's key/name change;
    * merge (renamed onto an existing player): slots shift, so the editor and every
      card are refreshed."""
    n_out = 4 + _CARD_FIELDS * gconf.MAX_PLAYER_CARDS
    base = 4 + slot * _CARD_FIELDS
    new_name = str(new_name or "").strip()
    rows2, players2, msg = players_mod.rename_player(rows, players, key, new_name)
    if not msg:
        out = [gr.skip()] * n_out
        if not new_name:
            out[base + 3] = gr.update(value=key)
        return tuple(out)
    if len(players2) == len(players_mod.normalize_players(players)):
        # keep the 'more players than cards' warning: a rename doesn't change the count
        status = f"{msg} {_overflow_note(players2)}".strip()
        out = [players2, rows2, gr.skip(), status] + [gr.skip()] * (n_out - 4)
        out[base + 2] = gr.update(value=new_name)          # hidden key
        out[base + 3] = gr.update(value=new_name)          # name field
        return tuple(out)
    note = _overflow_note(players2)
    # a merge changes rows on the server: an authoritative render (rev), so the editor
    # rebuilds from these rows even if they happen to equal the ones it last built from
    return (players2, rows2, editor_mod.render_editor(rows2, players2, rev=time.time_ns()),
            f"{msg} {note}".strip(), *_card_updates(players2, rows2))


def _rename_handler(slot):
    """Bind a card's slot index (each card's blur gets its own handler)."""
    def handler(key, new_name, players, rows):
        return _rename_card(slot, key, new_name, players, rows)
    return handler


def _recolour_card(key, colour, players, rows):
    """Commit a player's colour. Wired to the picker's .change (Gradio 6's ColorPicker
    doesn't dispatch .input from its hex field), which ALSO fires when a card refresh
    sets the colour programmatically — so an unchanged colour is a no-op: no state
    write and no editor re-render (that would steal focus from a row being edited)."""
    players2 = players_mod.recolour_player(players, key, colour)
    if players2 == players_mod.normalize_players(players):
        return gr.skip(), gr.skip()
    return players2, editor_mod.render_editor(rows, players2)


def _sync_cards(players, rows):
    """After a bulk edit: add any new player name typed into 'Assign player' and
    refresh the cards + editor buttons. Outputs: players, editor_html, player_status,
    then every card."""
    players2 = players_mod.sync_players(players, rows)
    return (players2, editor_mod.render_editor(rows, players2), _overflow_note(players2),
            *_card_updates(players2, rows))


# ---- transcript editor handlers (thin wrappers over gameplay.editing) ------

# Bulk-edit handlers operate on the authoritative rows_state and RE-RENDER the editor
# HTML (the JS observer rebuilds from the new data). They reuse the pure editing.py
# helpers verbatim. Each returns (rows, editor_html, status); spk_rows (the player
# list) colours the view. Their renders are AUTHORITATIVE (a fresh rev): without one
# the editor only rebuilds when the rows differ from those it last built from, so a
# bulk edit that happens to undo in-editor changes (e.g. re-assigning rows a speaker
# button changed) would be ignored — and the next commit would write the old rows back.
def _rows_render(rows, spk_rows) -> str:
    return editor_mod.render_editor(rows, spk_rows, rev=time.time_ns())


def _edit_assign(rows, span, speaker, spk_rows):
    rows2 = edit_mod.assign_speaker(rows, span, speaker)
    return (rows2, _rows_render(rows2, spk_rows),
            f"Assigned '{speaker}' to rows {span or '(none)'}.")


def _edit_find_replace(rows, find, repl, whole_word, spk_rows):
    rows2, n = edit_mod.find_replace(rows, find, repl, whole_word=bool(whole_word))
    return (rows2, _rows_render(rows2, spk_rows),
            f"Replaced {n} occurrence(s) of '{find}'.")


def _edit_merge(rows, span, spk_rows):
    rows2 = edit_mod.merge_rows(rows, span)
    return (rows2, _rows_render(rows2, spk_rows),
            f"Merged rows {span or '(none)'}.")


def _edit_split(rows, row_num, spk_rows):
    try:
        i = int(row_num)
    except (TypeError, ValueError):
        return rows, gr.skip(), "Enter a row number to split."
    rows2 = edit_mod.split_row(rows, i)
    return rows2, _rows_render(rows2, spk_rows), f"Split row {i}."


def _rows_from_dom(payload):
    """Commit handler: the js reader returned the live editor rows JSON -> rows_state.
    On a parse miss keep the prior state (gr.skip) instead of wiping edits; never
    re-renders the editor, so there's no echo loop."""
    rows = editor_mod.parse_bridge(payload)
    return rows if rows is not None else gr.skip()


# ---- captions on/off + caption look ----------------------------------------

def _captions_on(choice) -> bool:
    return str(choice or "on") != "off"


def _toggle_captions(choice):
    """Show/hide for: [ASR controls, captions-off note, step ②, caption style panel,
    caption-style-off note]. Classes, not `visible`, so every section — the transcript
    editor above all — stays mounted with its live rows."""
    on = _captions_on(choice)
    return tuple(gr.update(elem_classes=_shown(flag))
                 for flag in (on, not on, on, on, not on))


def _colour_picker_classes(by_player):
    """The single 'Text colour' picker only matters when lines aren't coloured by
    player."""
    return gr.update(elem_classes=_shown(not by_player))


def _apply_preset(name):
    """Set every caption-look control from a preset; 'Custom' leaves them as-is."""
    preset = gconf.CAPTION_PRESETS.get(name)
    if not preset:
        return tuple(gr.skip() for _ in _PRESET_KEYS)
    return tuple(preset[k] for k in _PRESET_KEYS)


def _matches_preset(preset: dict, values) -> bool:
    """Do the look controls (in _PRESET_KEYS order) equal this preset? The text colour
    only counts when the preset uses a single colour (it's hidden otherwise)."""
    font, size, outline, upper, anim, mode, by_player, color = values
    try:
        same = (font == preset["font"]
                and int(round(float(size))) == int(preset["size"])
                and abs(float(outline) - float(preset["outline"])) < 1e-6
                and bool(upper) == preset["uppercase"]
                and bool(anim) == preset["animate"]
                and mode == preset["mode"]
                and bool(by_player) == preset["by_player"])
    except (TypeError, ValueError):
        return False
    if same and not preset["by_player"]:
        return players_mod.to_hex(color) == players_mod.to_hex(preset["color"])
    return same


def _preset_label(name, *values):
    """Keep the 'Look' label truthful: it names a preset only while every control still
    equals it, else 'Custom'. Runs on .change, so it's right whichever event a widget
    dispatches — and applying a preset matches it by definition, so it never flips."""
    preset = gconf.CAPTION_PRESETS.get(name)
    if preset is None or _matches_preset(preset, values):
        return gr.skip()
    return _CUSTOM


def _build_opts(effects, overlay_choice, pos, start, dur, font, posy, spk_rows,
                reframe_mode=None, x_off=None, y_off=None, fill_frac=None,
                censor=None, censor_audio_mode=None,
                hook_enabled=False, hook_text="", hook_voice="",
                caption_mode=None, caption_offset=None, *,
                captions_enabled=True, caption_size=None, caption_outline=None,
                caption_uppercase=None, caption_animate=None,
                caption_by_player=None, caption_color=None):
    overlay_name = None if overlay_choice in (None, "", _NONE) else overlay_choice

    def pick(v, default):
        return default if v is None else v

    return ManualOptions(
        effects=list(effects or []),
        overlay_name=overlay_name,
        overlay_position=pos,
        overlay_start=float(start or 0),
        overlay_duration=float(dur or 0),
        caption_font=font or gconf.CAPTION_FONT,
        caption_pos_y_frac=float(posy),
        speaker_colors=_parse_speaker_rows(spk_rows),
        reframe_mode=reframe_mode or gconf.REFRAME_MODE,
        crop_x_offset=gconf.REFRAME_CROP_X_OFFSET if x_off is None else float(x_off),
        crop_y_offset=gconf.REFRAME_CROP_Y_OFFSET if y_off is None else float(y_off),
        fill_fraction=gconf.REFRAME_FILL_FRACTION if fill_frac is None else float(fill_frac),
        censor=censor or "both",
        censor_audio_mode=censor_audio_mode or gconf.CENSOR_AUDIO_MODE,
        hook_enabled=bool(hook_enabled),
        hook_text=hook_text or "",
        hook_voice=hook_voice or "",
        caption_mode=caption_mode or gconf.CAPTION_CHUNK_MODE,
        caption_offset=gconf.CAPTION_OFFSET_S if caption_offset is None
        else float(caption_offset),
        captions_enabled=bool(captions_enabled),
        caption_size=int(pick(caption_size, gconf.CAPTION_FONTSIZE)),
        caption_outline=float(pick(caption_outline, gconf.CAPTION_OUTLINE)),
        caption_uppercase=bool(pick(caption_uppercase, gconf.CAPTION_UPPERCASE)),
        caption_animate=bool(pick(caption_animate, gconf.CAPTION_ANIMATE)),
        caption_by_player=bool(pick(caption_by_player, gconf.CAPTION_BY_PLAYER)),
        caption_color=players_mod.to_hex(caption_color, gconf.CAPTION_SINGLE_COLOR),
    )


def _caption_opts(font, size, outline, upper, anim, mode, by_player, color, posy,
                  offset) -> dict:
    """The caption controls (in GUI order) as _build_opts keyword arguments."""
    return dict(font=font, posy=posy, caption_mode=mode, caption_offset=offset,
                caption_size=size, caption_outline=outline, caption_uppercase=upper,
                caption_animate=anim, caption_by_player=by_player,
                caption_color=color)


def _do_preview_captions(tx_clip, rows, spk_rows, captions, font, size, outline,
                         upper, anim, mode, by_player, color, posy, offset):
    """Preview the captions on the clip the transcript BELONGS to (tx_clip), so one
    clip's transcript can never be previewed over another clip's video."""
    if not _captions_on(captions):
        raise gr.Error("Player captions are off — turn them on in step ① to preview.")
    if not tx_clip:
        raise gr.Error("Transcribe a clip first.")
    clip = GameplayClip(tx_clip)
    if not clip.has_source():
        raise gr.Error("No source clip — upload and transcribe first.")
    transcript = Transcript.from_rows(rows)
    opts = _build_opts([], _NONE, gconf.OVERLAY_DEFAULT_POSITION, 0, 0,
                       spk_rows=spk_rows, captions_enabled=True,
                       **_caption_opts(font, size, outline, upper, anim, mode,
                                       by_player, color, posy, offset))
    try:
        return str(manual_mod.preview_captions(clip, transcript, opts))
    except FriendlyError as fe:
        raise gr.Error(str(fe), duration=None)
    except Exception as e:                       # noqa: BLE001
        raise gr.Error(str(friendly(e)), duration=None)


# ---- handlers --------------------------------------------------------------

def _set_clip_name(video):
    if not video:
        return ""
    return slugify(Path(video).stem)


def _do_transcribe(video, diarize, speakers, clip_name):
    """Stream transcribe progress (status only). Caches transcript.json so the
    chained editor-load step can read it back."""
    if not video:
        raise gr.Error("Upload a gameplay clip first.")
    num_speakers = None if str(speakers) in ("", "Auto") else int(speakers)
    log: list[str] = []

    def emit(msg):
        log.append(msg)

    emit("Importing clip...")
    yield "\n".join(log)
    try:
        clip = transcribe_mod.import_source(video, name=clip_name or None)
        # transcribe_clip drives `emit`; we can't interleave yields from inside a
        # blocking call, so we surface the staged messages it logs on return.
        captured: list[str] = []
        transcribe_mod.transcribe_clip(
            clip, progress=lambda m: captured.append(m), force=True,
            diarize=bool(diarize), num_speakers=num_speakers)
        log.extend(captured)
        yield "\n".join(log)
    except FriendlyError as fe:
        raise gr.Error(str(fe), duration=None)
    except Exception as e:                       # noqa: BLE001
        raise gr.Error(str(friendly(e)), duration=None)


def _editor_payload(t: Transcript, rev=None):
    """(rows, editor_html, players, note) for the transcript step. `rev` marks the
    render as an authoritative load (see editor.render_editor)."""
    if not t.single_speaker:
        note = (f"**{len(t.words)} words**, {len(t.speakers)} players. Name each "
                f"player on their card below — the name goes on every line. Edits "
                f"here ARE the captions, so fix any mis-heard words now.")
    elif t.diarized:
        # diarisation RAN but only one voice dominated — not a token problem
        note = (f"**{len(t.words)} words**. Diarisation ran but collapsed to one "
                f"dominant player. If that's wrong, set the speaker per row with the "
                f"buttons, or re-transcribe. Edits here ARE the captions.")
    else:
        note = (f"**{len(t.words)} words**, single speaker (default colour). "
                f"Set `HF_TOKEN` in .env + accept the pyannote licence and "
                f"re-transcribe to tell players apart. Edits here ARE the captions.")
    n_cen = sum(1 for w in t.words if w.censor)
    if n_cen:
        note += f" 🔇 {n_cen} word(s) flagged for censor."
    players = players_mod.sync_players(_speaker_rows(t), t.to_rows())
    rows = t.to_rows()
    return rows, editor_mod.render_editor(rows, players, rev=rev), players, note


def _load_editor(clip_name):
    """Authoritative (re)load of the transcript — after Transcribe and on Revert.
    Outputs: rows, editor_html, players, note, transcribed-clip name, player status,
    then every card slot. Carries a fresh render revision so the editor really
    discards local edits even when the reloaded rows match what it last saw."""
    rev = time.time_ns()
    if clip_name:
        clip = GameplayClip(clip_name)
        if clip.has_transcript():
            rows, html, players, note = _editor_payload(
                Transcript.load(clip.transcript_path), rev=rev)
            return (rows, html, players, note, clip_name, _overflow_note(players),
                    *_card_updates(players, rows))
        note = "No transcript yet — run ① Transcribe."
    else:
        note = "Transcribe a clip to populate the transcript editor."
    return ([], editor_mod.render_editor([], rev=rev), [], note, "", "",
            *_card_updates([], []))


def _preview_hook(text, voice):
    """Synthesize the hook line so the user can hear the voice before building.
    Cached by (text, voice) — repeated previews of the same line don't re-bill."""
    if not (text or "").strip():
        raise gr.Error("Type a hook line to preview.")
    try:
        from gameplay import hook as hook_mod
        wav, _dur = hook_mod.synthesize_hook(text, (voice or "").strip() or None,
                                             gconf.GAMEPLAY_DIR / "_hook_preview")
        return str(wav)
    except Exception as e:                       # noqa: BLE001
        raise gr.Error(str(friendly(e)), duration=None)


def _resolve_build_clip(video, tx_clip, captions):
    """Which clip to build -> (clip name, needs import). Checked against `tx_clip`, the
    clip the transcript rows BELONG to — never against the last clip built, so a
    captions-off build of another clip can't make a valid captions-on build fail.

    Captions OFF: build straight from the uploaded video (no transcription needed).
    Captions ON: build the transcribed clip; refuse if a different clip has been
    uploaded since (it would otherwise get the old clip's captions)."""
    if not _captions_on(captions):
        if not video:
            if tx_clip and GameplayClip(tx_clip).has_source():
                return tx_clip, False
            raise gr.Error("Upload a gameplay clip first.")
        name = slugify(Path(video).stem)
        return name, not GameplayClip(name).has_source()
    if not tx_clip:
        raise gr.Error("Transcribe the clip first (step ①) — or turn player captions "
                       "off to build without a transcript.")
    if video and slugify(Path(video).stem) != tx_clip:
        raise gr.Error("You've loaded a different clip since transcribing — click "
                       "Transcribe for this clip, or turn player captions off.")
    return tx_clip, False


def _prepare_build(video, tx_clip, captions):
    """Resolve (and if needed import) the clip to build, with visible progress —
    importing can re-encode a VFR capture, which takes a while. Yields
    (clip name or None, progress message); the last frame carries the name."""
    name, needs_import = _resolve_build_clip(video, tx_clip, captions)
    if needs_import:
        yield None, "Importing the clip (normalising it if needed)…"
        try:
            transcribe_mod.import_source(video, name=name)
        except FriendlyError as fe:
            raise gr.Error(str(fe), duration=None)
        except Exception as e:                   # noqa: BLE001
            raise gr.Error(str(friendly(e)), duration=None)
    yield name, "Starting the build…"


def _run_build(video, tx_clip, rows, spk_rows, captions, *settings):
    """The whole Build as ONE event: resolve/import the clip, build it, show the
    result. Keeping the clip name local to this call means nothing outside it (a
    second click, a new upload) can swap which clip the result is read from. Clears
    the previous result first so a new build never looks already finished.
    Outputs: build log, result video."""
    name = ""
    for frame_name, msg in _prepare_build(video, tx_clip, captions):
        name = frame_name or name
        yield msg, None
    log = ""
    for log in _do_build(name, tx_clip, rows, spk_rows, captions, *settings):
        yield log, gr.skip()
    out = GameplayClip(name).final_path
    yield log, (str(out) if out.exists() else None)


def _build_busy():
    """Lock the Build button while a build runs (a second click would otherwise be
    silently dropped mid-chain)."""
    return gr.update(interactive=False, value="Building…")


def _build_idle():
    """Unlock the Build button. Chained on BOTH .then and .failure of the build:
    Gradio 6's frontend skips .then steps after an error."""
    return gr.update(interactive=True, value="Build Short")


def _build_intro(captions_on: bool, has_transcript: bool, censor: str,
                 n_rows: int) -> list[str]:
    """Opening build-log lines that say what will actually happen to captions and the
    profanity censor (the audio bleep needs a transcript)."""
    if captions_on:
        return [f"Using your edited transcript ({n_rows} rows)."]
    bleep = censor in ("both", "audio")
    if has_transcript and bleep:
        return ["Player captions off — the transcript is only used to bleep profanity "
                "in the audio."]
    if has_transcript:
        return ["Player captions off — building without captions (censor off)."
                if censor == "off" else
                "Player captions off — building without captions (the 'caption only' "
                "censor has nothing to mask)."]
    if bleep:
        return ["Player captions off — building straight from the clip.",
                "⚠ The profanity bleep needs a transcript, so the audio is NOT censored. "
                "Transcribe this clip first (then turn captions off) if it needs bleeping."]
    return ["Player captions off — building straight from the clip (no transcript)."]


def _do_build(clip_name, tx_clip, rows, spk_rows, captions, effects, overlay_choice,
              pos, start, dur, reframe_mode, x_off, y_off, fill_frac, censor,
              censor_audio_mode, hook_enabled, hook_text, hook_voice,
              font, size, outline, upper, anim, mode, by_player, color, posy, offset):
    if not clip_name:
        raise gr.Error("Upload a gameplay clip first.")
    clip = GameplayClip(clip_name)
    if not clip.has_source():
        raise gr.Error("No source clip — upload a clip first.")
    captions_on = _captions_on(captions)
    # The rows belong to the clip that was transcribed; never apply them to another
    # clip (that would bleep/caption the wrong moments).
    transcript = Transcript.from_rows(rows if tx_clip and tx_clip == clip_name else [])
    if captions_on and not transcript.words:
        raise gr.Error("The transcript is empty — transcribe the clip (or add rows), "
                       "or turn player captions off.")
    opts = _build_opts(effects, overlay_choice, pos, start, dur, spk_rows=spk_rows,
                       reframe_mode=reframe_mode, x_off=x_off, y_off=y_off,
                       fill_frac=fill_frac, censor=censor,
                       censor_audio_mode=censor_audio_mode, hook_enabled=hook_enabled,
                       hook_text=hook_text, hook_voice=hook_voice,
                       captions_enabled=captions_on,
                       **_caption_opts(font, size, outline, upper, anim, mode,
                                       by_player, color, posy, offset))
    # make the source of truth unambiguous: the build uses the CURRENT editor rows, and
    # the log says what really happens to captions + the censor
    log = _build_intro(captions_on, bool(transcript.words), censor or "both",
                       len(transcript.words))
    yield "\n".join(log)
    try:
        for ev in run_manual(clip, transcript, opts, force=True):
            log.append(ev["msg"])
            yield "\n".join(log)
    except FriendlyError as fe:
        raise gr.Error(str(fe), duration=None)
    except Exception as e:                       # noqa: BLE001
        raise gr.Error(str(friendly(e)), duration=None)


def _refresh_overlays():
    return gr.update(choices=[_NONE] + ov_mod.list_overlays())


# ---- tab layout ------------------------------------------------------------

def _step(title: str):
    gr.Markdown(f"#### {title}", elem_classes=["gp-step"])


def build_gameplay_tab() -> "gr.Video":
    """Create the Gameplay tab. Must be called inside the app's gr.Blocks/gr.Tabs
    context. All components and handler wiring are local to this function. Returns the
    clip uploader so the full-auto tab can load a candidate into it."""
    with gr.Tab("🎮 Gameplay"):
        gr.HTML(_GP_STYLE, padding=False, container=False)
        clip_state = gr.State("")        # the clip last sent to Transcribe
        tx_clip_state = gr.State("")     # the clip the transcript rows belong to
        rows_state = gr.State([])        # authoritative [text,speaker,start,end,censor]
        players_state = gr.State([])     # [[name, "#RRGGBB"], ...] — the players
        gr.Markdown(
            "### Gameplay clip → 9:16 Short\n"
            "**①** upload a clip and choose whether to caption the players · "
            "**②** fix the transcript and name the players · **③** style it · "
            "**④** build.", elem_classes=["gp-hint"])

        # ---------------------------------------------------- ① clip & captions
        _step("① Clip & captions")
        with gr.Row():
            clip_video = gr.Video(label="Gameplay clip (pre-trimmed)")
            with gr.Column():
                captions_radio = gr.Radio(
                    choices=[("On — transcribe and caption the players", "on"),
                             ("Off — no captions (skips transcription)", "off")],
                    value="on" if gconf.CAPTION_ON_BY_DEFAULT else "off",
                    label="Player captions")
                caps_on = gconf.CAPTION_ON_BY_DEFAULT
                with gr.Column(elem_classes=_shown(caps_on)) as asr_col:
                    with gr.Row():
                        diarize_cb = gr.Checkbox(
                            value=True,
                            label="Tell players apart (diarize; needs HF_TOKEN)")
                        speakers_dd = gr.Dropdown(
                            choices=["Auto", "1", "2", "3", "4", "5"], value="Auto",
                            label="How many players are talking?",
                            info="Pin it if Auto mixes people up")
                    transcribe_btn = gr.Button("Transcribe / Re-transcribe",
                                               variant="primary")
                    gr.Markdown("Runs WhisperX once — **re-transcribing overwrites your "
                                "edits**. Build never re-runs it.",
                                elem_classes=["gp-hint"])
                    transcribe_status = gr.Textbox(label="Transcribe log", lines=3,
                                                   interactive=False)
                captions_off_md = gr.Markdown(
                    "**Player captions are off** — no transcription needed. Build "
                    "makes the Short straight from the clip; layout, overlay, effects "
                    "and the narrated hook still apply. (The audio bleep needs a "
                    "transcript, so it only applies if you transcribed this clip.)",
                    elem_classes=_shown(not caps_on))

        # ----------------------------------------------- ② transcript & players
        with gr.Column(elem_classes=_shown(caps_on)) as transcript_col:
            _step("② Transcript & players")
            editor_md = gr.Markdown("Transcribe a clip to populate the transcript editor.")
            # editor_html is the interactive view of rows_state; tx_commit/tx_in are the
            # hidden commit channel: the editor's JS clicks tx_commit on each commit and
            # its js= reader pulls the live rows from the DOM (Gradio 6 ignores a
            # programmatic textbox value-set, but a real click + js reader is reliable).
            # Both are hidden via the app-level .txe-hidden CSS rule.
            editor_html = gr.HTML(editor_mod.render_editor([]))
            tx_commit = gr.Button("commit", elem_id=editor_mod.COMMIT_ELEM_ID,
                                  elem_classes=["txe-hidden"])
            tx_in = gr.Textbox("", elem_id="tx-in", elem_classes=["txe-hidden"])
            with gr.Row():
                reload_grid_btn = gr.Button("↺ Revert to last transcribe", scale=0,
                                            size="sm")
                gr.Markdown(
                    "Keyboard-first: **↑/↓** or **Enter** walk rows, type to fix a word, "
                    "**Alt+1…N** set the speaker (a shift-selected range too), **Alt+B** "
                    "speaker to all below, **Alt+D** delete. Profanity auto-censors at "
                    "build; click 🔇 to force it.", elem_classes=["gp-hint"])

            gr.Markdown(
                "**Players** — give each speaker a name and colour. Renaming updates "
                "every line; giving two speakers the same name **merges** them (the fix "
                "when one person was split in two).", elem_classes=["gp-hint"])
            player_status = gr.Markdown()
            cards = []
            with gr.Row(elem_classes=["gp-players"]):
                for _i in range(gconf.MAX_PLAYER_CARDS):
                    with gr.Column(min_width=170,
                                   elem_classes=_card_classes(False)) as card_col:
                        card_head = gr.Markdown()
                        card_key = gr.Textbox(show_label=False, container=False,
                                              elem_classes=["gp-hidden"])
                        card_name = gr.Textbox(show_label=False, container=False,
                                               max_lines=1, placeholder="Player name")
                        card_colour = gr.ColorPicker(show_label=False, container=False)
                    cards.append((card_col, card_head, card_key, card_name, card_colour))
            card_outs = [c for card in cards for c in card]
            card_cols = [card[0] for card in cards]
            card_heads = [card[1] for card in cards]

            # Bulk-edit tools — gameplay ASR/diarization is noisy, so correcting the
            # transcript is a core step. These cover the tedious bits (a whole stretch
            # mislabelled, a name misheard, a mis-segmented phrase).
            with gr.Accordion("✏ Bulk edits", open=False):
                edit_status = gr.Markdown("Row numbers are 1-based.")
                with gr.Row():
                    edit_span_tb = gr.Textbox(
                        label="Rows", placeholder="e.g. 3-10 or 3,4,7", scale=2)
                    edit_speaker_tb = gr.Textbox(label="Player", placeholder="Chan",
                                                 scale=2)
                    edit_assign_btn = gr.Button("Assign player to rows", scale=1)
                with gr.Row():
                    edit_find_tb = gr.Textbox(label="Find", placeholder="Jet", scale=2)
                    edit_repl_tb = gr.Textbox(label="Replace", placeholder="Jett",
                                              scale=2)
                    edit_whole_cb = gr.Checkbox(value=False, label="Whole word")
                    edit_replace_btn = gr.Button("Replace all", scale=1)
                with gr.Row():
                    edit_merge_btn = gr.Button("Merge rows (uses Rows above)")
                    edit_split_row_n = gr.Number(label="Split row #", precision=0)
                    edit_split_btn = gr.Button("Split row")

        # ---------------------------------------------------------------- ③ style
        _step("③ Style")
        with gr.Tabs():
            with gr.Tab("Captions"):
                captions_style_off_md = gr.Markdown(
                    "Player captions are **off** (step ①) — turn them on to style them.",
                    elem_classes=_shown(not caps_on))
                with gr.Column(elem_classes=_shown(caps_on)) as cap_style_col:
                    preset = gconf.CAPTION_PRESETS[gconf.CAPTION_DEFAULT_PRESET]
                    with gr.Row():
                        preset_dd = gr.Dropdown(
                            choices=list(gconf.CAPTION_PRESETS) + [_CUSTOM],
                            value=gconf.CAPTION_DEFAULT_PRESET, label="Look",
                            info="A starting point — every setting stays editable")
                        font_dd = gr.Dropdown(choices=gconf.CAPTION_FONTS,
                                              value=preset["font"], label="Font")
                        caption_mode_dd = gr.Dropdown(
                            choices=[("Short phrases (recommended)", "phrase"),
                                     ("One word at a time (karaoke)", "word")],
                            value=preset["mode"], label="Pacing")
                    with gr.Row():
                        size_sl = gr.Slider(60, 220, value=preset["size"], step=2,
                                            label="Text size")
                        outline_sl = gr.Slider(0, 14, value=preset["outline"],
                                               step=0.5, label="Outline thickness")
                    with gr.Row():
                        upper_cb = gr.Checkbox(value=preset["uppercase"],
                                               label="UPPERCASE")
                        anim_cb = gr.Checkbox(value=preset["animate"],
                                              label="Pop-in animation")
                        by_player_cb = gr.Checkbox(value=preset["by_player"],
                                                   label="Colour each line by player")
                        color_cp = gr.ColorPicker(
                            value=preset["color"], label="Text colour",
                            elem_classes=_shown(not preset["by_player"]))
                    with gr.Row():
                        posy_sl = gr.Slider(0.3, 0.9, value=gconf.CAPTION_POS_Y_FRAC,
                                            step=0.01,
                                            label="Height on screen (higher = lower)",
                                            info="0.72 sits above the overlay banner")
                        caption_offset_sl = gr.Slider(
                            -1.0, 1.0, value=gconf.CAPTION_OFFSET_S, step=0.05,
                            label="Timing nudge (s)", info="− earlier · + later")
                    preview_caps_btn = gr.Button("▶ Preview captions (first 8s)",
                                                 variant="secondary")
                    preview_video = gr.Video(label="Caption preview")
            with gr.Tab("Layout"):
                reframe_mode_dd = gr.Dropdown(
                    choices=[("Fill (gameplay fills frame — recommended)", "fill"),
                             ("Fit & crop (fill, centred)", "fit_crop"),
                             ("Blur-pad (full frame, blurred bars)", "blur_pad"),
                             ("Zoom blur-pad (bigger gameplay band)", "zoom_blur")],
                    value=gconf.REFRAME_MODE, label="Layout (9:16 reframe)")
                with gr.Row():
                    cropx_sl = gr.Slider(0.0, 1.0, value=gconf.REFRAME_CROP_X_OFFSET,
                                         step=0.05,
                                         label="Fill crop X (0=left, 0.5=centre, 1=right)")
                    cropy_sl = gr.Slider(0.0, 1.0, value=gconf.REFRAME_CROP_Y_OFFSET,
                                         step=0.05, label="Fill crop Y (0=top, 1=bottom)")
                    fillfrac_sl = gr.Slider(1.0, 2.0, value=gconf.REFRAME_FILL_FRACTION,
                                            step=0.05, label="Fill zoom (1.0 = just fills)")
            with gr.Tab("Overlay"):
                with gr.Row():
                    _ov_choices = [_NONE] + ov_mod.list_overlays()
                    overlay_dd = gr.Dropdown(
                        choices=_ov_choices,
                        value=(gconf.LIKE_SUB_OVERLAY
                               if gconf.LIKE_SUB_OVERLAY in _ov_choices else _NONE),
                        label="Like/subscribe overlay", scale=3)
                    refresh_ov_btn = gr.Button("↻ Refresh list", scale=0)
                with gr.Row():
                    overlay_pos_dd = gr.Dropdown(choices=ov_mod.POSITIONS,
                                                 value=gconf.OVERLAY_DEFAULT_POSITION,
                                                 label="Position")
                    overlay_start_n = gr.Number(value=gconf.OVERLAY_DEFAULT_START,
                                                label="Start (s)")
                    overlay_dur_n = gr.Number(value=gconf.OVERLAY_DEFAULT_DURATION,
                                              label="Duration (s, 0 = whole clip)")
            with gr.Tab("Audio & extras"):
                with gr.Row():
                    censor_dd = gr.Dropdown(
                        choices=[("Bleep audio + mask caption", "both"),
                                 ("Audio only", "audio"), ("Caption only", "caption"),
                                 ("Off", "off")],
                        value=("both" if gconf.CENSOR_ENABLED else "off"),
                        label="Profanity censor")
                    censor_mode_dd = gr.Dropdown(
                        choices=["bleep", "mute", "duck"],
                        value=gconf.CENSOR_AUDIO_MODE, label="Censor sound")
                effects_cbg = gr.CheckboxGroup(
                    choices=[("Punch-zoom on loud beats", "punch_zoom"),
                             ("Subtle shake on loud beats", "shake")],
                    label="Effects (driven by the audio energy)")
                with gr.Row():
                    hook_enable_cb = gr.Checkbox(
                        value=False,
                        label="Narrated hook (read an opening line over the clip)")
                    hook_text_tb = gr.Textbox(
                        label="Hook line", scale=3,
                        placeholder="The time I got ganked by 3 people playing Yone")
                    hook_voice_tb = gr.Textbox(value=gconf.HOOK_VOICE,
                                               label="Hook voice id", scale=1)
                    hook_preview_btn = gr.Button("▶ Preview voice", scale=0)
                hook_audio = gr.Audio(label="Hook preview", interactive=False)
                gr.Markdown("Game audio ducks under the narration and swells back. The "
                            "hook gets its own caption colour (even with player captions "
                            "off). Preview bills ElevenLabs once per (line, voice) — "
                            "repeats are cached.", elem_classes=["gp-hint"])

        # ---------------------------------------------------------------- ④ build
        _step("④ Build")
        build_btn = gr.Button("Build Short", variant="primary")
        build_status = gr.Textbox(label="Build log", lines=5, interactive=False)
        result_video = gr.Video(label="Result (9:16 Short)")

        # ================================================================ wiring
        # Editors often export MP4s without faststart (moov atom at the end), which the
        # browser can't play ("Video not playable") even though ffmpeg reads them fine.
        # Remux such uploads to faststart so the preview plays (fail-safe: unchanged on
        # any issue). Same path then feeds Transcribe / Build.
        clip_video.upload(transcribe_mod.playable_preview, clip_video, clip_video)

        editor_outs = [rows_state, editor_html, players_state, editor_md, tx_clip_state,
                       player_status, *card_outs]
        reveal = ([players_state, rows_state], card_cols)   # see _card_visibility

        captions_radio.change(_toggle_captions, captions_radio,
                              [asr_col, captions_off_md, transcript_col, cap_style_col,
                               captions_style_off_md])
        transcribe_btn.click(_set_clip_name, clip_video, clip_state) \
            .then(_do_transcribe, [clip_video, diarize_cb, speakers_dd, clip_state],
                  transcribe_status) \
            .then(_load_editor, clip_state, editor_outs) \
            .then(_card_visibility, *reveal)
        # the editor's JS clicks tx_commit on each commit (blur / Enter / a speaker or
        # row op); the js reader returns the live rows -> rows_state, then the player
        # cards' line counts refresh. No editor re-render here, so no echo loop. It
        # shares the players' concurrency group so a commit and a rename (both write
        # rows_state) apply in the order they were made.
        tx_commit.click(_rows_from_dom, tx_in, rows_state, js=editor_mod.READ_ROWS_JS,
                        concurrency_id="gp-players") \
            .then(_card_heads, [players_state, rows_state], card_heads)
        # discard accidental edits and reload from the saved transcript.json
        reload_grid_btn.click(_load_editor, clip_state, editor_outs) \
            .then(_card_visibility, *reveal)

        # player cards: a name commits on blur (Enter blurs the field, so there's one
        # commit path, never two racing). On that same blur the browser relabels the
        # editor in place straight away (window.__txeRename) — focus in a row survives,
        # and a Build clicked right after already reads the new labels — while the
        # server updates the state. Colour commits on change (no-op when a card refresh
        # merely re-sets the same colour). The hidden key pins which player the card
        # showed. Shared concurrency keeps renames, recolours and commits ordered.
        rename_outs = [players_state, rows_state, editor_html, player_status, *card_outs]
        for slot, (_col, _head, key_tb, name_tb, colour_cp) in enumerate(cards):
            name_tb.submit(None, None, None,
                           js="() => { const a = document.activeElement; "
                              "if (a) a.blur(); }")
            name_tb.blur(None, [key_tb, name_tb], None,
                         js="(k, n) => { if (window.__txeRename) "
                            "window.__txeRename(k, n); }")
            name_tb.blur(_rename_handler(slot),
                         [key_tb, name_tb, players_state, rows_state],
                         rename_outs, concurrency_id="gp-players")
            colour_cp.change(_recolour_card,
                             [key_tb, colour_cp, players_state, rows_state],
                             [players_state, editor_html], concurrency_id="gp-players")

        # bulk-edit wiring — each reuses an editing.py helper on rows_state, then the
        # player list syncs (a new name typed into 'Assign player' gets a card). They
        # share the players' concurrency group, so a row commit fired by the click's
        # blur lands BEFORE the bulk edit reads rows_state.
        sync_outs = [players_state, editor_html, player_status, *card_outs]
        edit_assign_btn.click(_edit_assign,
                              [rows_state, edit_span_tb, edit_speaker_tb, players_state],
                              [rows_state, editor_html, edit_status],
                              concurrency_id="gp-players") \
            .then(_sync_cards, [players_state, rows_state], sync_outs) \
            .then(_card_visibility, *reveal)
        edit_replace_btn.click(_edit_find_replace,
                               [rows_state, edit_find_tb, edit_repl_tb, edit_whole_cb,
                                players_state],
                               [rows_state, editor_html, edit_status],
                               concurrency_id="gp-players")
        edit_merge_btn.click(_edit_merge, [rows_state, edit_span_tb, players_state],
                             [rows_state, editor_html, edit_status],
                             concurrency_id="gp-players") \
            .then(_card_heads, [players_state, rows_state], card_heads)
        edit_split_btn.click(_edit_split, [rows_state, edit_split_row_n, players_state],
                             [rows_state, editor_html, edit_status],
                             concurrency_id="gp-players") \
            .then(_card_heads, [players_state, rows_state], card_heads)

        # caption look: a preset sets every control; the label then stays truthful —
        # it reads "Custom" as soon as any control no longer equals the chosen preset.
        look_controls = [font_dd, size_sl, outline_sl, upper_cb, anim_cb,
                         caption_mode_dd, by_player_cb, color_cp]
        preset_dd.input(_apply_preset, preset_dd, look_controls)
        for comp in look_controls:
            comp.change(_preset_label, [preset_dd, *look_controls], preset_dd)
        by_player_cb.change(_colour_picker_classes, by_player_cb, color_cp)

        caption_controls = [font_dd, size_sl, outline_sl, upper_cb, anim_cb,
                            caption_mode_dd, by_player_cb, color_cp, posy_sl,
                            caption_offset_sl]
        preview_caps_btn.click(_rows_from_dom, tx_in, rows_state,
                               js=editor_mod.READ_ROWS_JS,
                               concurrency_id="gp-players") \
            .then(_do_preview_captions,
                  [tx_clip_state, rows_state, players_state, captions_radio,
                   *caption_controls],
                  preview_video)

        refresh_ov_btn.click(_refresh_overlays, None, overlay_dd)
        hook_preview_btn.click(_preview_hook, [hook_text_tb, hook_voice_tb], hook_audio)

        build_inputs = [clip_video, tx_clip_state, rows_state, players_state,
                        captions_radio, effects_cbg, overlay_dd, overlay_pos_dd,
                        overlay_start_n, overlay_dur_n, reframe_mode_dd, cropx_sl,
                        cropy_sl, fillfrac_sl, censor_dd, censor_mode_dd,
                        hook_enable_cb, hook_text_tb, hook_voice_tb, *caption_controls]
        # lock the button, flush the editor's latest in-DOM edits into rows_state (the
        # click blurs the focused row, but the blur-commit round-trip may not have
        # landed yet — this js read guarantees rows_state is current), then run the
        # whole build as one event (resolve/import -> build -> result, so nothing can
        # swap the clip between steps), then unlock. Gradio 6's frontend runs NO .then
        # step after an error, so the unlock hangs off both outcomes of every step that
        # can fail — otherwise a refused build (e.g. not transcribed yet) left the
        # button stuck on 'Building…' until a reload, which loses the transcript edits.
        flush = build_btn.click(_build_busy, None, build_btn) \
            .then(_rows_from_dom, tx_in, rows_state, js=editor_mod.READ_ROWS_JS,
                  concurrency_id="gp-players")
        flush.failure(_build_idle, None, build_btn)
        run = flush.success(_run_build, build_inputs, [build_status, result_video])
        run.then(_build_idle, None, build_btn)
        run.failure(_build_idle, None, build_btn)

        # exposed so the full-auto tab can hand a detected candidate straight into this
        # uploader ("Refine in manual mode") — the cross-tab wiring lives in app.py.
        return clip_video
