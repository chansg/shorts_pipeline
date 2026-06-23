"""The Gameplay tab for the unified GUI — a self-contained module so app.py only
needs an import + one call inside its `gr.Tabs()` block (the lore wizard is
untouched).

Manual mode only: upload a pre-trimmed clip -> Transcribe -> editable transcript
gate (text/speaker/timing + speaker colours) -> reframe + caption + effects toggles
+ overlay picker -> Build (streams) -> 9:16 preview.

The experimental full-auto long-video processor lives in its own landing entry
(fullauto/) and exports a 16:9 YouTube video — it is NOT part of this page.

Streaming handlers update only their status Textbox while running (yielding
gr.update() for other outputs), then a chained .then() populates the editor /
preview from disk — mirroring how app.py's run_assemble streams.
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr

from orchestrator.errors import FriendlyError, friendly
from gameplay import config as gconf
from gameplay import editing as edit_mod
from gameplay import editor as editor_mod
from gameplay import manual as manual_mod
from gameplay import overlay as ov_mod
from gameplay import transcribe as transcribe_mod
from gameplay.manual import ManualOptions, run_manual
from gameplay.state import GameplayClip, slugify
from gameplay.transcript import Transcript

_NONE = "(none)"


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
    """Prefill the speaker→colour grid (and the editor's speaker buttons) with palette
    hex colours. Detected speakers come first (each the colour the renderer auto-assigns),
    then we PAD with default `SPEAKER_NN` rows up to DEFAULT_SPEAKER_ROWS so there are
    always at least that many speakers to assign to — handy when the diariser found fewer
    than were really talking. An unused speaker row is harmless (a colour only applies to
    a speaker that appears in a cue)."""
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
    out = {}
    for row in rows or []:
        row = list(row) + ["", ""]
        name = str(row[0] or "").strip()
        rgb = _hex_to_rgb(row[1])
        if name and rgb:
            out[name] = rgb
    return out


# ---- transcript editor handlers (thin wrappers over gameplay.editing) ------

# Bulk-edit handlers operate on the authoritative rows_state and RE-RENDER the editor
# HTML (the JS observer rebuilds from the new data). They reuse the pure editing.py
# helpers verbatim. Each returns (rows, editor_html, status); spk_rows colours the view.
def _edit_assign(rows, span, speaker, spk_rows):
    rows2 = edit_mod.assign_speaker(rows, span, speaker)
    return (rows2, editor_mod.render_editor(rows2, spk_rows),
            f"Assigned '{speaker}' to rows {span or '(none)'}.")


def _edit_find_replace(rows, find, repl, whole_word, spk_rows):
    rows2, n = edit_mod.find_replace(rows, find, repl, whole_word=bool(whole_word))
    return (rows2, editor_mod.render_editor(rows2, spk_rows),
            f"Replaced {n} occurrence(s) of '{find}'.")


def _edit_merge(rows, span, spk_rows):
    rows2 = edit_mod.merge_rows(rows, span)
    return (rows2, editor_mod.render_editor(rows2, spk_rows),
            f"Merged rows {span or '(none)'}.")


def _edit_split(rows, row_num, spk_rows):
    try:
        i = int(row_num)
    except (TypeError, ValueError):
        return rows, gr.skip(), "Enter a row number to split."
    rows2 = edit_mod.split_row(rows, i)
    return rows2, editor_mod.render_editor(rows2, spk_rows), f"Split row {i}."


def _rows_from_dom(payload):
    """Commit handler: the js reader returned the live editor rows JSON -> rows_state.
    On a parse miss keep the prior state (gr.skip) instead of wiping edits; never
    re-renders the editor, so there's no echo loop."""
    rows = editor_mod.parse_bridge(payload)
    return rows if rows is not None else gr.skip()


def _editor_after_speaker(rows, spk_rows):
    """Re-render the editor so speaker recolours/renames show inline as the colour grid
    is edited."""
    return editor_mod.render_editor(rows, spk_rows)


def _do_preview_captions(clip_name, rows, spk_rows, font, posy):
    if not clip_name:
        raise gr.Error("Transcribe a clip first.")
    clip = GameplayClip(clip_name)
    if not clip.has_source():
        raise gr.Error("No source clip — upload and transcribe first.")
    transcript = Transcript.from_rows(rows)
    opts = _build_opts([], _NONE, gconf.OVERLAY_DEFAULT_POSITION, 0, 0, font, posy,
                       spk_rows)
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


def _swatches_md(spk_rows) -> str:
    """Inline speaker→colour chips so the mapping is visible while editing."""
    chips = []
    for row in spk_rows or []:
        row = list(row) + ["", ""]
        name = str(row[0] or "").strip()
        if not name:
            continue
        hexv = str(row[1] or "").strip() or "#888888"
        chips.append(
            f"<span style='display:inline-block;padding:1px 8px;margin:2px;"
            f"border-radius:6px;background:{hexv};color:#000;font-weight:600'>"
            f"{name}</span>")
    return ("**Speaker colours:** " + " ".join(chips)) if chips else ""


def _editor_payload(t: Transcript):
    """(rows, editor_html, speaker_rows, note, swatch_md) for the transcript editor."""
    if not t.single_speaker:
        note = (f"**{len(t.words)} words**, {len(t.speakers)} speakers. "
                f"Rename speakers in the *speaker* column; recolour below. "
                f"Edits here ARE the captions — fix any ASR errors now.")
    elif t.diarized:
        # diarisation RAN but only one voice dominated — not a token problem
        note = (f"**{len(t.words)} words**. Diarisation ran but collapsed to one "
                f"dominant speaker. If that's wrong, set the speaker per row in the "
                f"grid, or re-transcribe. Edits here ARE the captions.")
    else:
        note = (f"**{len(t.words)} words**, single speaker (default colour). "
                f"Set `HF_TOKEN` in .env + accept the pyannote licence and "
                f"re-transcribe to colour per speaker. Edits here ARE the captions.")
    n_cen = sum(1 for w in t.words if w.censor)
    if n_cen:
        note += f" 🔇 {n_cen} word(s) flagged for censor."
    spk_rows = _speaker_rows(t)
    rows = t.to_rows()
    return (rows, editor_mod.render_editor(rows, spk_rows), spk_rows, note,
            _swatches_md(spk_rows))


def _load_editor(clip_name):
    empty = ([], editor_mod.render_editor([]), [],
             "Transcribe a clip to populate the transcript editor.", "")
    if not clip_name:
        return empty
    clip = GameplayClip(clip_name)
    if not clip.has_transcript():
        return ([], editor_mod.render_editor([]), [], "No transcript yet — run "
                "Transcribe.", "")
    return _editor_payload(Transcript.load(clip.transcript_path))


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


def _build_opts(effects, overlay_choice, pos, start, dur, font, posy, spk_rows,
                reframe_mode=None, x_off=None, y_off=None, fill_frac=None,
                censor=None, censor_audio_mode=None,
                hook_enabled=False, hook_text="", hook_voice="",
                caption_mode=None, caption_offset=None):
    overlay_name = None if overlay_choice in (None, "", _NONE) else overlay_choice
    return ManualOptions(
        effects=list(effects or []),
        overlay_name=overlay_name,
        overlay_position=pos,
        overlay_start=float(start or 0),
        overlay_duration=float(dur or 0),
        caption_font=font,
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
    )


def _do_build(clip_name, rows, spk_rows, effects, overlay_choice, pos, start,
              dur, font, posy, reframe_mode, x_off, y_off, fill_frac,
              censor, censor_audio_mode, hook_enabled, hook_text, hook_voice,
              caption_mode, caption_offset):
    if not clip_name:
        raise gr.Error("Transcribe a clip first.")
    clip = GameplayClip(clip_name)
    if not clip.has_source():
        raise gr.Error("No source clip — upload and transcribe first.")
    transcript = Transcript.from_rows(rows)
    if not transcript.words:
        raise gr.Error("The transcript grid is empty — transcribe a clip (or add "
                       "rows) before building.")
    opts = _build_opts(effects, overlay_choice, pos, start, dur, font, posy, spk_rows,
                       reframe_mode, x_off, y_off, fill_frac, censor, censor_audio_mode,
                       hook_enabled, hook_text, hook_voice, caption_mode, caption_offset)
    # make the source of truth unambiguous: the build uses the CURRENT grid, edits included
    log: list[str] = [f"Using your edited transcript ({len(transcript.words)} rows)."]
    yield "\n".join(log)
    try:
        for ev in run_manual(clip, transcript, opts, force=True):
            log.append(ev["msg"])
            yield "\n".join(log)
    except FriendlyError as fe:
        raise gr.Error(str(fe), duration=None)
    except Exception as e:                       # noqa: BLE001
        raise gr.Error(str(friendly(e)), duration=None)


def _show_result(clip_name):
    if not clip_name:
        return None
    p = GameplayClip(clip_name).final_path
    return str(p) if p.exists() else None


def _refresh_overlays():
    return gr.update(choices=[_NONE] + ov_mod.list_overlays())


# ---- tab layout ------------------------------------------------------------

def build_gameplay_tab() -> "gr.Video":
    """Create the Gameplay tab. Must be called inside the app's gr.Blocks/gr.Tabs
    context. All components and handler wiring are local to this function. Returns the
    clip uploader so the full-auto tab can load a candidate into it."""
    with gr.Tab("🎮 Gameplay"):
        clip_state = gr.State("")
        gr.Markdown(
            "### Gameplay → Short\nUpload a pre-trimmed clip, fix the transcript, "
            "then reframe + caption + effects + overlay into a 9:16 Short. "
            "Transcription uses WhisperX on your GPU (diarization needs `HF_TOKEN`).")

        # -- 1. upload + transcribe --
        with gr.Row():
            clip_video = gr.Video(label="Gameplay clip (pre-trimmed)")
            with gr.Column():
                diarize_cb = gr.Checkbox(
                    value=True,
                    label="Diarize speakers (needs HF_TOKEN; else single-speaker)")
                speakers_dd = gr.Dropdown(
                    choices=["Auto", "1", "2", "3", "4", "5"], value="Auto",
                    label="Speakers (pin the count if Auto mislabels — it can't split "
                          "two voices talking at once)")
                transcribe_btn = gr.Button("① Transcribe / Re-transcribe",
                                           variant="primary")
                gr.Markdown("Runs WhisperX (the only ASR step) — **overwrites any "
                            "grid edits**. Build never re-runs ASR.")
                transcribe_status = gr.Textbox(label="Transcribe log", lines=4,
                                               interactive=False)

        # -- 2. transcript gate (fast keyboard editor — see gameplay/editor.py) --
        editor_md = gr.Markdown("Transcribe a clip to populate the transcript editor.")
        # rows_state is the authoritative [text,speaker,start,end,censor] list the build
        # reads; editor_html is the interactive view; tx_bridge is the hidden commit
        # channel (the editor's JS writes rows JSON here on commit -> .input -> rows_state).
        rows_state = gr.State([])
        editor_html = gr.HTML(editor_mod.render_editor([]))
        # Commit channel: the editor's JS clicks this hidden button on each commit; its
        # js= reader pulls the live rows straight from the DOM into Python (Gradio 6
        # ignores a programmatic textbox value-set, but a real button click + js reader
        # is reliable). tx_in is the real-component input slot the js return replaces.
        # Both are hidden via the .txe-hidden CSS rule (app-level).
        tx_commit = gr.Button("commit", elem_id=editor_mod.COMMIT_ELEM_ID,
                              elem_classes=["txe-hidden"])
        tx_in = gr.Textbox("", elem_id="tx-in", elem_classes=["txe-hidden"])
        with gr.Row():
            reload_grid_btn = gr.Button("↺ Revert to last transcribe", scale=0,
                                        size="sm")
            gr.Markdown(
                "Keyboard-first: **↑/↓** or **Enter** walk rows, type to fix a word, "
                "**Alt+1…N** set the speaker (a shift-selected range too), **Alt+B** "
                "speaker to all below, **Alt+D** delete. Profanity auto-censors at build; "
                "click 🔇 to force it. The buttons below reuse the same edits.")
        speaker_swatch_md = gr.Markdown()        # inline speaker→colour chips
        speaker_df = gr.Dataframe(
            headers=["speaker", "color (hex)"], datatype=["str", "str"],
            type="array", interactive=True, row_count=(1, "dynamic"),
            column_widths=["62%", "38%"], wrap=True, max_height=220,
            label="Speaker colours (blank = auto palette; explicit hex wins)")

        # Bulk-edit tools — gameplay ASR/diarization is noisy, so correcting the
        # transcript is a core step. Edit text inline in the grid above; use these
        # for the tedious bits (a whole stretch mislabelled, a name misheard, a
        # mis-segmented phrase). Then preview just the captions before building.
        with gr.Accordion("✏ Bulk edits & caption preview", open=False):
            edit_status = gr.Markdown("Row numbers are 1-based (see the grid).")
            with gr.Row():
                edit_span_tb = gr.Textbox(
                    label="Rows", placeholder="e.g. 3-10 or 3,4,7", scale=2)
                edit_speaker_tb = gr.Textbox(label="Speaker", placeholder="Chan",
                                             scale=2)
                edit_assign_btn = gr.Button("Assign speaker to rows", scale=1)
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
            with gr.Row():
                preview_caps_btn = gr.Button("↻ Re-apply captions (preview)",
                                             variant="secondary")
            preview_video = gr.Video(label="Caption preview (first 8s, captions only)")

        # -- 3. styling controls --
        with gr.Row():
            effects_cbg = gr.CheckboxGroup(
                choices=[("Punch-zoom on loud beats", "punch_zoom"),
                         ("Subtle shake on loud beats", "shake")],
                label="Effects (audio-energy driven; optional)")
            font_dd = gr.Dropdown(choices=["Anton", "Arial"],
                                  value=gconf.CAPTION_FONT, label="Caption font")
            posy_sl = gr.Slider(0.3, 0.9, value=gconf.CAPTION_POS_Y_FRAC, step=0.01,
                                label="Caption Y (higher = lower; 0.72 reads on the "
                                      "fill layout, clear of the overlay)")
        with gr.Row():
            caption_mode_dd = gr.Dropdown(
                choices=[("Phrase chunks (forgiving — recommended)", "phrase"),
                         ("One word at a time (karaoke)", "word")],
                value=gconf.CAPTION_CHUNK_MODE, label="Caption style")
            caption_offset_sl = gr.Slider(
                -1.0, 1.0, value=gconf.CAPTION_OFFSET_S, step=0.05,
                label="Caption timing nudge (s; − earlier, + later)")
        with gr.Row():
            _ov_choices = [_NONE] + ov_mod.list_overlays()
            overlay_dd = gr.Dropdown(
                choices=_ov_choices,
                value=(gconf.LIKE_SUB_OVERLAY if gconf.LIKE_SUB_OVERLAY in _ov_choices
                       else _NONE),
                label="Like/subscribe overlay")
            overlay_pos_dd = gr.Dropdown(choices=ov_mod.POSITIONS,
                                         value=gconf.OVERLAY_DEFAULT_POSITION,
                                         label="Overlay position")
            overlay_start_n = gr.Number(value=gconf.OVERLAY_DEFAULT_START,
                                        label="Overlay start (s)")
            overlay_dur_n = gr.Number(value=gconf.OVERLAY_DEFAULT_DURATION,
                                      label="Overlay duration (s, 0 = whole clip)")
            refresh_ov_btn = gr.Button("↻ overlays")
        with gr.Row():
            reframe_mode_dd = gr.Dropdown(
                choices=[("Fill (gameplay fills frame — recommended)", "fill"),
                         ("Fit & crop (fill, centred)", "fit_crop"),
                         ("Blur-pad (full frame, blurred bars)", "blur_pad"),
                         ("Zoom blur-pad (bigger gameplay band)", "zoom_blur")],
                value=gconf.REFRAME_MODE, label="Layout (9:16 reframe)")
            cropx_sl = gr.Slider(0.0, 1.0, value=gconf.REFRAME_CROP_X_OFFSET, step=0.05,
                                 label="Fill crop X (0=left, 0.5=centre, 1=right)")
            cropy_sl = gr.Slider(0.0, 1.0, value=gconf.REFRAME_CROP_Y_OFFSET, step=0.05,
                                 label="Fill crop Y (0=top, 1=bottom)")
            fillfrac_sl = gr.Slider(1.0, 2.0, value=gconf.REFRAME_FILL_FRACTION,
                                    step=0.05, label="Fill zoom (1.0 = just fills)")
        with gr.Row():
            censor_dd = gr.Dropdown(
                choices=[("Bleep audio + mask caption", "both"),
                         ("Audio only", "audio"), ("Caption only", "caption"),
                         ("Off", "off")],
                value=("both" if gconf.CENSOR_ENABLED else "off"),
                label="Profanity censor")
            censor_mode_dd = gr.Dropdown(
                choices=["bleep", "mute", "duck"], value=gconf.CENSOR_AUDIO_MODE,
                label="Censor audio mode")
        with gr.Row():
            hook_enable_cb = gr.Checkbox(
                value=False, label="Narrated hook (read an opening line over the clip)")
            hook_text_tb = gr.Textbox(
                label="Hook line", scale=3,
                placeholder="The time I got ganked by 3 people playing Yone")
            hook_voice_tb = gr.Textbox(value=gconf.HOOK_VOICE, label="Hook voice id",
                                       scale=1)
            hook_preview_btn = gr.Button("▶ Preview voice", scale=0)
        hook_audio = gr.Audio(label="Hook preview", interactive=False)
        gr.Markdown("Game audio ducks under the narration and swells back. The hook "
                    "is captioned in the NARRATOR colour. Preview bills ElevenLabs "
                    "once per (line, voice) — repeats are cached.")

        # -- 4. build --
        with gr.Row():
            build_btn = gr.Button("② Build from current transcript",
                                   variant="primary")
        build_status = gr.Textbox(label="Build log", lines=5, interactive=False)
        result_video = gr.Video(label="Result (9:16 Short)")

        # -- wiring --
        # Editors often export MP4s without faststart (moov atom at the end), which the
        # browser can't play ("Video not playable") even though ffmpeg reads them fine.
        # Remux such uploads to faststart so the preview plays (fail-safe: unchanged on
        # any issue). Same path then feeds Transcribe.
        clip_video.upload(transcribe_mod.playable_preview, clip_video, clip_video)
        _editor_out = [rows_state, editor_html, speaker_df, editor_md, speaker_swatch_md]
        transcribe_btn.click(_set_clip_name, clip_video, clip_state) \
            .then(_do_transcribe, [clip_video, diarize_cb, speakers_dd, clip_state],
                  transcribe_status) \
            .then(_load_editor, clip_state, _editor_out)
        # the editor's JS clicks tx_commit on each commit (blur / Enter / a speaker or
        # row op); the js reader returns the live rows -> rows_state. No editor re-render
        # here, so there is no echo loop; gr.skip() keeps prior state on a parse miss.
        tx_commit.click(_rows_from_dom, tx_in, rows_state, js=editor_mod.READ_ROWS_JS)
        # colour-grid edits: refresh the inline swatches AND recolour the editor inline
        speaker_df.change(_swatches_md, speaker_df, speaker_swatch_md)
        speaker_df.change(_editor_after_speaker, [rows_state, speaker_df], editor_html)
        # discard accidental edits and reload from the saved transcript.json
        reload_grid_btn.click(_load_editor, clip_state, _editor_out)

        # bulk-edit wiring — each reuses an editing.py helper on rows_state and
        # re-renders the editor (the JS observer rebuilds from the new data).
        edit_assign_btn.click(_edit_assign,
                              [rows_state, edit_span_tb, edit_speaker_tb, speaker_df],
                              [rows_state, editor_html, edit_status])
        edit_replace_btn.click(_edit_find_replace,
                               [rows_state, edit_find_tb, edit_repl_tb, edit_whole_cb,
                                speaker_df],
                               [rows_state, editor_html, edit_status])
        edit_merge_btn.click(_edit_merge, [rows_state, edit_span_tb, speaker_df],
                             [rows_state, editor_html, edit_status])
        edit_split_btn.click(_edit_split, [rows_state, edit_split_row_n, speaker_df],
                             [rows_state, editor_html, edit_status])
        preview_caps_btn.click(_rows_from_dom, tx_in, rows_state,
                               js=editor_mod.READ_ROWS_JS) \
            .then(_do_preview_captions,
                  [clip_state, rows_state, speaker_df, font_dd, posy_sl],
                  preview_video)

        refresh_ov_btn.click(_refresh_overlays, None, overlay_dd)

        hook_preview_btn.click(_preview_hook, [hook_text_tb, hook_voice_tb], hook_audio)

        build_inputs = [clip_state, rows_state, speaker_df, effects_cbg,
                        overlay_dd, overlay_pos_dd, overlay_start_n, overlay_dur_n,
                        font_dd, posy_sl, reframe_mode_dd, cropx_sl, cropy_sl,
                        fillfrac_sl, censor_dd, censor_mode_dd,
                        hook_enable_cb, hook_text_tb, hook_voice_tb,
                        caption_mode_dd, caption_offset_sl]
        # flush the editor's latest in-DOM edits into rows_state BEFORE building (the
        # click blurs the focused row, but the blur-commit round-trip may not have
        # landed yet — this js read guarantees rows_state is current).
        build_btn.click(_rows_from_dom, tx_in, rows_state,
                        js=editor_mod.READ_ROWS_JS) \
            .then(_do_build, build_inputs, build_status) \
            .then(_show_result, clip_state, result_video)

        # exposed so the full-auto tab can hand a detected candidate straight into this
        # uploader ("Refine in manual mode") — the cross-tab wiring lives in app.py.
        return clip_video
