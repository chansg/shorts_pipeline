"""The Gameplay tab for the unified GUI — a self-contained module so app.py only
needs an import + one call inside its `gr.Tabs()` block (the lore wizard is
untouched).

Manual mode (fully working):
  upload clip -> Transcribe -> editable transcript gate (text/speaker/timing +
  speaker colours) -> reframe + caption + effects toggles + overlay picker ->
  Build (streams) -> preview.

Full-auto mode lives under an "Experimental" accordion in the same tab.

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
from gameplay import manual as manual_mod
from gameplay import autopipeline as ap_mod
from gameplay import overlay as ov_mod
from gameplay import transcribe as transcribe_mod
from gameplay.manual import ManualOptions, run_manual
from gameplay.state import AutoSession, GameplayClip, slugify
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
    """Prefill the speaker→colour grid with the palette the renderer would
    auto-assign, so the user sees (and can tweak) the actual colours."""
    rows = []
    for i, s in enumerate(t.speakers):
        rgb = gconf.SPEAKER_PALETTE[i % len(gconf.SPEAKER_PALETTE)]
        rows.append([s, _rgb_to_hex(rgb)])
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

def _edit_assign(rows, span, speaker):
    return edit_mod.assign_speaker(rows, span, speaker), f"Assigned '{speaker}' to rows {span or '(none)'}."


def _edit_find_replace(rows, find, repl, whole_word):
    rows2, n = edit_mod.find_replace(rows, find, repl, whole_word=bool(whole_word))
    return rows2, f"Replaced {n} occurrence(s) of '{find}'."


def _edit_merge(rows, span):
    return edit_mod.merge_rows(rows, span), f"Merged rows {span or '(none)'}."


def _edit_split(rows, row_num):
    try:
        i = int(row_num)
    except (TypeError, ValueError):
        return rows, "Enter a row number to split."
    return edit_mod.split_row(rows, i), f"Split row {i}."


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


def _do_transcribe(video, diarize, clip_name):
    """Stream transcribe progress (status only). Caches transcript.json so the
    chained editor-load step can read it back."""
    if not video:
        raise gr.Error("Upload a gameplay clip first.")
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
            diarize=bool(diarize))
        log.extend(captured)
        yield "\n".join(log)
    except FriendlyError as fe:
        raise gr.Error(str(fe), duration=None)
    except Exception as e:                       # noqa: BLE001
        raise gr.Error(str(friendly(e)), duration=None)


def _editor_payload(t: Transcript):
    """(rows, speaker_rows, note) for the transcript editor from a Transcript."""
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
    return t.to_rows(), _speaker_rows(t), note


def _load_editor(clip_name):
    if not clip_name:
        return [], [], "Transcribe a clip to populate the transcript editor."
    clip = GameplayClip(clip_name)
    if not clip.has_transcript():
        return [], [], "No transcript yet — run Transcribe."
    return _editor_payload(Transcript.load(clip.transcript_path))


def _build_opts(effects, overlay_choice, pos, start, dur, font, posy, spk_rows):
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
    )


def _do_build(clip_name, rows, spk_rows, effects, overlay_choice, pos, start,
              dur, font, posy):
    if not clip_name:
        raise gr.Error("Transcribe a clip first.")
    clip = GameplayClip(clip_name)
    if not clip.has_source():
        raise gr.Error("No source clip — upload and transcribe first.")
    transcript = Transcript.from_rows(rows)
    if not transcript.words:
        raise gr.Error("The transcript grid is empty — transcribe a clip (or add "
                       "rows) before building.")
    opts = _build_opts(effects, overlay_choice, pos, start, dur, font, posy, spk_rows)
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


# ---- full-auto: detect -> review gallery -> load-into-manual / batch --------

def _do_detect(video, backend, max_clips, diarize):
    """Detect + rank candidates for a long video (no building). Streams the staged
    progress, persists candidates + thumbnails for the review step."""
    if not video:
        raise gr.Error("Upload a long gameplay video first.")
    captured: list[str] = []
    try:
        ap_mod.detect_candidates(
            video, backend=backend, max_clips=int(max_clips), diarize=bool(diarize),
            progress=lambda m: captured.append(m))
    except FriendlyError as fe:
        raise gr.Error(str(fe), duration=None)
    except Exception as e:                       # noqa: BLE001
        raise gr.Error(str(friendly(e)), duration=None)
    yield "\n".join(captured)


def _cand_label(i, c) -> str:
    return (f"{i+1}. [{c.category}] {c.start:.0f}-{c.end:.0f}s · "
            f"{c.score:.2f} · {c.caption or '(no caption)'}")


def _load_candidates_ui(video):
    """Populate the review gallery / table / selector from the persisted session."""
    if not video:
        return [], [], gr.update(choices=[], value=[])
    session = AutoSession(Path(video).stem)
    cands = ap_mod.load_candidates(session)
    gallery, rows, labels = [], [], []
    for i, c in enumerate(cands):
        label = _cand_label(i, c)
        labels.append(label)
        rows.append([i + 1, c.category, f"{c.start:.0f}-{c.end:.0f}s",
                     round(c.score, 2), c.caption])
        thumb = session.preview_path(i)
        if thumb.exists():
            gallery.append((str(thumb), label))
    return gallery, rows, gr.update(choices=labels, value=[])


def _selected_indices(selected) -> list[int]:
    out = []
    for label in selected or []:
        try:
            out.append(int(str(label).split(".", 1)[0]) - 1)
        except (ValueError, IndexError):
            continue
    return sorted(set(out))


def _load_into_manual(video, selected):
    """Load the FIRST selected candidate into the manual flow (cut clip + sliced
    transcript), so the user QCs + builds it there."""
    if not video:
        raise gr.Error("Run Detect first.")
    idxs = _selected_indices(selected)
    if not idxs:
        raise gr.Error("Tick a candidate to load into manual mode.")
    session = AutoSession(Path(video).stem)
    cands = ap_mod.load_candidates(session)
    cand = cands[idxs[0]]
    try:
        clip, sub = ap_mod.load_candidate(session, video, cand)
    except FriendlyError as fe:
        raise gr.Error(str(fe), duration=None)
    rows, spk_rows, note = _editor_payload(sub)
    note = (f"Loaded full-auto candidate **[{cand.category}] "
            f"{cand.start:.0f}-{cand.end:.0f}s** into manual mode. " + note)
    return clip.name, str(clip.source_path()), rows, spk_rows, note


def _batch_build(video, selected, effects, overlay_choice, pos, start, dur, font,
                 posy, spk_rows):
    """Optional: build all selected candidates with the current manual defaults."""
    if not video:
        raise gr.Error("Run Detect first.")
    idxs = _selected_indices(selected)
    if not idxs:
        raise gr.Error("Tick candidates to batch-build.")
    session = AutoSession(Path(video).stem)
    cands = ap_mod.load_candidates(session)
    opts = _build_opts(effects, overlay_choice, pos, start, dur, font, posy, spk_rows)
    log, files = [], []
    for n, i in enumerate(idxs):
        c = cands[i]
        tag = f"{n+1}/{len(idxs)} [{c.category}] {c.start:.0f}-{c.end:.0f}s"
        log.append(f"Building {tag}...")
        yield "\n".join(log), gr.update()
        try:
            out = ap_mod.build_candidate(session, video, c, opts)
            if out:
                files.append(str(out))
            log.append(f"  done -> {out}")
            yield "\n".join(log), files
        except Exception as e:                   # noqa: BLE001 — contain per-candidate
            log.append(f"  failed ({type(e).__name__}: {e}); skipped.")
            yield "\n".join(log), gr.update()


# ---- tab layout ------------------------------------------------------------

def build_gameplay_tab() -> None:
    """Create the Gameplay tab. Must be called inside the app's gr.Blocks/gr.Tabs
    context. All components and handler wiring are local to this function."""
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
                transcribe_btn = gr.Button("① Transcribe / Re-transcribe",
                                           variant="primary")
                gr.Markdown("Runs WhisperX (the only ASR step) — **overwrites any "
                            "grid edits**. Build never re-runs ASR.")
                transcribe_status = gr.Textbox(label="Transcribe log", lines=4,
                                               interactive=False)

        # -- 2. transcript gate --
        editor_md = gr.Markdown("Transcribe a clip to populate the transcript editor.")
        transcript_df = gr.Dataframe(
            headers=Transcript.HEADERS, datatype=["str", "str", "number", "number"],
            type="array", interactive=True, label="Transcript (editable)",
            row_count=(1, "dynamic"))
        speaker_df = gr.Dataframe(
            headers=["speaker", "color (hex)"], datatype=["str", "str"],
            type="array", interactive=True, row_count=(1, "dynamic"),
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
                                label="Caption Y (higher = lower; 0.78 keeps it in "
                                      "the blur band, off the HUD)")
        with gr.Row():
            overlay_dd = gr.Dropdown(choices=[_NONE] + ov_mod.list_overlays(),
                                     value=_NONE, label="Like/subscribe overlay")
            overlay_pos_dd = gr.Dropdown(choices=ov_mod.POSITIONS,
                                         value=gconf.OVERLAY_DEFAULT_POSITION,
                                         label="Overlay position")
            overlay_start_n = gr.Number(value=gconf.OVERLAY_DEFAULT_START,
                                        label="Overlay start (s)")
            overlay_dur_n = gr.Number(value=gconf.OVERLAY_DEFAULT_DURATION,
                                      label="Overlay duration (s, 0 = whole clip)")
            refresh_ov_btn = gr.Button("↻ overlays")

        # -- 4. build --
        with gr.Row():
            build_btn = gr.Button("② Build from current transcript",
                                   variant="primary")
        build_status = gr.Textbox(label="Build log", lines=5, interactive=False)
        result_video = gr.Video(label="Result (9:16 Short)")

        # -- experimental full-auto: detect -> review -> load-into-manual --
        with gr.Accordion("⚠ Experimental — full-auto (long video → review → "
                          "load into manual)", open=False):
            gr.Markdown(
                "**Experimental.** Ingest a long (~1hr) video, auto-detect & "
                "categorise highlights (audio-energy + LLM over the diarized "
                "transcript). Review the candidates with previews, then **load a "
                "pick into manual mode above** to QC its transcript and build it "
                "(the human-in-the-loop path) — or batch-build selected with the "
                "current settings. Compute-heavy; transcribe needs your GPU. "
                "Detection is failure-contained.")
            auto_video = gr.Video(label="Long gameplay video")
            with gr.Row():
                auto_diarize_cb = gr.Checkbox(
                    value=True, label="Diarize (needs HF_TOKEN + accepted licence)")
                auto_backend_dd = gr.Dropdown(
                    choices=["ollama", "claude", "none"],
                    value=gconf.AUTO_LLM_BACKEND,
                    label="LLM backend for categorization")
                auto_maxclips = gr.Slider(1, 12, value=8, step=1,
                                          label="Max candidate clips")
                auto_detect_btn = gr.Button("① Detect highlights", variant="primary")
            auto_status = gr.Textbox(label="Full-auto log", lines=8,
                                     interactive=False)
            auto_gallery = gr.Gallery(label="Candidate previews", columns=4,
                                      height=240, object_fit="contain")
            auto_summary_df = gr.Dataframe(
                headers=["#", "category", "window", "score", "caption"],
                type="array", interactive=False, label="Candidates")
            auto_select_cbg = gr.CheckboxGroup(
                choices=[], label="Select candidates (by #)")
            with gr.Row():
                auto_load_btn = gr.Button("② Load selected into manual",
                                          variant="primary")
                auto_batch_btn = gr.Button("Batch-build selected (optional)")
            auto_files = gr.Files(label="Batch-built candidate Shorts")

        # -- wiring --
        transcribe_btn.click(_set_clip_name, clip_video, clip_state) \
            .then(_do_transcribe, [clip_video, diarize_cb, clip_state],
                  transcribe_status) \
            .then(_load_editor, clip_state,
                  [transcript_df, speaker_df, editor_md])

        # transcript bulk-edit wiring (each returns updated grid rows + a status)
        edit_assign_btn.click(_edit_assign,
                              [transcript_df, edit_span_tb, edit_speaker_tb],
                              [transcript_df, edit_status])
        edit_replace_btn.click(_edit_find_replace,
                               [transcript_df, edit_find_tb, edit_repl_tb,
                                edit_whole_cb],
                               [transcript_df, edit_status])
        edit_merge_btn.click(_edit_merge, [transcript_df, edit_span_tb],
                             [transcript_df, edit_status])
        edit_split_btn.click(_edit_split, [transcript_df, edit_split_row_n],
                             [transcript_df, edit_status])
        preview_caps_btn.click(
            _do_preview_captions,
            [clip_state, transcript_df, speaker_df, font_dd, posy_sl],
            preview_video)

        refresh_ov_btn.click(_refresh_overlays, None, overlay_dd)

        build_inputs = [clip_state, transcript_df, speaker_df, effects_cbg,
                        overlay_dd, overlay_pos_dd, overlay_start_n, overlay_dur_n,
                        font_dd, posy_sl]
        build_btn.click(_do_build, build_inputs, build_status) \
            .then(_show_result, clip_state, result_video)

        # full-auto: detect (streams) -> populate review gallery/table/selector
        auto_detect_btn.click(
            _do_detect,
            [auto_video, auto_backend_dd, auto_maxclips, auto_diarize_cb],
            auto_status) \
            .then(_load_candidates_ui, auto_video,
                  [auto_gallery, auto_summary_df, auto_select_cbg])

        # load the first selected candidate INTO the manual flow (pre-filled)
        auto_load_btn.click(
            _load_into_manual, [auto_video, auto_select_cbg],
            [clip_state, clip_video, transcript_df, speaker_df, editor_md])

        # optional: batch-build all selected with the current manual settings
        auto_batch_btn.click(
            _batch_build,
            [auto_video, auto_select_cbg, effects_cbg, overlay_dd, overlay_pos_dd,
             overlay_start_n, overlay_dur_n, font_dd, posy_sl, speaker_df],
            [auto_status, auto_files])
