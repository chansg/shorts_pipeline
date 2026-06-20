"""The Full-Auto Experiment view for the unified GUI — a self-contained module so
app.py only needs an import + one call inside the `fullauto` container's gr.Tabs().

Flow: drop in a long video -> Detect & categorise highlights (review gallery) ->
Build the chosen highlights into ONE 16:9 YouTube video. It does NOT touch the 9:16
Shorts backend (no reframe/overlay/karaoke) — full-auto exports landscape.

Streaming handlers update only their status Textbox while running (yielding
gr.update() for other outputs), mirroring how the rest of the GUI streams.
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr

from orchestrator.errors import FriendlyError, friendly
from fullauto import pipeline as ap
from fullauto import clips as clip_mod
from gameplay import config as gconf
from gameplay.state import AutoSession


# ---- helpers ---------------------------------------------------------------

def _cand_label(i, c) -> str:
    return (f"{i+1}. [{c.category}] {c.start:.0f}-{c.end:.0f}s · "
            f"{c.score:.2f} · {c.caption or '(no caption)'}")


def _selected_indices(selected) -> list[int]:
    out = []
    for label in selected or []:
        try:
            out.append(int(str(label).split(".", 1)[0]) - 1)
        except (ValueError, IndexError):
            continue
    return sorted(set(out))


# ---- handlers --------------------------------------------------------------

def _do_detect(video, backend, max_clips, diarize):
    """Detect + rank candidates for a long video (no building). Streams the staged
    progress, persists candidates + thumbnails for the review step."""
    if not video:
        raise gr.Error("Upload a long video first.")
    captured: list[str] = []
    try:
        ap.detect_candidates(
            video, backend=backend, max_clips=int(max_clips), diarize=bool(diarize),
            progress=lambda m: captured.append(m))
    except FriendlyError as fe:
        raise gr.Error(str(fe), duration=None)
    except Exception as e:                       # noqa: BLE001
        raise gr.Error(str(friendly(e)), duration=None)
    yield "\n".join(captured)


def _load_candidates_ui(video):
    """Populate the review gallery / table / selector from the persisted session."""
    if not video:
        return [], [], gr.update(choices=[], value=[])
    session = AutoSession(Path(video).stem)
    cands = ap.load_candidates(session)
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


def _build_youtube(video, selected):
    """Assemble the selected candidates into ONE 16:9 YouTube video (native res)."""
    if not video:
        raise gr.Error("Run Detect first.")
    idxs = _selected_indices(selected)
    if not idxs:
        raise gr.Error("Tick at least one candidate to build.")
    session = AutoSession(Path(video).stem)
    cands = ap.load_candidates(session)
    chosen = [cands[i] for i in idxs if 0 <= i < len(cands)]
    log: list[str] = [f"Building a 16:9 YouTube video from {len(chosen)} highlight(s)..."]
    yield "\n".join(log), gr.update()
    try:
        out = ap.build_youtube(session, video, chosen, progress=log.append)
        log.append(f"Done -> {out}")
        yield "\n".join(log), str(out)
    except FriendlyError as fe:
        raise gr.Error(str(fe), duration=None)
    except Exception as e:                       # noqa: BLE001
        raise gr.Error(str(friendly(e)), duration=None)


# ---- highlight-clips flow (audio-reaction + HUD -> 9:16 raw candidates) -----

def _clip_label(c) -> str:
    hud = ("+".join(c.hud_events) if c.hud_events else "no HUD")
    return f"{c.rank} · {c.start:.0f}-{c.end:.0f}s · score {c.score:.2f} · {hud}"


def _do_detect_clips(video, threshold, pre_roll, post_roll, max_candidates, hud_on):
    """Run audio-reaction (+ optional HUD) detection over a long video and export
    ranked 9:16 raw candidates. Streams the staged log; persists candidates.json."""
    if not video:
        raise gr.Error("Upload a long video first.")
    # apply the per-run knobs (calibration surface) before detecting
    gconf.REACTION_THRESHOLD = float(threshold)
    gconf.PRE_ROLL_S = float(pre_roll)
    gconf.POST_ROLL_S = float(post_roll)
    captured: list[str] = []
    try:
        clip_mod.run_highlight_detection(
            video, hud_enabled=bool(hud_on), max_candidates=int(max_candidates),
            progress=captured.append)
    except FriendlyError as fe:
        raise gr.Error(str(fe), duration=None)
    except Exception as e:                       # noqa: BLE001
        raise gr.Error(str(friendly(e)), duration=None)
    yield "\n".join(captured)


def _load_clips_ui(video):
    """Populate the gallery / table / refine selector from the persisted manifest."""
    empty = gr.update(choices=[], value=None)
    if not video:
        return [], [], empty
    session = AutoSession(Path(video).stem)
    cands = clip_mod.load_manifest(session)
    gallery, rows, labels = [], [], []
    for c in cands:
        label = _clip_label(c)
        labels.append(label)
        rows.append([c.rank, f"{c.start:.0f}-{c.end:.0f}s", c.duration,
                     round(c.score, 2), round(c.audio_score, 2),
                     "+".join(c.hud_events) or "—", c.why])
        if c.preview_path and Path(c.preview_path).exists():
            gallery.append((c.preview_path, label))
    return gallery, rows, gr.update(choices=labels, value=(labels[0] if labels else None))


def _refine_source_for(video, label):
    """Resolve the chosen candidate label to its source clip path (the manual input)."""
    if not video or not label:
        return None
    session = AutoSession(Path(video).stem)
    for c in clip_mod.load_manifest(session):
        if _clip_label(c) == label:
            return c.source_path or c.clip_path or None
    return None


# ---- view layout -----------------------------------------------------------

def build_fullauto_view(manual_clip_video=None) -> dict:
    """Create the Full-Auto container's tabs. Returns the hand-off handles
    {refine_btn, refine_video, refine_dd} so app.py can wire 'Refine in manual mode'
    into the Gameplay uploader (`manual_clip_video`) and route to that tab."""
    handles: dict = {}
    with gr.Tab("🎯 Highlight clips → manual"):
        gr.Markdown(
            "### Long video → ranked 9:16 candidate clips\n"
            "Drop in a lengthy gameplay VOD (hours of League/ARAM). Detection finds "
            "**funny / rage** moments from **voice reactions** (a sudden “WHAT?!”), "
            "with League **HUD events** (multikills / aces) as a score booster, and "
            "exports each as a **generous raw 9:16 clip** to refine in the Gameplay "
            "tab. No transcript/LLM needed — robust and fast. Thresholds are a "
            "calibration surface; the log shows the score curve.")
        hl_video = gr.Video(label="Long video (gameplay VOD / NVIDIA capture)")
        with gr.Row():
            hl_threshold = gr.Slider(0.05, 0.9, value=gconf.REACTION_THRESHOLD,
                                     step=0.05, label="Reaction threshold (lower = more)")
            hl_pre = gr.Slider(2, 20, value=gconf.PRE_ROLL_S, step=1,
                               label="Pre-roll (setup, s)")
            hl_post = gr.Slider(2, 25, value=gconf.POST_ROLL_S, step=1,
                                label="Post-roll (payoff, s)")
            hl_max = gr.Slider(3, 40, value=gconf.MAX_CANDIDATES, step=1,
                               label="Max candidates")
            hl_hud = gr.Checkbox(value=gconf.HUD_SCAN_ENABLED,
                                 label="HUD scan (booster; safe to fail)")
        hl_detect_btn = gr.Button("① Detect highlight clips", variant="primary")
        hl_status = gr.Textbox(label="Detection log (score curve + ranking)",
                               lines=10, interactive=False)
        hl_gallery = gr.Gallery(label="Candidate previews", columns=4, height=240,
                                object_fit="contain")
        hl_df = gr.Dataframe(
            headers=["#", "window", "dur", "score", "audio", "HUD", "why"],
            type="array", interactive=False, label="Ranked candidates")
        with gr.Row():
            hl_refine_dd = gr.Dropdown(choices=[], label="Pick a candidate to refine")
            hl_refine_btn = gr.Button("② Refine in manual mode →", variant="primary")
        hl_refine_video = gr.Video(label="Selected candidate (raw 9:16 preview)")

        hl_detect_btn.click(
            _do_detect_clips,
            [hl_video, hl_threshold, hl_pre, hl_post, hl_max, hl_hud],
            hl_status) \
            .then(_load_clips_ui, hl_video, [hl_gallery, hl_df, hl_refine_dd])
        # preview the chosen candidate's 9:16 clip when the selector changes
        hl_refine_dd.change(
            lambda v, l: (_refine_source_for(v, l) or None), [hl_video, hl_refine_dd],
            hl_refine_video)
        handles = {"refine_btn": hl_refine_btn, "video": hl_video, "dd": hl_refine_dd,
                   "manual_target": manual_clip_video}

    with gr.Tab("⚗ Full-Auto"):
        gr.Markdown(
            "### Long video → 16:9 YouTube highlights\n"
            "**Experimental.** Drop in raw YouTube footage or a long gameplay VOD; "
            "auto-detect & categorise highlights (audio-energy + reaction keywords + "
            "an LLM judge over the diarized transcript), review the candidates with "
            "previews, then assemble the picks into one **16:9 YouTube video** "
            "(full resolution — not a 9:16 Short). Compute-heavy; transcription needs "
            "your GPU. Detection is failure-contained.")

        auto_video = gr.Video(label="Long video (raw YouTube / gameplay VOD)")
        with gr.Row():
            auto_diarize_cb = gr.Checkbox(
                value=True, label="Diarize (needs HF_TOKEN + accepted licence)")
            auto_backend_dd = gr.Dropdown(
                choices=["ollama", "claude", "none"],
                value=gconf.AUTO_LLM_BACKEND,
                label="LLM backend for categorization")
            auto_maxclips = gr.Slider(1, 12, value=8, step=1,
                                      label="Max candidate highlights")
            auto_detect_btn = gr.Button("① Detect highlights", variant="primary")
        auto_status = gr.Textbox(label="Full-auto log", lines=8, interactive=False)
        auto_gallery = gr.Gallery(label="Candidate previews", columns=4,
                                  height=240, object_fit="contain")
        auto_summary_df = gr.Dataframe(
            headers=["#", "category", "window", "score", "caption"],
            type="array", interactive=False, label="Candidates")
        auto_select_cbg = gr.CheckboxGroup(
            choices=[], label="Select highlights (by #)")
        with gr.Row():
            auto_build_btn = gr.Button("② Build selected → 16:9 YouTube video",
                                       variant="primary")
        build_status = gr.Textbox(label="Build log", lines=4, interactive=False)
        result_video = gr.Video(label="Result (16:9 YouTube video)")

        # -- wiring --
        # detect (streams) -> populate review gallery/table/selector
        auto_detect_btn.click(
            _do_detect,
            [auto_video, auto_backend_dd, auto_maxclips, auto_diarize_cb],
            auto_status) \
            .then(_load_candidates_ui, auto_video,
                  [auto_gallery, auto_summary_df, auto_select_cbg])

        # build the selected highlights into one 16:9 YouTube video
        auto_build_btn.click(_build_youtube, [auto_video, auto_select_cbg],
                             [build_status, result_video])

    return handles
