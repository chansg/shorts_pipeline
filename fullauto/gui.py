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


# ---- view layout -----------------------------------------------------------

def build_fullauto_view() -> None:
    """Create the Full-Auto tab. Must be called inside the app's gr.Blocks/gr.Tabs
    context. All components and handler wiring are local to this function."""
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
