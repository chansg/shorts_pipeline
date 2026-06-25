"""The Full-Auto Experiment view for the unified GUI — a self-contained module so
app.py only needs an import + one call inside the `fullauto` container's gr.Tabs().

Flow: drop in a long video -> Detect & categorise highlights (review gallery) ->
Build the chosen highlights into ONE 16:9 YouTube video. It does NOT touch the 9:16
Shorts backend (no reframe/overlay/karaoke) — full-auto exports landscape.

Streaming handlers update only their status Textbox while running (yielding
gr.update() for other outputs), mirroring how the rest of the GUI streams.
"""
from __future__ import annotations

import os
import queue
import threading
from pathlib import Path

import gradio as gr

from orchestrator.errors import FriendlyError, friendly
from fullauto import pipeline as ap
from fullauto import clips as clip_mod
from fullauto import candidates as cand_mod
from gameplay import config as gconf
from gameplay.state import AutoSession

_CAND_DONE = object()        # sentinel: worker thread finished streaming progress


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


def _mode_key(mode) -> str:
    return "aram" if str(mode).lower().startswith("aram") else "generic"


def _do_detect_clips(video, mode, threshold, pre_roll, post_roll, max_candidates, hud_on):
    """Run detection over a long video and export ranked 9:16 raw candidates. Streams
    the staged log; persists candidates.json. `mode` switches generic audio-reaction vs
    ARAM multikill-led."""
    if not video:
        raise gr.Error("Upload a long video first.")
    # apply the per-run knobs (calibration surface) before detecting
    gconf.REACTION_THRESHOLD = float(threshold)
    gconf.PRE_ROLL_S = float(pre_roll)
    gconf.POST_ROLL_S = float(post_roll)
    captured: list[str] = []
    try:
        clip_mod.run_highlight_detection(
            video, mode=_mode_key(mode), hud_enabled=bool(hud_on),
            max_candidates=int(max_candidates), progress=captured.append)
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


# ---- candidate export (batch -> raw 60-90s trims, both audio tracks) --------

def _resolve_cand_inputs(files, paths_text) -> list[Path]:
    """Combine the hand-picked files and the folder/paths box into source videos.
    Reuses candidates.collect_inputs (folder-glob + ext-filter + de-dup)."""
    items = list(files or [])
    for line in (paths_text or "").splitlines():
        line = line.strip().strip('"')
        if line:
            items.append(line)
    return cand_mod.collect_inputs(items)


def _cand_summary_rows(results) -> list[list]:
    """Per-source summary table from the returned candidates.json manifests: source (+
    count), then each candidate's rank / category / peak (score) / one-line why."""
    rows: list[list] = []
    for out_dir, manifest in results or []:
        src = manifest.get("source") or Path(str(out_dir)).name
        cands = manifest.get("candidates", [])
        if not cands:
            rows.append([f"{src} (0)", "-", "-", "-",
                         manifest.get("note", "no candidates")])
            continue
        for j, c in enumerate(cands):
            m, s = divmod(int(round(c.get("peak_s", 0) or 0)), 60)
            rows.append([f"{src} ({len(cands)})" if j == 0 else "",
                         str(c.get("rank", "")), c.get("category", ""),
                         f"{m}m{s:02d}s ({c.get('score', '')})", c.get("why", "")])
    return rows


def _do_candidate_export(files, paths_text, out_dir):
    """Run the candidate-export batch on a WORKER THREAD, streaming per-source/stage/clip
    progress into the log (live). Calls fullauto.candidates.run_batch — the SAME callable
    the CLI main() calls — with a progress callback piped through a queue, so the UI thread
    only yields widget updates and never blocks. One bad source is logged by run_batch and
    the batch continues. Outputs: (log, summary_df, run_button)."""
    inputs = _resolve_cand_inputs(files, paths_text)
    out_dir = (out_dir or "").strip() or str(gconf.CAND_OUTPUT_DIR)
    if not inputs:
        yield ("Pick one or more MP4 files, or enter a folder / file paths above, "
               "then Run.", gr.update(), gr.update(interactive=True))
        return

    q: "queue.Queue" = queue.Queue()
    result: dict = {}

    def worker():
        try:
            result["res"] = cand_mod.run_batch(inputs, out_dir, progress=q.put)
        except Exception as e:                    # noqa: BLE001 — surface, never hang
            q.put(f"FAILED: {type(e).__name__}: {e}")
            result["res"] = []
        finally:
            q.put(_CAND_DONE)

    threading.Thread(target=worker, daemon=True).start()
    log = [f"Processing {len(inputs)} source(s) -> {out_dir}"]
    yield "\n".join(log), gr.update(), gr.update(interactive=False)   # disable Run
    while True:
        msg = q.get()
        if msg is _CAND_DONE:
            break
        log.append(str(msg))
        yield "\n".join(log), gr.update(), gr.update(interactive=False)

    rows = _cand_summary_rows(result.get("res", []))
    log.append(f"Done - {len(result.get('res', []))} source(s) processed.")
    yield "\n".join(log), rows, gr.update(interactive=True)           # re-enable Run


def _open_cand_folder(out_dir) -> str:
    """Open the output folder in the OS file browser (the app runs locally). Fail-safe."""
    path = (out_dir or "").strip() or str(gconf.CAND_OUTPUT_DIR)
    try:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        opener = getattr(os, "startfile", None)
        if opener is not None:                    # Windows
            opener(str(p))
        else:                                     # mac/linux fallback
            import subprocess
            import sys
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(p)])
        return f"Opened `{p}`"
    except Exception as e:                        # noqa: BLE001
        return f"Could not open `{path}`: {type(e).__name__}"


# ---- view layout -----------------------------------------------------------

def build_fullauto_view(manual_clip_video=None) -> dict:
    """Create the Full-Auto container's tabs. Returns the hand-off handles
    {refine_btn, refine_video, refine_dd} so app.py can wire 'Refine in manual mode'
    into the Gameplay uploader (`manual_clip_video`) and route to that tab."""
    handles: dict = {}
    with gr.Tab("🎯 Highlight clips → manual"):
        gr.Markdown(
            "### Long video → ranked 9:16 candidate clips\n"
            "Drop in a lengthy gameplay VOD. **Generic** finds **funny / rage** moments "
            "from **voice reactions** (a sudden “WHAT?!”) with HUD events as a booster. "
            "**ARAM (League)** instead hunts **multikills**: it scans the whole clip for "
            "the centre banner, collapses each fight's escalating banners to its top "
            "tier, and surfaces every **triple / quadra / penta (+ ace)** as a candidate "
            "(the voice reaction breaks ties). Each is exported as a **generous raw 9:16 "
            "clip** to refine in the Gameplay tab. No transcript/LLM needed.")
        hl_video = gr.Video(label="Long video (gameplay VOD / NVIDIA capture)")
        with gr.Row():
            hl_mode = gr.Dropdown(
                choices=["Generic (audio reactions)", "ARAM (League multikills)"],
                value=("ARAM (League multikills)" if gconf.GAME_MODE == "aram"
                       else "Generic (audio reactions)"),
                label="Game mode")
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
            [hl_video, hl_mode, hl_threshold, hl_pre, hl_post, hl_max, hl_hud],
            hl_status) \
            .then(_load_clips_ui, hl_video, [hl_gallery, hl_df, hl_refine_dd])
        # preview the chosen candidate's 9:16 clip when the selector changes
        hl_refine_dd.change(
            lambda v, l: (_refine_source_for(v, l) or None), [hl_video, hl_refine_dd],
            hl_refine_video)
        handles = {"refine_btn": hl_refine_btn, "video": hl_video, "dd": hl_refine_dd,
                   "manual_target": manual_clip_video}

    with gr.Tab("📁 Candidate export (batch)"):
        gr.Markdown(
            "### Batch recordings → top-5 raw 60-90s trims per source\n"
            "Hand-pick recordings **or** point at a folder; each source's strongest "
            "moments are exported as **raw clips** (no reframe/captions/overlay) that "
            "**keep both audio tracks**, alongside a `candidates.json`. Detection is "
            "voice-energy + HUD-OCR: loud squad reactions become **banter**, kill banners "
            "(Pentakill / Ace / …) become **play**. Single-track sources and a missing "
            "Tesseract degrade with a warning in the log — the run still completes.")
        cand_files = gr.File(
            label="Recordings — pick multiple MP4s",
            file_count="multiple", file_types=[".mp4", ".mkv", ".mov", ".m4v"],
            type="filepath")
        cand_paths = gr.Textbox(
            label="…or a folder / file paths (one per line) — process every MP4 in a folder",
            value=str(gconf.CAND_INPUT_DIR), lines=2,
            info="Leave the file picker empty to process this whole folder.")
        cand_out = gr.Textbox(label="Output folder", value=str(gconf.CAND_OUTPUT_DIR))
        cand_run_btn = gr.Button("Run candidate export", variant="primary")
        cand_log = gr.Textbox(label="Progress log (per source / stage / clip)",
                              lines=12, interactive=False)
        cand_summary = gr.Dataframe(
            headers=["Source (count)", "Rank", "Category", "Peak (score)", "Why"],
            type="array", interactive=False, label="Results per source")
        with gr.Row():
            cand_open_btn = gr.Button("Open output folder")
            cand_open_status = gr.Markdown()

        cand_run_btn.click(
            _do_candidate_export, [cand_files, cand_paths, cand_out],
            [cand_log, cand_summary, cand_run_btn])
        cand_open_btn.click(_open_cand_folder, cand_out, cand_open_status)

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
