"""Music-montage builder tab for the Shorts view — pick gameplay clips (in play order)
+ one MP3, set where in the song to start, hit Run. Mirrors the candidate-export GUI:
the work runs on a WORKER THREAD feeding a queue that the Gradio generator drains and
yields from, so progress streams live and the UI never blocks. Calls gameplay.montage.
build_montage (which reuses the gameplay reframe + overlay components)."""
from __future__ import annotations

import os
import queue
import threading
from pathlib import Path

import gradio as gr

from gameplay import config as gconf
from gameplay import montage as montage_mod

_MONTAGE_DONE = object()


# ---- helpers ---------------------------------------------------------------

def _clip_lines(text) -> list[str]:
    """Ordered clip paths from the textbox (one per line; play order = top to bottom).
    Unquotes pasted 'Copy as path' values so surrounding quotes don't break the path."""
    return [p for p in (montage_mod._unquote(ln) for ln in (text or "").splitlines()) if p]


def _append_clips(picked, current):
    """Append newly-picked clip paths to the ordered list (preserving existing order)."""
    lines = _clip_lines(current)
    for p in picked or []:
        if p and p not in lines:
            lines.append(str(p))
    return "\n".join(lines)


def _set_music(picked):
    return str(picked) if picked else ""


def _open_folder(out_dir) -> str:
    path = (out_dir or "").strip() or str(gconf.MONTAGE_OUTPUT_DIR)
    try:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        opener = getattr(os, "startfile", None)
        if opener is not None:                       # Windows
            opener(str(p))
        else:                                        # mac/linux fallback
            import subprocess
            import sys
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(p)])
        return f"Opened `{p}`"
    except Exception as e:                           # noqa: BLE001
        return f"Could not open `{path}`: {type(e).__name__}"


# ---- threaded run handler --------------------------------------------------

def _do_montage(clips_text, music, music_start, out_dir):
    """Build the montage on a worker thread, streaming progress. Outputs:
    (log, result_video, run_button)."""
    clips = _clip_lines(clips_text)
    music = montage_mod._unquote(music)
    out_dir = montage_mod._unquote(out_dir) or str(gconf.MONTAGE_OUTPUT_DIR)
    if not clips:
        yield ("Add at least one gameplay clip (pick files or paste paths), then Run.",
               gr.update(), gr.update(interactive=True))
        return
    if not music:
        yield ("Pick one music file (MP3) for the montage, then Run.",
               gr.update(), gr.update(interactive=True))
        return

    out_path = Path(out_dir) / montage_mod._out_name(Path(str(music)))
    q: "queue.Queue" = queue.Queue()
    result: dict = {}

    def worker():
        try:
            result["out"] = montage_mod.build_montage(
                clips, music, music_start, out_path=out_path, progress=q.put)
        except Exception as e:                       # noqa: BLE001 — surface, never hang
            q.put(f"ERROR: {type(e).__name__}: {e}")
            result["out"] = None
        finally:
            q.put(_MONTAGE_DONE)

    threading.Thread(target=worker, daemon=True).start()
    log = [f"Building montage from {len(clips)} clip(s) -> {out_dir}"]
    yield "\n".join(log), gr.update(), gr.update(interactive=False)   # disable Run
    while True:
        msg = q.get()
        if msg is _MONTAGE_DONE:
            break
        log.append(str(msg))
        yield "\n".join(log), gr.update(), gr.update(interactive=False)

    out = result.get("out")
    yield ("\n".join(log), (str(out) if out else gr.update()),
           gr.update(interactive=True))             # re-enable Run


# ---- view ------------------------------------------------------------------

def build_montage_tab():
    """The '🎬 Music Montage' tab. Self-contained; mounted inside the Shorts view."""
    gr.Markdown(
        "### Gameplay clips + a music track → one 9:16 montage Short\n"
        "Pick several gameplay MP4s (they play in the order listed — reorder/remove by "
        "editing the list) and **one** royalty-free MP3. Clips are reframed to 9:16 and "
        "stitched with light crossfades; the **game audio is denoised and ducked to ~20% "
        "under the music**, which starts at the offset you set (skip to the drop); the "
        "**GamerChans overlay** is applied. Reframe + overlay are the gameplay pipeline's "
        "own components.")
    m_clips = gr.Textbox(
        label="Gameplay clips — one path per line, in play order",
        lines=4, placeholder="C:\\Users\\chansg\\Videos\\clip1.mp4")
    m_clip_picker = gr.File(
        label="…or pick clips to add (appended to the list above)",
        file_count="multiple", file_types=[".mp4", ".mkv", ".mov", ".m4v"],
        type="filepath")
    with gr.Row():
        m_music = gr.Textbox(label="Music track (one MP3)", scale=3,
                             placeholder="C:\\Users\\chansg\\Music\\track.mp3")
        m_start = gr.Textbox(label="Music start (mm:ss or seconds)", value="0:00", scale=1)
    m_music_picker = gr.File(label="…or pick the MP3", file_count="single",
                             file_types=[".mp3", ".wav", ".m4a", ".aac", ".ogg"],
                             type="filepath")
    m_out = gr.Textbox(label="Output folder", value=str(gconf.MONTAGE_OUTPUT_DIR))
    m_run = gr.Button("Build montage", variant="primary")
    m_log = gr.Textbox(label="Progress log", lines=10, interactive=False)
    m_result = gr.Video(label="Montage (9:16)")
    with gr.Row():
        m_open = gr.Button("Open output folder")
        m_open_status = gr.Markdown()

    # pickers feed the ordered list / music box (so order is explicit + editable)
    m_clip_picker.change(_append_clips, [m_clip_picker, m_clips], m_clips)
    m_music_picker.change(_set_music, m_music_picker, m_music)
    m_run.click(_do_montage, [m_clips, m_music, m_start, m_out],
                [m_log, m_result, m_run])
    m_open.click(_open_folder, m_out, m_open_status)
