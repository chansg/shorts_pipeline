"""Reframe a clip to 9:16 by blur-pad: the full source frame is centred, and the
top/bottom are filled with a blurred, scaled copy of the same frame (no black
bars, no cropping the action).

One ffmpeg graph — no per-frame Python compositing. Reuses the lore pipeline's
ffmpeg runner (`modules.assemble._run`) and probe so behaviour matches.
"""
from __future__ import annotations

from pathlib import Path

from modules.assemble import _run, _has_audio
from gameplay import config as gconf


def blur_pad_filter(w: int, h: int, blur: int, fps: int,
                    out_label: str = "v") -> str:
    """The split/scale/boxblur/overlay graph that turns one video stream into a
    centred frame over a blurred fill. Returns a filter_complex chain ending in
    [out_label]."""
    return (
        f"[0:v]split=2[bg][fg];"
        # background: cover the whole frame, then blur it
        f"[bg]scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},boxblur={blur}:1,setsar=1[bgb];"
        # foreground: the full source frame, fit inside the frame (no crop)
        f"[fg]scale={w}:{h}:force_original_aspect_ratio=decrease,setsar=1[fgs];"
        # centre the foreground over the blurred background
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2:format=auto,"
        f"fps={fps},format=yuv420p[{out_label}]"
    )


def reframe(src: str | Path, out: str | Path) -> Path:
    """Blur-pad `src` to a 1080x1920 clip at config.FPS, preserving the source
    audio (if any). Idempotent: skips if `out` already exists."""
    src, out = Path(src), Path(out)
    if out.exists():
        return out
    w, h, fps = gconf.WIDTH, gconf.HEIGHT, gconf.FPS
    graph = blur_pad_filter(w, h, gconf.BLUR_RADIUS, fps)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-filter_complex", graph, "-map", "[v]",
    ]
    if _has_audio(src):
        cmd += ["-map", "0:a", "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
            "-preset", "medium", "-crf", "20", str(out)]
    _run(cmd)
    return out
