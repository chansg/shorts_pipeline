"""Reframe a 16:9 clip into a 1080x1920 (9:16) frame. Three layout modes:

  - "blur_pad" (default): the full source frame centred over a blurred, scaled copy
    of itself — no black bars, no cropping, but a ~16:9 strip carries the action and
    most pixels are blur.
  - "fit_crop": scale to FILL the frame and crop the sides — the gameplay fills the
    whole 1080x1920 at full resolution (sharpest; loses the horizontal edges).
  - "zoom_blur": blur-pad but the centred gameplay band scaled up by ZOOM_BLUR_SCALE
    (bigger band, smaller blur bars, more bitrate on the gameplay).

One ffmpeg graph per clip — no per-frame Python compositing. This is a CACHED
intermediate stage (reused by the caption preview), so it encodes near-lossless
(gameplay.encode.intermediate_args); the quality-governing encode is the final one.
"""
from __future__ import annotations

from pathlib import Path

from modules.assemble import _run, _has_audio
from gameplay import config as gconf
from gameplay import encode as enc

MODES = ("blur_pad", "fit_crop", "zoom_blur")


def _blur_pad(w: int, h: int, blur: int, fps: int, out: str) -> str:
    return (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},boxblur={blur}:1,setsar=1[bgb];"
        f"[fg]scale={w}:{h}:force_original_aspect_ratio=decrease,setsar=1[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2:format=auto,"
        f"fps={fps},format=yuv420p[{out}]"
    )


def _fit_crop(w: int, h: int, fps: int, out: str) -> str:
    # Scale to COVER the frame (increase), then crop the overflow (the sides for a
    # 16:9 source) so the gameplay fills 1080x1920 at full resolution.
    return (
        f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},setsar=1,fps={fps},format=yuv420p[{out}]"
    )


def _zoom_blur(w: int, h: int, blur: int, zoom: float, fps: int, out: str) -> str:
    # Like blur_pad, but enlarge the fitted foreground by `zoom` and crop it back to
    # at most the frame size — a bigger gameplay band over thinner blur bars.
    return (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},boxblur={blur}:1,setsar=1[bgb];"
        f"[fg]scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"scale=ceil(iw*{zoom}/2)*2:ceil(ih*{zoom}/2)*2,"
        f"crop=min(iw\\,{w}):min(ih\\,{h}):(iw-min(iw\\,{w}))/2:(ih-min(ih\\,{h}))/2,"
        f"setsar=1[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2:format=auto,"
        f"fps={fps},format=yuv420p[{out}]"
    )


def reframe_filter(mode: str | None, w: int, h: int, *, blur: int | None = None,
                   zoom: float | None = None, fps: int | None = None,
                   out_label: str = "v") -> str:
    """The filter_complex graph (ending in [out_label]) for the given layout mode.
    Unknown modes fall back to blur_pad. Pure string builder — unit-testable."""
    blur = gconf.BLUR_RADIUS if blur is None else blur
    zoom = gconf.ZOOM_BLUR_SCALE if zoom is None else zoom
    fps = gconf.FPS if fps is None else fps
    if mode == "fit_crop":
        return _fit_crop(w, h, fps, out_label)
    if mode == "zoom_blur":
        return _zoom_blur(w, h, blur, zoom, fps, out_label)
    return _blur_pad(w, h, blur, fps, out_label)


def blur_pad_filter(w: int, h: int, blur: int, fps: int,
                    out_label: str = "v") -> str:
    """Back-compat shim — the blur_pad graph."""
    return _blur_pad(w, h, blur, fps, out_label)


def reframe(src: str | Path, out: str | Path, mode: str | None = None) -> Path:
    """Reframe `src` to 1080x1920 at config.FPS using the chosen layout `mode`
    (defaults to config.REFRAME_MODE), preserving the source audio (if any).
    Idempotent: skips if `out` already exists. Near-lossless intermediate encode."""
    src, out = Path(src), Path(out)
    if out.exists():
        return out
    w, h, fps = gconf.WIDTH, gconf.HEIGHT, gconf.FPS
    mode = mode or gconf.REFRAME_MODE
    graph = reframe_filter(mode, w, h, fps=fps)
    cmd = ["ffmpeg", "-y", "-i", str(src), "-filter_complex", graph, "-map", "[v]"]
    if _has_audio(src):
        cmd += ["-map", "0:a", "-c:a", "aac", "-b:a", "192k"]
    cmd += [*enc.intermediate_args(), str(out)]
    _run(cmd)
    return out
