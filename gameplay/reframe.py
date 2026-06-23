"""Reframe a 16:9 clip into a 1080x1920 (9:16) frame. Layout modes:

  - "fill" (default, recommended): scale to cover the frame, zoom by FILL_FRACTION,
    and crop to 1080x1920 with adjustable X/Y offset — the gameplay fills the whole
    frame at full resolution and most of the bitrate (sharpest). Loses the far
    horizontal edges; for ARAM the fight is centre and the minimap can be biased back
    with CROP_X_OFFSET.
  - "fit_crop": cover + centre-crop (== fill at fraction 1.0, centred). Kept as an alias.
  - "blur_pad": the full source frame centred over a blurred, scaled copy of itself —
    no crop, but a ~16:9 strip carries the action and most pixels are blur.
  - "zoom_blur": blur-pad but the centred gameplay band scaled up by ZOOM_BLUR_SCALE.

One ffmpeg graph per clip — no per-frame Python compositing. This is a CACHED
intermediate stage (reused by the caption preview), so it encodes near-lossless
(gameplay.encode.intermediate_args); the quality-governing encode is the final one.
"""
from __future__ import annotations

from pathlib import Path

from modules.assemble import _run, _has_audio
from gameplay import config as gconf
from gameplay import encode as enc

MODES = ("tall", "fill", "fit_crop", "blur_pad", "zoom_blur")


def _fill(w: int, h: int, frac: float, x_off: float, y_off: float, fps: int,
          out: str) -> str:
    # Cover the frame (min scale to fill both dims), zoom in by `frac` (>=1; 1.0 = just
    # cover), then crop W x H. x_off/y_off in 0..1 bias which slice survives
    # (0.5 = centre, 0 = left/top, 1 = right/bottom) — e.g. push the minimap back in.
    z = max(1.0, float(frac))
    x_off = min(1.0, max(0.0, float(x_off)))
    y_off = min(1.0, max(0.0, float(y_off)))
    return (
        f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"scale=ceil(iw*{z}/2)*2:ceil(ih*{z}/2)*2,"
        f"crop={w}:{h}:(iw-{w})*{x_off}:(ih-{h})*{y_off},"
        f"setsar=1,fps={fps},format=yuv420p[{out}]"
    )


def _tall(w: int, h: int, frac: float, blur: int, x_off: float, y_off: float,
          fps: int, out: str) -> str:
    # A FULL-WIDTH gameplay band that fills `frac` of the height (uniform cover-crop, no
    # stretch), centred over a blurred full-frame fill that shows through as a thin frame
    # top/bottom. More vertical than blur_pad, more horizontal context than full-crop fill.
    frac = min(1.0, max(0.4, float(frac)))
    bh = max(2, (int(round(h * frac)) // 2) * 2)        # band height (even)
    x_off = min(1.0, max(0.0, float(x_off)))
    y_off = min(1.0, max(0.0, float(y_off)))
    return (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},"
        f"boxblur={blur}:1,setsar=1[bgb];"
        f"[fg]scale={w}:{bh}:force_original_aspect_ratio=increase,"
        f"crop={w}:{bh}:(iw-{w})*{x_off}:(ih-{bh})*{y_off},setsar=1[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2:format=auto,fps={fps},format=yuv420p[{out}]"
    )


def _blur_pad(w: int, h: int, blur: int, fps: int, out: str) -> str:
    return (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},boxblur={blur}:1,setsar=1[bgb];"
        f"[fg]scale={w}:{h}:force_original_aspect_ratio=decrease,setsar=1[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2:format=auto,"
        f"fps={fps},format=yuv420p[{out}]"
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
                   fill_frac: float | None = None, x_off: float | None = None,
                   y_off: float | None = None, tall_frac: float | None = None,
                   out_label: str = "v") -> str:
    """The filter_complex graph (ending in [out_label]) for the given layout mode.
    Unknown modes fall back to blur_pad. Pure string builder — unit-testable."""
    blur = gconf.BLUR_RADIUS if blur is None else blur
    zoom = gconf.ZOOM_BLUR_SCALE if zoom is None else zoom
    fps = gconf.FPS if fps is None else fps
    fill_frac = gconf.REFRAME_FILL_FRACTION if fill_frac is None else fill_frac
    x_off = gconf.REFRAME_CROP_X_OFFSET if x_off is None else x_off
    y_off = gconf.REFRAME_CROP_Y_OFFSET if y_off is None else y_off
    tall_frac = gconf.REFRAME_TALL_HEIGHT_FRAC if tall_frac is None else tall_frac
    if mode == "tall":
        return _tall(w, h, tall_frac, blur, x_off, y_off, fps, out_label)
    if mode == "fill":
        return _fill(w, h, fill_frac, x_off, y_off, fps, out_label)
    if mode == "fit_crop":           # cover + centre-crop == fill @ 1.0, centred
        return _fill(w, h, 1.0, 0.5, 0.5, fps, out_label)
    if mode == "zoom_blur":
        return _zoom_blur(w, h, blur, zoom, fps, out_label)
    return _blur_pad(w, h, blur, fps, out_label)


def blur_pad_filter(w: int, h: int, blur: int, fps: int,
                    out_label: str = "v") -> str:
    """Back-compat shim — the blur_pad graph."""
    return _blur_pad(w, h, blur, fps, out_label)


def reframe(src: str | Path, out: str | Path, mode: str | None = None,
            x_off: float | None = None, y_off: float | None = None,
            fill_frac: float | None = None) -> Path:
    """Reframe `src` to 1080x1920 at config.FPS using the chosen layout `mode`
    (defaults to config.REFRAME_MODE), preserving the source audio (if any). For the
    `fill` layout, `x_off`/`y_off`/`fill_frac` tune the crop (default from config).
    Idempotent: skips if `out` already exists. Near-lossless intermediate encode."""
    src, out = Path(src), Path(out)
    if out.exists():
        return out
    w, h, fps = gconf.WIDTH, gconf.HEIGHT, gconf.FPS
    mode = mode or gconf.REFRAME_MODE
    graph = reframe_filter(mode, w, h, fps=fps, x_off=x_off, y_off=y_off,
                           fill_frac=fill_frac)
    cmd = ["ffmpeg", "-y", "-i", str(src), "-filter_complex", graph, "-map", "[v]"]
    if _has_audio(src):
        cmd += ["-map", "0:a", "-c:a", "aac", "-b:a", "192k"]
    cmd += [*enc.intermediate_args(), str(out)]
    _run(cmd)
    return out
