"""Composite a transparent overlay asset (like/subscribe animation) onto a clip.

Supports alpha video (.mov/.webm/.mkv with an alpha pixel format) and transparent
images (.png). Uses ffmpeg's `overlay` filter, respecting alpha, with an `enable`
window so the overlay shows for a chosen start-time/duration. No per-frame Python
compositing.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from modules.assemble import _run, _has_audio
from gameplay import config as gconf
from orchestrator.errors import FriendlyError

_VIDEO_EXTS = {".mov", ".webm", ".mkv", ".mp4"}
_IMAGE_EXTS = {".png"}
# Pixel formats that carry an alpha channel.
_ALPHA_PIX_FMTS = {
    "yuva420p", "yuva422p", "yuva444p", "yuva420p10le", "yuva444p10le",
    "rgba", "argb", "bgra", "abgr", "ya8", "ya16le", "pal8",
}
_MARGIN = 40  # px inset from the frame edge for edge positions

POSITIONS = ["top-left", "top-center", "top-right", "center",
             "bottom-left", "bottom-center", "bottom-right"]


def list_overlays() -> list[str]:
    """Overlay asset filenames available in overlays/ (video or transparent png)."""
    out = []
    for p in sorted(gconf.OVERLAYS_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in (_VIDEO_EXTS | _IMAGE_EXTS):
            out.append(p.name)
    return out


def _pix_fmt(path: Path) -> str | None:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=pix_fmt", "-of", "json", str(path)],
        capture_output=True, text=True,
    ).stdout
    try:
        return json.loads(out)["streams"][0]["pix_fmt"]
    except (KeyError, IndexError, ValueError, json.JSONDecodeError):
        return None


def has_alpha(path: str | Path) -> bool:
    return _pix_fmt(Path(path)) in _ALPHA_PIX_FMTS


def _position_xy(position: str) -> tuple[str, str]:
    """ffmpeg overlay x:y expressions (W/H = main, w/h = overlay)."""
    m = _MARGIN
    horiz = {"left": str(m), "center": "(W-w)/2", "right": f"W-w-{m}"}
    vert = {"top": str(m), "center": "(H-h)/2", "bottom": f"H-h-{m}"}
    parts = (position or gconf.OVERLAY_DEFAULT_POSITION).split("-")
    v = vert.get(parts[0], "H-h-%d" % m)
    h = horiz.get(parts[-1], "(W-w)/2")
    return h, v


def composite(base: str | Path, overlay_name: str, out: str | Path,
              position: str | None = None, start: float | None = None,
              duration: float | None = None, width_frac: float | None = None,
              pos_y_frac: float | None = None) -> Path:
    """Composite the named overlay (from overlays/) onto `base`, writing `out`.

    The overlay is scaled to `width_frac` of the frame width (aspect preserved) and
    centred horizontally (per `position`), with its vertical CENTRE at `pos_y_frac` —
    so the wide branded banner lands as a bottom bar that clears the caption band.
    `duration` of 0 / None means show until the end of the clip. Raises a
    FriendlyError for the real failure modes (asset missing, no alpha channel).
    Idempotent."""
    base, out = Path(base), Path(out)
    if out.exists():
        return out
    asset = gconf.OVERLAYS_DIR / overlay_name
    if not asset.exists():
        raise FriendlyError(
            f"Overlay asset not found: {asset}\nPut a transparent .mov/.webm/.png "
            f"in the overlays/ folder, then pick it again.")
    if not has_alpha(asset):
        raise FriendlyError(
            f"Overlay '{overlay_name}' has no alpha channel (pix_fmt="
            f"{_pix_fmt(asset)}). Use a transparent .png or an alpha video "
            f"(e.g. ProRes 4444 .mov or VP9 .webm).")

    position = position or gconf.OVERLAY_DEFAULT_POSITION
    start = gconf.OVERLAY_DEFAULT_START if start is None else float(start)
    width_frac = gconf.OVERLAY_WIDTH_FRAC if width_frac is None else float(width_frac)
    pos_y_frac = gconf.OVERLAY_POS_Y_FRAC if pos_y_frac is None else float(pos_y_frac)
    is_video = asset.suffix.lower() in _VIDEO_EXTS

    enable = ""
    if duration and float(duration) > 0:
        enable = f":enable='between(t,{start:.3f},{start + float(duration):.3f})'"
    elif start > 0:
        enable = f":enable='gte(t,{start:.3f})'"

    # Scale the overlay to `width_frac` of the frame width (aspect preserved, even
    # height). Horizontal anchor from the position preset (centre by default); the
    # vertical CENTRE sits at `pos_y_frac` so the banner lands in a fixed low band.
    ow = max(2, int(round(gconf.WIDTH * width_frac)))
    x_anchor = _position_xy(position)[0]
    y_expr = f"H*{pos_y_frac:.4f}-h/2"

    # The overlay input must be an endless stream so it persists across its enable
    # window: loop a video overlay (a short animation repeats), and loop a still
    # png (a single frame would otherwise show only at t=0 then vanish).
    if is_video:
        inputs = ["-i", str(base), "-stream_loop", "-1", "-i", str(asset)]
        pre = f"[1:v]setpts=PTS-STARTPTS+{start}/TB,scale={ow}:-2[ov];"
    else:
        inputs = ["-i", str(base), "-loop", "1", "-i", str(asset)]
        pre = f"[1:v]scale={ow}:-2[ov];"
    graph = (f"{pre}[0:v][ov]overlay={x_anchor}:{y_expr}{enable}:"
             f"eof_action=pass:format=auto[v]")

    from gameplay import encode as enc
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", graph, "-map", "[v]"]
    if _has_audio(base):
        cmd += ["-map", "0:a", "-c:a", "copy"]
    # Overlay is the LAST pass when used, so it's the quality-targeted final encode.
    cmd += [*enc.final_args(), "-shortest", str(out)]
    _run(cmd)
    return out
