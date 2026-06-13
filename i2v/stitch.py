"""Stitch rendered clips into a single reel with ffmpeg.

Two modes:
  - concat (default): fast, lossless-ish stream copy, hard cuts between clips.
  - xfade: re-encodes with a crossfade between every clip for a smoother reel.

ffmpeg must be on PATH. On Windows: `winget install Gyan.FFmpeg` or grab a
static build and add it to PATH.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger("i2v.stitch")


def _ensure_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found on PATH. Install it and re-run with --stitch.")
    return exe


def _ordered_clips(out_dir: Path, order: list[str]) -> list[Path]:
    """Resolve output clip paths in the order given (by clip name)."""
    paths = [out_dir / f"{name}.mp4" for name in order]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        log.warning("Stitch skipping missing clips: %s", ", ".join(missing))
    return [p for p in paths if p.exists()]


def concat(out_dir: Path, order: list[str], dest: Path) -> Path:
    """Hard-cut concatenation via the ffmpeg concat demuxer (stream copy)."""
    ffmpeg = _ensure_ffmpeg()
    clips = _ordered_clips(out_dir, order)
    if not clips:
        raise RuntimeError("No clips available to stitch.")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        for clip in clips:
            fh.write(f"file '{clip.resolve().as_posix()}'\n")
        list_file = fh.name

    cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_file,
           "-c", "copy", str(dest)]
    log.info("Concatenating %d clips -> %s", len(clips), dest)
    subprocess.run(cmd, check=True)
    Path(list_file).unlink(missing_ok=True)
    return dest


def crossfade(out_dir: Path, order: list[str], dest: Path, duration: float = 0.5) -> Path:
    """Crossfade every clip into the next. Re-encodes (slower, smoother)."""
    ffmpeg = _ensure_ffmpeg()
    clips = _ordered_clips(out_dir, order)
    if len(clips) < 2:
        return concat(out_dir, order, dest)

    durations = [_probe_duration(c) for c in clips]
    inputs: list[str] = []
    for clip in clips:
        inputs += ["-i", str(clip)]

    # Chain xfade filters; offset accumulates as (sum of prior durations) - (n * fade).
    filt: list[str] = []
    last_label = "0:v"
    offset = 0.0
    for i in range(1, len(clips)):
        offset += durations[i - 1] - duration
        out_label = f"v{i}"
        filt.append(
            f"[{last_label}][{i}:v]xfade=transition=fade:duration={duration}:"
            f"offset={offset:.3f}[{out_label}]"
        )
        last_label = out_label

    cmd = [ffmpeg, "-y", *inputs, "-filter_complex", ";".join(filt),
           "-map", f"[{last_label}]", "-pix_fmt", "yuv420p", str(dest)]
    log.info("Crossfading %d clips (%.2fs) -> %s", len(clips), duration, dest)
    subprocess.run(cmd, check=True)
    return dest


def _probe_duration(clip: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found on PATH (needed for crossfade timing).")
    out = subprocess.run(
        [ffprobe, "-v", "quiet", "-print_format", "json", "-show_format", str(clip)],
        check=True, capture_output=True, text=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])
