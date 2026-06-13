"""Bring-your-own-audio: import mp3s, loudness-normalize, register as cues.

`import_audio()` takes one or more audio files, runs a single-pass ffmpeg
`loudnorm` (deterministic, sane streaming levels), writes the result into
`assets/sfx/imported/<name>.wav` (48 kHz stereo), and registers it in
`sfx_map.json` so it can be referenced exactly like a library cue — as a bare
tag `<name>` or as `@import/<name>` — on any layer (one-shot, motif, bed).

Imported files are gitignored; the registration lives in sfx_map.json.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from orchestrator.audio_spec import SFX_DIR, SFX_MAP_PATH, _AUDIO_EXT

IMPORTED_DIR = SFX_DIR / "imported"

# Streaming-sane loudness target (LUFS / true-peak dB / loudness-range).
TARGET_I = -16.0
TARGET_TP = -1.5
TARGET_LRA = 11.0


class ImportError_(RuntimeError):
    """Raised when an import can't be read or normalized."""


def _slugify(stem: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    return s[:40] or "track"


def _unique_name(name: str, sfx_map: dict[str, str]) -> str:
    """Avoid clobbering an existing tag or imported file."""
    base, n, cand = name, 1, name
    while cand in sfx_map or list(IMPORTED_DIR.glob(f"{cand}.*")):
        n += 1
        cand = f"{base}_{n}"
    return cand


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ImportError_(f"ffmpeg failed:\n{' '.join(cmd)}\n\n{proc.stderr[-1500:]}")
    return proc.stderr


def register_import(name: str, rel_path: str, map_path: Path | None = None) -> None:
    """Add/replace a tag in sfx_map.json (created if missing)."""
    map_path = map_path or SFX_MAP_PATH
    data = {}
    if map_path.exists():
        try:
            data = json.loads(map_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    data[name] = rel_path
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def import_audio(src: str | Path, name: str | None = None,
                 map_path: Path | None = None) -> str:
    """Import & normalize ONE file. Returns the registered tag name.

    The same name is reachable as a bare tag or as `@import/<name>`.
    """
    src = Path(src)
    if not src.exists():
        raise ImportError_(f"Import source not found: {src}")
    if src.suffix.lower() not in _AUDIO_EXT:
        raise ImportError_(
            f"Unsupported audio type '{src.suffix}'. Supported: "
            f"{', '.join(sorted(_AUDIO_EXT))}.")

    IMPORTED_DIR.mkdir(parents=True, exist_ok=True)
    from orchestrator.audio_spec import load_sfx_map
    sfx_map = load_sfx_map(map_path)

    tag = _unique_name(_slugify(name or src.stem), sfx_map)
    out = IMPORTED_DIR / f"{tag}.wav"

    # Single-pass loudnorm → 48k stereo wav for clean, deterministic mixing.
    _run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-af", f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}",
        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
        str(out),
    ])
    if not out.exists() or out.stat().st_size == 0:
        raise ImportError_(f"Normalization produced no output for {src.name}.")

    register_import(tag, f"imported/{out.name}", map_path)
    return tag


def import_many(srcs: list[str | Path], map_path: Path | None = None) -> list[str]:
    """Import several files; returns the list of registered tag names."""
    return [import_audio(s, map_path=map_path) for s in srcs]
