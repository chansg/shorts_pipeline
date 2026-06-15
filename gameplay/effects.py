"""Optional, configurable effects layer for gameplay Shorts.

Effects are driven off an audio-energy envelope (loud reactions = action /
laughter / clutch) and applied as ffmpeg video filters — no per-frame Python
compositing. The starter set is punch-zoom and a subtle shake; both are toggles.

The design is a registry (`EFFECTS`) so more effects (speed-ramp, hit-flash, sfx
stinger) can be added later without restructuring: each entry takes the list of
beat times and returns a per-frame expression contribution.

Energy envelope: we decode the audio to mono 16-bit PCM via ffmpeg and compute a
windowed RMS in numpy (numpy is already a core dep). Peaks are the moments whose
loudness is `ENERGY_PEAK_Z` standard deviations above the mean.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from modules.assemble import _has_audio
from gameplay import config as gconf

_PCM_RATE = 8000   # plenty for a loudness envelope; keeps the decode tiny


def energy_envelope(src: str | Path, window_s: float | None = None):
    """Return (times, rms) arrays of the windowed loudness of `src`'s audio.
    Empty arrays if the clip has no audio."""
    src = Path(src)
    if not _has_audio(src):
        return np.array([]), np.array([])
    window_s = window_s or gconf.ENERGY_WINDOW_S
    proc = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", str(src), "-vn",
         "-ac", "1", "-ar", str(_PCM_RATE), "-f", "s16le", "-"],
        capture_output=True,
    )
    raw = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    if raw.size == 0:
        return np.array([]), np.array([])
    win = max(1, int(_PCM_RATE * window_s))
    n = raw.size // win
    if n == 0:
        return np.array([]), np.array([])
    frames = raw[: n * win].reshape(n, win)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    times = (np.arange(n) + 0.5) * window_s
    return times, rms


def detect_beats(src: str | Path, z: float | None = None,
                 max_peaks: int | None = None, min_gap: float | None = None
                 ) -> list[float]:
    """Times (seconds) of loud moments — local RMS maxima whose loudness is `z`
    std-devs above the mean. Peaks within `min_gap` are merged; at most
    `max_peaks` strongest are returned (keeps the ffmpeg expression bounded)."""
    z = gconf.ENERGY_PEAK_Z if z is None else z
    max_peaks = max_peaks or gconf.ENERGY_MAX_PEAKS
    min_gap = min_gap or gconf.ENERGY_MIN_GAP_S
    times, rms = energy_envelope(src)
    if rms.size < 3:
        return []
    thresh = rms.mean() + z * rms.std()
    cand = [(rms[i], times[i]) for i in range(1, len(rms) - 1)
            if rms[i] >= thresh and rms[i] >= rms[i - 1] and rms[i] >= rms[i + 1]]
    cand.sort(reverse=True)            # loudest first
    kept: list[float] = []
    for _, t in cand:
        if all(abs(t - k) >= min_gap for k in kept):
            kept.append(t)
        if len(kept) >= max_peaks:
            break
    return sorted(kept)


# ---- effect expression contributions -------------------------------------
# Effects are realised with a single `zoompan` filter: unlike `crop` (whose w/h
# are fixed at init), zoompan evaluates z/x/y per OUTPUT frame, so the zoom and
# shake can vary over time. Time is derived from the output frame counter `on`
# and the target fps. Each registry entry contributes a zoom term and/or an
# (x, y) pan offset; add a new effect by adding an entry + a contributor.

SHAKE_BASE_ZOOM = 1.04   # headroom (constant) so the shake has room to move


def _gauss_pulse(beats: list[float], sigma: float, t: str) -> str:
    """sum of exp(-((t-beat)/sigma)^2) — a smooth bump at each beat, ~0 elsewhere."""
    return "+".join(f"exp(-(({t}-{b:.3f})/{sigma})^2)" for b in beats) or "0"


def _punch_zoom(beats, t):
    """Returns (zoom_term, x_off, y_off)."""
    pulse = _gauss_pulse(beats, gconf.PUNCH_ZOOM_SIGMA, t)
    return f"{gconf.PUNCH_ZOOM_AMOUNT}*({pulse})", None, None


def _shake(beats, t):
    env = _gauss_pulse(beats, gconf.SHAKE_SIGMA, t)
    a, f = gconf.SHAKE_AMPLITUDE, gconf.SHAKE_FREQ
    # a constant zoom headroom term + an oscillating pan gated to the beats
    return (f"{SHAKE_BASE_ZOOM - 1}",
            f"{a}*sin({f}*{t})*({env})", f"{a}*cos({f}*{t})*({env})")


# Registry: name -> {label, contrib}. contrib(beats, t_expr) -> (zoom, x, y).
# Add new effects (speed-ramp, hit-flash, ...) here without touching the builder.
EFFECTS = {
    "punch_zoom": {"label": "Punch-zoom on loud beats", "contrib": _punch_zoom},
    "shake": {"label": "Subtle shake on loud beats", "contrib": _shake},
}


def build_effects_filter(beats: list[float], enabled: list[str],
                         w: int, h: int) -> str | None:
    """Combine the enabled effects into one `zoompan` filter, or None if nothing
    to do. With no beat nearby the zoom resolves to 1.0 and the pan to 0, i.e. the
    untouched frame."""
    enabled = [e for e in (enabled or []) if e in EFFECTS]
    if not enabled or not beats:
        return None

    fps = gconf.FPS
    t = f"(on/{fps})"
    zoom_terms = ["1"]
    x_offs, y_offs = [], []
    for name in enabled:
        z, xo, yo = EFFECTS[name]["contrib"](beats, t)
        if z:
            zoom_terms.append(f"({z})")
        if xo:
            x_offs.append(f"({xo})")
        if yo:
            y_offs.append(f"({yo})")

    zexpr = "+".join(zoom_terms)
    xexpr = "iw/2-(iw/zoom/2)" + ("+" + "+".join(x_offs) if x_offs else "")
    yexpr = "ih/2-(ih/zoom/2)" + ("+" + "+".join(y_offs) if y_offs else "")
    return (f"zoompan=z='{zexpr}':d=1:x='{xexpr}':y='{yexpr}':"
            f"s={w}x{h}:fps={fps}")


def apply_effects(src: str | Path, out: str | Path, enabled: list[str],
                  beats: list[float] | None = None) -> tuple[Path, list[float]]:
    """Apply the enabled effects to `src`, writing `out`. Returns (out, beats).
    If nothing is enabled or no beats are found, `out` is a straight copy so the
    pipeline can treat the effects stage uniformly. Idempotent."""
    from modules.assemble import _run
    src, out = Path(src), Path(out)
    beats = detect_beats(src) if beats is None else beats
    if out.exists():
        return out, beats
    vf = build_effects_filter(beats, enabled, gconf.WIDTH, gconf.HEIGHT)
    cmd = ["ffmpeg", "-y", "-i", str(src)]
    if vf:
        cmd += ["-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "medium", "-crf", "20", "-r", str(gconf.FPS)]
    else:
        cmd += ["-c:v", "copy"]      # nothing to do — straight passthrough
    cmd += ["-c:a", "copy"]          # gameplay audio is preserved untouched
    cmd.append(str(out))
    _run(cmd)
    return out, beats
