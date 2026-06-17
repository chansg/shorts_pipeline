"""Single source of truth for output encode quality (gameplay + full-auto).

The old per-stage encodes were `libx264 -crf 20 -preset medium` with no profile or
faststart, and a clip went through up to four of them (reframe -> effects -> captions
-> overlay), compounding loss to ~half the bitrate a phone needs. These helpers fix
both: the FINAL output is a quality-targeted CRF encode (profile high + faststart for
mobile), and intermediates are near-lossless so only the final CRF governs.

Return plain ffmpeg arg lists (not full commands) so each call site keeps control of
its inputs/filters/maps and just appends the codec args before the output path.
"""
from __future__ import annotations

from gameplay import config as gconf


def final_args(crf: int | None = None, preset: str | None = None) -> list[str]:
    """x264 args for a clip's FINAL, phone-ready encode: constant-quality CRF, H.264
    High profile, yuv420p, +faststart (mobile streaming), fixed fps. CRF/preset
    default to gconf.OUTPUT_CRF / OUTPUT_PRESET."""
    return [
        "-c:v", "libx264",
        "-preset", str(preset or gconf.OUTPUT_PRESET),
        "-crf", str(gconf.OUTPUT_CRF if crf is None else crf),
        "-pix_fmt", "yuv420p",
        "-profile:v", gconf.OUTPUT_PROFILE,
        "-movflags", "+faststart",
        "-r", str(gconf.FPS),
    ]


def intermediate_args(crf: int | None = None, preset: str | None = None) -> list[str]:
    """x264 args for an INTERMEDIATE pass (e.g. the cached blur-pad): near-lossless
    so the only quality-governing encode is the final one. No profile/faststart —
    those only matter on the delivered file."""
    return [
        "-c:v", "libx264",
        "-preset", str(preset or gconf.INTERMEDIATE_PRESET),
        "-crf", str(gconf.INTERMEDIATE_CRF if crf is None else crf),
        "-pix_fmt", "yuv420p",
        "-r", str(gconf.FPS),
    ]
