"""Multi-layer audio mixer: SFX cues + beds + VO → one deterministic track.

Two halves:

  resolve_placements()  PURE. Turns an AudioSpec's anchors into absolute
                        start times using the scene timeline and Whisper word
                        timings. No ffmpeg, no disk — fully unit-testable.
                        Raises clear errors for out-of-range scenes / unknown
                        anchor words.

  render_mix()          Builds a SINGLE ffmpeg filter_complex graph:
                        per cue  ->  atrim + volume + pan + afade in/out + adelay
                        then      ->  amix(normalize=0)   [never auto-normalize]
                        optional  ->  sidechaincompress (SFX duck under the VO)
                        then      ->  amix VO + SFX (normalize=0)

The order of cues in the graph is stable (beds, motifs in order, one-shots in
scene order) so the same spec always produces the same command — deterministic.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from orchestrator.audio_spec import AudioSpec, Cue


@dataclass
class Placement:
    """A cue resolved to an absolute position on the final timeline."""
    path: Path
    start: float          # seconds from video start
    gain_db: float
    pan: float | None
    fade_in: float
    fade_out: float
    loop: bool
    label: str


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def pan_gains(pan: float) -> tuple[float, float]:
    """Constant-power L/R gains for pan in [-1, 1] (-1=L, 0=centre, 1=R)."""
    pan = max(-1.0, min(1.0, pan))
    angle = (pan + 1.0) * 0.25 * math.pi      # 0..pi/2
    return math.cos(angle), math.sin(angle)


def _norm_word(s: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", s.lower())


def find_word_time(words, word: str, occurrence: int = 1) -> float | None:
    """Start time of the Nth (1-based) occurrence of `word`, or None."""
    target = _norm_word(word)
    seen = 0
    for w in words or []:
        if _norm_word(w.text) == target:
            seen += 1
            if seen >= occurrence:
                return float(w.start)
    return None


def resolve_cue_time(cue: Cue, scenes_timeline: list[dict], words) -> float:
    """Absolute start time for one cue. Raises ValueError with a clear message."""
    if cue.layer in ("ambient_bed", "music_bed"):
        return 0.0
    a = cue.anchor or {}
    if "time" in a:
        return max(0.0, float(a["time"]))
    if "word" in a:
        t = find_word_time(words, a["word"], int(a.get("occurrence", 1)))
        if t is None:
            raise ValueError(
                f"{cue.label}: anchor word '{a['word']}' "
                f"(occurrence {a.get('occurrence', 1)}) was not found in the "
                f"narration. Check spelling against the script.")
        return max(0.0, t + float(a.get("offset", 0.0)))
    if "scene" in a:
        idx = int(a["scene"])
        if idx < 1 or idx > len(scenes_timeline):
            raise ValueError(
                f"{cue.label}: anchor scene {idx} is out of range "
                f"(1..{len(scenes_timeline)}).")
        return max(0.0, float(scenes_timeline[idx - 1]["start"])
                   + float(a.get("offset", 0.0)))
    raise ValueError(f"{cue.label}: cue has no usable anchor.")


def resolve_placements(spec: AudioSpec, scenes_timeline: list[dict],
                       words, total_duration: float) -> list[Placement]:
    """Resolve every cue to a Placement. Stable order, clear errors."""
    out: list[Placement] = []
    errors: list[str] = []
    ordered: list[Cue] = []
    if spec.ambient_bed:
        ordered.append(spec.ambient_bed)
    if spec.music_bed:
        ordered.append(spec.music_bed)
    ordered += spec.motifs
    ordered += spec.oneshots

    for cue in ordered:
        try:
            start = resolve_cue_time(cue, scenes_timeline, words)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if start >= total_duration:
            errors.append(f"{cue.label}: start {start:.2f}s is at/after the end "
                          f"of the video ({total_duration:.2f}s).")
            continue
        out.append(Placement(path=cue.path, start=start, gain_db=cue.gain_db,
                             pan=cue.pan, fade_in=cue.fade_in,
                             fade_out=cue.fade_out, loop=cue.loop,
                             label=cue.label))
    if errors:
        raise ValueError("Cannot place SFX cues:\n  - " + "\n  - ".join(errors))
    return out


def _probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format",
         str(path)], capture_output=True, text=True).stdout
    try:
        return float(json.loads(out)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return 0.0


def _cue_chain(idx: int, p: Placement, total_duration: float) -> tuple[str, str]:
    """Build the per-cue filter chain. Returns (chain_string, out_label)."""
    if p.loop:
        play = max(0.1, total_duration - p.start)
    else:
        dur = _probe_duration(p.path)
        play = max(0.1, min(dur or total_duration, total_duration - p.start))

    parts = [
        "aformat=sample_rates=48000:channel_layouts=stereo",
        f"atrim=duration={play:.3f}",
        "asetpts=PTS-STARTPTS",
        f"volume={db_to_linear(p.gain_db):.5f}",
    ]
    if p.pan is not None:
        lg, rg = pan_gains(p.pan)
        parts.append(f"pan=stereo|c0={lg:.4f}*c0|c1={rg:.4f}*c1")
    if p.fade_in > 0:
        parts.append(f"afade=t=in:st=0:d={p.fade_in:.3f}")
    if p.fade_out > 0 and play > p.fade_out:
        parts.append(f"afade=t=out:st={play - p.fade_out:.3f}:d={p.fade_out:.3f}")
    delay_ms = int(round(p.start * 1000))
    if delay_ms > 0:
        parts.append(f"adelay={delay_ms}:all=1")
    label = f"[s{idx}]"
    return f"[{idx + 1}:a]" + ",".join(parts) + label, label


def build_filter(placements: list[Placement], total_duration: float,
                 duck_enabled: bool, duck_amount_db: float,
                 duck_threshold: float) -> str:
    """Assemble the full filter_complex string (VO is input 0)."""
    chains: list[str] = []
    labels: list[str] = []
    for i, p in enumerate(placements):
        chain, label = _cue_chain(i, p, total_duration)
        chains.append(chain)
        labels.append(label)

    if len(labels) == 1:
        chains.append(f"{labels[0]}anull[sfxmix]")
    else:
        chains.append("".join(labels)
                      + f"amix=inputs={len(labels)}:normalize=0:"
                        "dropout_transition=0[sfxmix]")

    if duck_enabled:
        # VO splits into the audible track + a sidechain key for the compressor.
        chains.append("[0:a]aformat=sample_rates=48000:channel_layouts=stereo,"
                      "asplit=2[vo][vokey]")
        ratio = max(2.0, min(20.0, 1.0 + duck_amount_db / 2.0))
        chains.append(
            f"[sfxmix][vokey]sidechaincompress=threshold={duck_threshold:.4f}:"
            f"ratio={ratio:.2f}:attack=20:release=300:makeup=1[sfxduck]")
        chains.append("[vo][sfxduck]amix=inputs=2:normalize=0:"
                      "dropout_transition=0[mix]")
    else:
        chains.append("[0:a]aformat=sample_rates=48000:channel_layouts=stereo[vo]")
        chains.append("[vo][sfxmix]amix=inputs=2:normalize=0:"
                      "dropout_transition=0[mix]")
    return ";".join(chains)


def render_mix(voice: Path, placements: list[Placement], out: Path,
               total_duration: float, duck_enabled: bool = False,
               duck_amount_db: float = 8.0, duck_threshold: float = 0.05) -> Path:
    """Run the mix. Falls back to a plain VO copy when there are no placements."""
    voice, out = Path(voice), Path(out)
    if not placements:
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", str(voice), "-c:a", "aac", "-b:a", "192k",
                        str(out)], check=True)
        return out

    inputs: list[str] = ["-i", str(voice)]
    for p in placements:
        if p.loop:
            inputs += ["-stream_loop", "-1", "-i", str(p.path)]
        else:
            inputs += ["-i", str(p.path)]

    filt = build_filter(placements, total_duration, duck_enabled,
                        duck_amount_db, duck_threshold)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *inputs,
           "-filter_complex", filt, "-map", "[mix]",
           "-c:a", "aac", "-b:a", "192k", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"SFX mix failed:\nfilter: {filt}\n\n{proc.stderr[-2000:]}")
    return out


def mix_with_spec(voice: Path, out: Path, spec: AudioSpec,
                  scenes_timeline: list[dict], words, total_duration: float,
                  legacy_music: Path | None = None,
                  legacy_music_gain_db: float = -16.0) -> Path:
    """Top-level: resolve a spec and render. A legacy `--music` track, if given
    and no music_bed is set in the spec, is folded in as a looped bed."""
    placements = resolve_placements(spec, scenes_timeline, words, total_duration)
    if legacy_music and legacy_music.exists() and spec.music_bed is None:
        placements.insert(0, Placement(
            path=Path(legacy_music), start=0.0, gain_db=legacy_music_gain_db,
            pan=None, fade_in=1.5, fade_out=2.0, loop=True, label="legacy_music"))
    return render_mix(voice, placements, out, total_duration,
                      spec.duck_enabled, spec.duck_amount_db, spec.duck_threshold)
