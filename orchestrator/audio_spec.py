"""Sound-effects timeline model: parse + validate the manifest's `audio` block.

The SFX layer is an OPTIONAL extension of the existing prompt manifest
(`manifests/<name>.json`). A manifest with no `audio` key and no per-clip `sfx`
key behaves exactly as before — this module is never invoked on the legacy path.

Three layers (see README):
  - ambient_bed : one continuous track for the whole video (loops, gain in dB)
  - music_bed   : same idea, a second continuous bed (e.g. an imported song)
  - motif       : a recurring/loopable cue placed one or more times
  - one-shot    : a discrete cue anchored to a moment, gathered from each clip's
                  `sfx[]` array

A cue's `source` is a library TAG, an `@import/<name>` alias, or a raw path —
never a narration string, so TTS can physically never speak a cue tag.

This module is pure (no ffmpeg, no disk writes beyond reading the tag map): it
turns JSON into validated `Cue` objects and raises `AudioSpecError` with EVERY
problem listed at once, because silent/partial failure is the thing we are
fixing. Anchor → absolute-time resolution happens later in `modules.audio_mix`,
where the scene timeline and Whisper word timings are available.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SFX_DIR = ROOT / "assets" / "sfx"
SFX_MAP_PATH = SFX_DIR / "sfx_map.json"

LAYERS = ("ambient_bed", "music_bed", "motif", "oneshot")
_AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"}


class AudioSpecError(ValueError):
    """Raised with a multi-line, human-readable list of every spec problem."""


@dataclass
class Cue:
    """One resolved sound cue, ready for the mixer to place on the timeline."""
    source: str                      # as written: tag, @import/x, or path
    path: Path                       # resolved absolute file path
    layer: str                       # one of LAYERS
    label: str                       # stable id for ffmpeg node naming / errors
    gain_db: float = 0.0
    pan: float | None = None         # -1.0 (L) .. 1.0 (R); None = centred
    fade_in: float = 0.0
    fade_out: float = 0.0
    loop: bool = False
    anchor: dict | None = None       # raw anchor dict; resolved at mix time
    scene_index: int | None = None   # 1-based; default scene for anchor context


@dataclass
class AudioSpec:
    ambient_bed: Cue | None = None
    music_bed: Cue | None = None
    motifs: list[Cue] = field(default_factory=list)
    oneshots: list[Cue] = field(default_factory=list)
    duck_enabled: bool = False
    duck_amount_db: float = 8.0
    duck_threshold: float = 0.05

    def all_cues(self) -> list[Cue]:
        beds = [c for c in (self.ambient_bed, self.music_bed) if c]
        return beds + self.motifs + self.oneshots

    def is_empty(self) -> bool:
        return not self.all_cues()


def load_sfx_map(map_path: Path | None = None) -> dict[str, str]:
    """Load the tag→relative-path map. Missing file = empty library (not fatal)."""
    p = map_path or SFX_MAP_PATH
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AudioSpecError(f"{p.name} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise AudioSpecError(f"{p.name} must be a JSON object of tag→path.")
    return {str(k): str(v) for k, v in data.items()}


def resolve_source(source: str, sfx_map: dict[str, str],
                   sfx_dir: Path | None = None, root: Path | None = None) -> Path:
    """Resolve a cue `source` to an existing file, or raise a clear error.

    Order: library TAG → `@import/<name>` alias → raw path (abs or repo-relative).
    """
    sfx_dir = sfx_dir or SFX_DIR
    root = root or ROOT
    if not isinstance(source, str) or not source.strip():
        raise AudioSpecError("cue 'source' must be a non-empty string "
                             "(a library tag, @import/<name>, or a file path).")
    source = source.strip()

    # 1. library tag
    if source in sfx_map:
        p = (sfx_dir / sfx_map[source])
        if not p.exists():
            raise AudioSpecError(
                f"tag '{source}' maps to '{sfx_map[source]}' but that file is "
                f"missing under {sfx_dir.name}/. Fix sfx_map.json or add the file.")
        return p.resolve()

    # 2. imported alias  @import/<name>
    if source.startswith("@import/"):
        name = source[len("@import/"):]
        hits = [p for p in (sfx_dir / "imported").glob(f"{name}.*")
                if p.suffix.lower() in _AUDIO_EXT]
        if not hits:
            raise AudioSpecError(
                f"imported source '{source}' not found in {sfx_dir.name}/imported/. "
                f"Import it first (CLI --sfx-import, or the GUI upload).")
        return hits[0].resolve()

    # 3. raw path (absolute or relative to repo root)
    p = Path(source)
    if not p.is_absolute():
        p = root / p
    if p.exists():
        return p.resolve()

    raise AudioSpecError(
        f"source '{source}' is not a known tag, not an @import alias, and not an "
        f"existing file. Known tags: {', '.join(sorted(sfx_map)) or '(none)'}.")


def _num(val, name: str, errors: list[str], default: float = 0.0,
         lo: float | None = None, hi: float | None = None) -> float:
    if val is None:
        return default
    try:
        f = float(val)
    except (TypeError, ValueError):
        errors.append(f"{name}: expected a number, got {val!r}.")
        return default
    if lo is not None and f < lo:
        errors.append(f"{name}: {f} is below the minimum {lo}.")
    if hi is not None and f > hi:
        errors.append(f"{name}: {f} is above the maximum {hi}.")
    return f


def _parse_anchor(raw: dict | None, label: str, layer: str,
                  errors: list[str]) -> dict | None:
    """Validate anchor STRUCTURE. Time resolution happens at mix time.

    Accepted shapes:
      {"scene": <int>, "offset": <sec>}      segment-relative
      {"word": "<text>", "occurrence": <n>, "offset": <sec>}  word-anchored
      {"time": <sec>}                         absolute from video start
    Beds need no anchor (they span the whole video from 0).
    """
    if layer in ("ambient_bed", "music_bed"):
        return None
    if raw is None:
        errors.append(f"{label}: '{layer}' cue needs an 'at' anchor "
                      f"({{scene, offset}}, {{word}}, or {{time}}).")
        return None
    if not isinstance(raw, dict):
        errors.append(f"{label}: 'at' must be an object, got {raw!r}.")
        return None

    has = {k for k in ("scene", "word", "time") if k in raw}
    if not has:
        errors.append(f"{label}: 'at' must contain one of 'scene', 'word', "
                      f"or 'time'.")
        return None
    if len(has) > 1:
        errors.append(f"{label}: 'at' has conflicting keys {sorted(has)} — "
                      f"use exactly one of scene/word/time.")
        return None

    clean: dict = {}
    if "scene" in raw:
        try:
            clean["scene"] = int(raw["scene"])
        except (TypeError, ValueError):
            errors.append(f"{label}: 'at.scene' must be an integer (1-based).")
        clean["offset"] = _num(raw.get("offset"), f"{label}: at.offset", errors,
                               default=0.0)
    elif "word" in raw:
        w = raw["word"]
        if not isinstance(w, str) or not w.strip():
            errors.append(f"{label}: 'at.word' must be a non-empty string.")
        else:
            clean["word"] = w.strip()
        clean["occurrence"] = max(1, int(raw.get("occurrence", 1) or 1))
        clean["offset"] = _num(raw.get("offset"), f"{label}: at.offset", errors,
                               default=0.0)
        if "scene" in raw:  # optional scoping handled above; kept for context
            clean["scene"] = int(raw["scene"])
    else:  # time
        clean["time"] = _num(raw.get("time"), f"{label}: at.time", errors,
                             default=0.0, lo=0.0)
    return clean


def _parse_cue(raw: dict, layer: str, label: str, sfx_map: dict[str, str],
               errors: list[str], scene_index: int | None = None,
               sfx_dir: Path | None = None, root: Path | None = None) -> Cue | None:
    if not isinstance(raw, dict):
        errors.append(f"{label}: each cue must be an object, got {raw!r}.")
        return None

    path = ROOT / "__missing__"
    try:
        path = resolve_source(raw.get("source"), sfx_map, sfx_dir, root)
    except AudioSpecError as exc:
        errors.append(f"{label}: {exc}")

    gain = _num(raw.get("gain_db"), f"{label}: gain_db", errors, default=0.0,
                lo=-60.0, hi=24.0)
    pan = None
    if raw.get("pan") is not None:
        pan = _num(raw.get("pan"), f"{label}: pan", errors, default=0.0,
                   lo=-1.0, hi=1.0)
    fade_in = _num(raw.get("fade_in"), f"{label}: fade_in", errors, lo=0.0)
    fade_out = _num(raw.get("fade_out"), f"{label}: fade_out", errors, lo=0.0)
    loop = bool(raw.get("loop", layer in ("ambient_bed", "music_bed")))
    anchor = _parse_anchor(raw.get("at"), label, layer, errors)

    return Cue(source=str(raw.get("source", "")), path=path, layer=layer,
               label=label, gain_db=gain, pan=pan, fade_in=fade_in,
               fade_out=fade_out, loop=loop, anchor=anchor,
               scene_index=scene_index)


def parse_audio_spec(manifest: dict, sfx_map: dict[str, str] | None = None,
                     sfx_dir: Path | None = None,
                     root: Path | None = None) -> AudioSpec:
    """Build a validated AudioSpec from a manifest dict.

    Returns an empty spec when the manifest carries no audio data. Raises
    AudioSpecError listing every problem when the audio data is malformed.
    """
    if sfx_map is None:
        sfx_map = load_sfx_map()
    errors: list[str] = []
    spec = AudioSpec()

    audio = manifest.get("audio") or {}
    if audio and not isinstance(audio, dict):
        raise AudioSpecError("manifest 'audio' must be an object.")

    for bed in ("ambient_bed", "music_bed"):
        raw = audio.get(bed)
        if raw is None:
            continue
        cue = _parse_cue(raw, bed, bed, sfx_map, errors, sfx_dir=sfx_dir, root=root)
        if cue:
            setattr(spec, bed, cue)

    motifs = audio.get("motifs") or []
    if motifs and not isinstance(motifs, list):
        errors.append("audio.motifs must be a list.")
        motifs = []
    for i, raw in enumerate(motifs):
        cue = _parse_cue(raw, "motif", f"motif[{i}]", sfx_map, errors,
                         sfx_dir=sfx_dir, root=root)
        if cue:
            spec.motifs.append(cue)

    # one-shots: gathered from each clip's optional `sfx` array
    clips = manifest.get("clips") or []
    for ci, clip in enumerate(clips, start=1):
        cues = (clip or {}).get("sfx") or []
        if cues and not isinstance(cues, list):
            errors.append(f"clip {ci} ({clip.get('name', '?')}): 'sfx' must be a list.")
            continue
        for si, raw in enumerate(cues):
            label = f"scene {ci} sfx[{si}]"
            cue = _parse_cue(raw, "oneshot", label, sfx_map, errors,
                             scene_index=ci, sfx_dir=sfx_dir, root=root)
            if cue:
                # a one-shot with no explicit scene anchor defaults to its scene
                if cue.anchor and "scene" not in cue.anchor and \
                        "word" not in cue.anchor and "time" not in cue.anchor:
                    cue.anchor["scene"] = ci
                spec.oneshots.append(cue)

    duck = audio.get("ducking") or {}
    if duck and not isinstance(duck, dict):
        errors.append("audio.ducking must be an object.")
        duck = {}
    spec.duck_enabled = bool(duck.get("enabled", False))
    spec.duck_amount_db = _num(duck.get("amount_db"), "ducking.amount_db", errors,
                               default=8.0, lo=0.0, hi=48.0)
    spec.duck_threshold = _num(duck.get("threshold"), "ducking.threshold", errors,
                               default=0.05, lo=0.0, hi=1.0)

    if errors:
        raise AudioSpecError(
            "Audio spec has problems:\n  - " + "\n  - ".join(errors))
    return spec
