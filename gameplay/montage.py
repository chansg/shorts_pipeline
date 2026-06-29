"""Music-montage Shorts builder — several gameplay MP4s + one MP3 -> one 9:16 Short.

Replaces the old lore-video builder as the Shorts flow. The clips are reframed to 9:16
and stitched with light xfade crossfades; their game audio is denoised and ducked to a
low bed UNDER a dominant music track (the user picks where in the song to start); the
GamerChans overlay is applied. Reframe and overlay are REUSED from the gameplay pipeline
(gameplay.reframe / gameplay.overlay) — one source of truth, not re-implemented here.

Assembly (three passes, near-lossless intermediates so only the overlay pass governs
quality):
  1. reframe each clip -> seg_i.mp4 (9:16, normalised fps/size/SAR so xfade lines up)
  2. ONE ffmpeg graph: xfade the video chain + acrossfade the game-audio bed, denoise
     (afftdn) + duck (volume) that bed, fade the music in/out, and amix=normalize=0 with
     the music dominant -> montage_body.mp4
  3. gameplay.overlay.composite() lays the GamerChans banner on top = the final encode.

The filter-graph string builder, the duration maths and the timecode parser are PURE and
unit-tested without ffmpeg; build_montage is the ffmpeg-touching orchestrator.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Callable

from orchestrator.errors import FriendlyError, ensure_ffmpeg
from gameplay import config as gconf
from gameplay import encode as enc
from gameplay import reframe as reframe_mod
from gameplay import overlay as overlay_mod
from gameplay.state import slugify

Progress = Callable[[str], None]

_MUSIC_SR = 48000


# ---- PURE: timecode + durations --------------------------------------------

def _unquote(value) -> str:
    """Normalise a pasted path: strip whitespace and a single layer of surrounding quotes.
    Windows 'Copy as path' wraps the path in double quotes, which would otherwise become
    literal characters in the path and make it 'not found'."""
    s = str(value if value is not None else "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()
    return s


def parse_timecode(value) -> float:
    """Parse a music start offset: 'mm:ss', 'h:mm:ss', or plain seconds -> float seconds.
    Raises FriendlyError on junk. (Range checks are the caller's — it knows the track len.)"""
    s = str(value if value is not None else "").strip()
    if not s:
        return 0.0
    try:
        if ":" in s:
            sec = 0.0
            for part in s.split(":"):
                sec = sec * 60 + float(part)
            return sec
        return float(s)
    except ValueError:
        raise FriendlyError(f"Invalid time '{value}' - use mm:ss (e.g. 0:45) or seconds.")


def montage_duration(durations, tdur: float) -> float:
    """PURE. Total length after xfade: each of the (n-1) crossfades overlaps two clips by
    `tdur`, so the joined length is sum(durations) - (n-1)*tdur. One clip -> its own length."""
    durations = list(durations)
    if not durations:
        return 0.0
    n = len(durations)
    return float(sum(durations) - (n - 1) * tdur)


def xfade_offsets(durations, tdur: float) -> list[float]:
    """PURE. The xfade `offset` (start of the crossfade on the accumulated stream) for each
    of the (n-1) joins. Offset_k = L_{k-1} - tdur, where L grows by (dur_k - tdur) per join."""
    offsets: list[float] = []
    if len(durations) < 2:
        return offsets
    acc = float(durations[0])
    for k in range(1, len(durations)):
        offsets.append(round(acc - tdur, 4))
        acc += float(durations[k]) - tdur
    return offsets


# ---- PURE: the ffmpeg filter_complex ---------------------------------------

def build_filtergraph(durations, *, transition: str, tdur: float, game_gain: float,
                      denoise, music_fadein: float, music_fadeout: float,
                      music_fadeout_start: float, music_gain: float, fade_ends: bool,
                      fade_dur: float, montage_dur: float) -> str:
    """PURE. Build the filter_complex that maps inputs [0..n-1] (the reframed segments,
    video + one audio each) and input [n] (the seeked music) to [v] (xfaded video) and
    [a] (music + denoised/ducked game bed). Returns the graph string; maps are [v]/[a]."""
    n = len(durations)
    parts: list[str] = []

    # --- video: xfade chain (accumulated offsets) ---
    if n == 1:
        vbody = "[0:v]"
    else:
        offs = xfade_offsets(durations, tdur)
        prev = "[0:v]"
        for k in range(1, n):
            out = f"[vx{k}]"
            parts.append(f"{prev}[{k}:v]xfade=transition={transition}:"
                         f"duration={tdur:.4f}:offset={offs[k - 1]:.4f}{out}")
            prev = out
        vbody = prev
    if fade_ends:
        fo = max(0.0, montage_dur - fade_dur)
        parts.append(f"{vbody}fade=t=in:st=0:d={fade_dur:.3f},"
                     f"fade=t=out:st={fo:.3f}:d={fade_dur:.3f}[v]")
    else:
        parts.append(f"{vbody}null[v]")

    # --- game audio bed: per-seg normalise -> acrossfade chain -> denoise + duck ---
    for i in range(n):
        parts.append(f"[{i}:a:0]aresample={_MUSIC_SR},"
                     f"aformat=sample_fmts=fltp:channel_layouts=stereo[ga{i}]")
    if n == 1:
        gbody = "[ga0]"
    else:
        prev = "[ga0]"
        for k in range(1, n):
            out = f"[gx{k}]"
            parts.append(f"{prev}[ga{k}]acrossfade=d={tdur:.4f}:c1=tri:c2=tri{out}")
            prev = out
        gbody = prev
    duck = []
    if denoise:
        duck.append(f"afftdn=nr={denoise}")
    duck.append(f"volume={game_gain}")
    parts.append(f"{gbody}{','.join(duck)}[game]")

    # --- music bed (input n): fade in + fade out + gain (dominant) ---
    parts.append(
        f"[{n}:a]aresample={_MUSIC_SR},"
        f"aformat=sample_fmts=fltp:channel_layouts=stereo,"
        f"afade=t=in:st=0:d={music_fadein:.3f},"
        f"afade=t=out:st={music_fadeout_start:.3f}:d={music_fadeout:.3f},"
        f"volume={music_gain}[music]")

    # --- mix: music dominant, ducked game underneath (no auto-normalise) ---
    parts.append("[music][game]amix=inputs=2:duration=longest:normalize=0[a]")
    return ";".join(parts)


# ---- ffmpeg seams ----------------------------------------------------------

def _out_name(music: Path) -> str:
    return f"{slugify(music.stem) or 'montage'}_montage_{time.strftime('%Y%m%d_%H%M%S')}.mp4"


def _ensure_audio(seg: Path, dest: Path) -> Path:
    """Guarantee the segment has one audio stream (xfade/acrossfade need it). If the clip
    had no audio, mux a silent stereo track; otherwise return it unchanged."""
    from modules.assemble import _run, _has_audio
    if _has_audio(seg):
        return seg
    _run(["ffmpeg", "-y", "-i", str(seg), "-f", "lavfi",
          "-i", f"anullsrc=r={_MUSIC_SR}:cl=stereo", "-map", "0:v", "-map", "1:a",
          "-c:v", "copy", "-c:a", "aac", "-shortest", str(dest)])
    return dest


def build_montage(clip_paths, music_path, music_start, *, out_path=None,
                  progress: Progress | None = None) -> Path:
    """Build ONE 9:16 montage Short from `clip_paths` (in order) + `music_path`, with the
    music starting `music_start` seconds into the track. Returns the output path.

    Fail-safe: a clip that fails to reframe is logged and skipped; an invalid music offset
    or a missing file raises a FriendlyError (the GUI shows it without crashing). Music
    shorter than the montage fades out early with a WARNING (no loop unless configured)."""
    from modules.assemble import _run, _probe_duration
    emit = progress or (lambda m: None)
    ensure_ffmpeg()

    clips = [Path(p) for p in (_unquote(c) for c in (clip_paths or [])) if p]
    if not clips:
        raise FriendlyError("Add at least one gameplay clip to the montage.")
    for c in clips:
        if not c.exists():
            raise FriendlyError(f"Clip not found: {c}")
    music_str = _unquote(music_path)
    music = Path(music_str) if music_str else None
    if not music:
        raise FriendlyError("Pick a music file (one MP3) for the montage.")
    if not music.exists():
        raise FriendlyError(f"Music file not found: {music}")

    music_dur = float(_probe_duration(music) or 0.0)
    if music_dur <= 0:
        raise FriendlyError(f"Could not read the music duration of {music.name}.")
    start = parse_timecode(music_start)
    if start < 0:
        raise FriendlyError("Music start must be >= 0.")
    if start >= music_dur:
        raise FriendlyError(
            f"Music start {start:.1f}s is at/after the track length "
            f"({music_dur:.1f}s) - pick an earlier point.")

    with tempfile.TemporaryDirectory(prefix="montage_") as td:
        work = Path(td)
        segs: list[Path] = []
        for i, c in enumerate(clips):
            emit(f"Reframing clip {i + 1}/{len(clips)} to 9:16: {c.name}")
            try:
                seg = reframe_mod.reframe(c, work / f"seg_{i:03d}.mp4")   # REUSE reframe
                segs.append(_ensure_audio(seg, work / f"sega_{i:03d}.mp4"))
            except Exception as e:        # noqa: BLE001 — one bad clip mustn't sink the run
                emit(f"  WARNING: skipping clip {c.name} ({type(e).__name__}: {e})")
        if not segs:
            raise FriendlyError("No usable clips after reframing - check the inputs.")

        durations = [float(_probe_duration(s) or 0.0) for s in segs]
        n = len(segs)
        tdur = float(gconf.MONTAGE_TRANSITION_DURATION)
        if n >= 2:                                   # crossfade must fit inside each clip
            tdur = max(0.05, min(tdur, min(durations) * 0.5))
        else:
            tdur = 0.0
        montage_dur = montage_duration(durations, tdur)

        music_avail = music_dur - start
        loop = bool(gconf.MONTAGE_LOOP_MUSIC)
        if music_avail < montage_dur and not loop:
            emit(f"  WARNING: music has only {music_avail:.1f}s after a {start:.1f}s start "
                 f"but the montage is {montage_dur:.1f}s - it will fade out early "
                 f"(set MONTAGE_LOOP_MUSIC to loop instead).")
        eff_end = montage_dur if (loop or music_avail >= montage_dur) else music_avail
        fo = float(gconf.MONTAGE_MUSIC_FADEOUT)
        fo_start = max(0.0, min(montage_dur, eff_end) - fo)

        graph = build_filtergraph(
            durations, transition=str(gconf.MONTAGE_TRANSITION), tdur=tdur,
            game_gain=float(gconf.MONTAGE_GAME_AUDIO_GAIN), denoise=gconf.MONTAGE_DENOISE,
            music_fadein=float(gconf.MONTAGE_MUSIC_FADEIN), music_fadeout=fo,
            music_fadeout_start=fo_start, music_gain=float(gconf.MONTAGE_MUSIC_GAIN),
            fade_ends=bool(gconf.MONTAGE_FADE_ENDS),
            fade_dur=float(gconf.MONTAGE_FADE_DURATION), montage_dur=montage_dur)

        cmd = ["ffmpeg", "-y"]
        for s in segs:
            cmd += ["-i", str(s)]
        if loop and music_avail < montage_dur:
            cmd += ["-stream_loop", "-1"]
        cmd += ["-ss", f"{start:.3f}", "-i", str(music)]      # seek INTO the song
        body = work / "montage_body.mp4"
        cmd += ["-filter_complex", graph, "-map", "[v]", "-map", "[a]",
                *enc.intermediate_args(), "-c:a", "aac", "-b:a", "192k",
                "-t", f"{montage_dur:.3f}", str(body)]
        emit(f"Stitching {n} clip(s) with {tdur:.2f}s {gconf.MONTAGE_TRANSITION} "
             f"crossfades; denoising + ducking game audio to {gconf.MONTAGE_GAME_AUDIO_GAIN:.0%} "
             f"under the music ({montage_dur:.1f}s)...")
        _run(cmd)

        out = Path(out_path) if out_path else gconf.MONTAGE_OUTPUT_DIR / _out_name(music)
        out.parent.mkdir(parents=True, exist_ok=True)
        emit("Applying the GamerChans overlay (final encode)...")
        overlay_mod.composite(body, gconf.LIKE_SUB_OVERLAY, out, start=0.0, duration=0)
        emit(f"Done -> {out}")
        return out
