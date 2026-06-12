"""
Step 5 — Assemble.

ffmpeg does the heavy lifting:
  1. Each image -> 9:16 via SCALE-TO-COVER + CROP (never stretched), with a slow
     Ken Burns zoom.
  2. Crossfade between scenes for a filmic feel (no hard cuts).
  3. Mix voiceover + looped, faded background music.
  4. Burn in the .ass captions.
  5. Mux to a YouTube-Shorts-ready H.264 mp4.
"""
from __future__ import annotations
import subprocess
import json
from pathlib import Path
import config
from modules.visuals import Scene, is_video


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(cwd) if cwd else None)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{' '.join(cmd)}\n\n{proc.stderr[-2000:]}")
    return proc.stderr


def _probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True,
    ).stdout
    try:
        return float(json.loads(out)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return 0.0


def _crop_offsets(focus: str) -> tuple[str, str]:
    """Map a focus keyword to ffmpeg crop x/y expressions, so an off-centre
    subject (a lighthouse on the right, a face on the left) is kept instead of
    sliced out by the default centre crop. Horizontal: left/center/right.
    Vertical: top/center/bottom. Combine with '-' e.g. 'top-right'."""
    parts = {p.strip() for p in (focus or "center").lower().split("-") if p.strip()}
    if "left" in parts:
        x = "0"
    elif "right" in parts:
        x = "in_w-out_w"
    else:
        x = "(in_w-out_w)/2"
    if "top" in parts:
        y = "0"
    elif "bottom" in parts:
        y = "in_h-out_h"
    else:
        y = "(in_h-out_h)/2"
    return x, y


def _video_scene_clip(scene: Scene, out: Path, extra: float = 0.0) -> str:
    """Normalize a source VIDEO clip (e.g. from Higgsfield) to one scene:
    cover-crop to 9:16 (no stretch), fit to the scene's narration-driven
    duration (trim if longer, gently slow if shorter), 30fps, audio stripped.
    Returns a short note for the run log."""
    target = scene.duration + extra
    clip_dur = _probe_duration(scene.image)
    W, H = config.WIDTH, config.HEIGHT

    chain = []
    note = f"video, native {clip_dur:.1f}s -> {target:.1f}s"
    if clip_dur and clip_dur < target - 0.05:
        factor = target / clip_dur          # PTS multiplier (>1 = slower)
        chain.append(f"setpts={factor:.4f}*PTS")
        note += f" (slowed to {1/factor:.2f}x)"
    elif clip_dur > target + 0.05:
        note += " (trimmed)"
    cx, cy = _crop_offsets(getattr(scene, "focus", "center"))
    chain += [
        f"scale={W}:{H}:force_original_aspect_ratio=increase",
        f"crop={W}:{H}:{cx}:{cy}",
        f"fps={config.FPS}",
        "setsar=1", "format=yuv420p",
    ]
    _run([
        "ffmpeg", "-y", "-i", str(scene.image),
        "-t", f"{target:.3f}", "-vf", ",".join(chain), "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(config.FPS),
        str(out),
    ])
    return note


def _ken_burns_clip(scene: Scene, out: Path, extra: float = 0.0) -> str:
    """One image -> a 9:16 clip with a slow zoom. Scale-to-cover + crop means
    the image keeps its aspect ratio (no stretching); only overflow is cropped."""
    duration = scene.duration + extra
    frames = max(1, int(round(duration * config.FPS)))
    z = config.KEN_BURNS_ZOOM
    W, H = config.WIDTH, config.HEIGHT
    cx, cy = _crop_offsets(getattr(scene, "focus", "center"))
    # Work at 2x for a smooth zoom, cover-crop to 9:16 (anchored), then zoompan to WxH.
    vf = (
        f"scale={W*2}:{H*2}:force_original_aspect_ratio=increase,"
        f"crop={W*2}:{H*2}:{cx}:{cy},"
        f"zoompan=z='min(zoom+{(z-1)/frames:.6f},{z})':d={frames}:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"s={W}x{H}:fps={config.FPS},"
        f"setsar=1,format=yuv420p"
    )
    _run([
        "ffmpeg", "-y", "-loop", "1", "-i", str(scene.image),
        "-t", f"{duration:.3f}", "-vf", vf,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(config.FPS),
        str(out),
    ])
    return f"image (Ken Burns zoom {z}), {scene.duration:.1f}s"


def _concat_xfade(clips: list[Path], durations: list[float], out: Path,
                  transition: float) -> None:
    """Crossfade consecutive clips. Clips are rendered slightly long (by
    `transition`) so the fades don't shorten the timeline; the final mux
    trims any tail to the audio length."""
    if len(clips) == 1:
        _run(["ffmpeg", "-y", "-i", str(clips[0]), "-c", "copy", str(out)])
        return

    inputs: list[str] = []
    for c in clips:
        inputs += ["-i", str(c)]

    filt = []
    prev = "[0:v]"
    cum = 0.0
    for i in range(1, len(clips)):
        cum += durations[i - 1]           # offset = sum of scene durations so far
        label = f"[vx{i}]" if i < len(clips) - 1 else "[vout]"
        filt.append(
            f"{prev}[{i}:v]xfade=transition=fade:"
            f"duration={transition:.3f}:offset={cum:.3f}{label}"
        )
        prev = label
    _run([
        "ffmpeg", "-y", *inputs,
        "-filter_complex", ";".join(filt),
        "-map", "[vout]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(config.FPS),
        str(out),
    ])


def _mix_audio(voice: Path, music: Path | None, out: Path,
               voice_duration: float | None = None) -> None:
    if music and music.exists():
        # Loop music to cover the whole narration; fade in, and fade out at the end.
        afilter = f"[1:a]volume={config.MUSIC_VOLUME},afade=t=in:st=0:d=1.5"
        if voice_duration:
            afilter += f",afade=t=out:st={max(0.0, voice_duration - 2.0):.2f}:d=2.0"
        afilter += ("[m];[0:a][m]amix=inputs=2:duration=first:"
                    "dropout_transition=2[a]")
        _run([
            "ffmpeg", "-y", "-i", str(voice), "-stream_loop", "-1", "-i", str(music),
            "-filter_complex", afilter,
            "-map", "[a]", "-c:a", "aac", "-b:a", "192k", str(out),
        ])
    else:
        _run(["ffmpeg", "-y", "-i", str(voice), "-c:a", "aac", "-b:a", "192k", str(out)])


def _burn_and_mux(video: Path, audio: Path, ass: Path, out: Path) -> None:
    # ass filter treats ':' as a separator, so run from the subtitle's folder and
    # pass the bare filename (avoids the Windows drive-letter colon problem).
    video, audio, out = video.resolve(), audio.resolve(), out.resolve()
    _run([
        "ffmpeg", "-y", "-i", str(video), "-i", str(audio),
        "-vf", f"ass={ass.name}",
        "-map", "0:v", "-map", "1:a", "-shortest",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        str(out),
    ], cwd=ass.parent)


def _has_audio(path: Path) -> bool:
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    return bool(out)


def _cutaway_clip(scene: Scene, out: Path, extra: float = 0.0) -> str:
    """A cutaway/cutscene clip: LETTERBOX (pad) to 9:16 to preserve the full
    widescreen frame (no cropping the action), at its native duration and speed.
    Audio is stripped here — the clip's real audio is spliced into the narration
    track separately. The last frame is held for `extra` so the crossfade has
    material without slowing the footage."""
    W, H = config.WIDTH, config.HEIGHT
    chain = [
        f"scale={W}:{H}:force_original_aspect_ratio=decrease",
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black",
        f"fps={config.FPS}", "setsar=1", "format=yuv420p",
    ]
    if extra > 0:
        chain.append(f"tpad=stop_mode=clone:stop_duration={extra:.3f}")
    _run([
        "ffmpeg", "-y", "-i", str(scene.image),
        "-vf", ",".join(chain), "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(config.FPS),
        str(out),
    ])
    return f"CUTAWAY letterboxed ({scene.duration:.1f}s, own audio, narration paused)"


def _extract_audio(clip: Path, duration: float, out: Path) -> None:
    """Pull a clip's audio as 48k stereo wav (or silence if it has none)."""
    if _has_audio(clip):
        _run(["ffmpeg", "-y", "-i", str(clip), "-vn",
              "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(out)])
    else:
        _run(["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
              "-t", f"{duration:.3f}", "-c:a", "pcm_s16le", str(out)])


def _build_narration_audio(voice: Path, cutaways: list[dict], work: Path) -> Path:
    """Splice the narration so it PAUSES for each cutaway and the cutaway's own
    audio plays in the gap: voice[0:t1] + cut1 + voice[t1:t2] + cut2 + ... + tail.
    All segments normalized to 48k stereo so they concatenate cleanly."""
    if not cutaways:
        return voice
    cuts = sorted(cutaways, key=lambda c: c["narration_time"])
    segs: list[Path] = []
    prev = 0.0
    for idx, ca in enumerate(cuts):
        t = ca["narration_time"]
        seg = work / f"_na_seg_{idx}.wav"
        _run(["ffmpeg", "-y", "-i", str(voice), "-ss", f"{prev:.3f}", "-to", f"{t:.3f}",
              "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(seg)])
        segs.append(seg)
        cut = work / f"_na_cut_{idx}.wav"
        _extract_audio(ca["clip"], ca["duration"], cut)
        segs.append(cut)
        prev = t
    tail = work / "_na_tail.wav"
    _run(["ffmpeg", "-y", "-i", str(voice), "-ss", f"{prev:.3f}",
          "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(tail)])
    segs.append(tail)

    listf = work / "_na_concat.txt"
    listf.write_text("".join(f"file '{s.resolve()}'\n" for s in segs))
    full = work / "_narration_full.wav"
    _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listf),
          "-c", "copy", str(full)])
    return full


def assemble(scenes: list[Scene], voice_wav: Path, ass_file: Path,
             out_path: Path, music: Path | None = None,
             voice_duration: float | None = None,
             cutaways: list[dict] | None = None) -> list[str]:
    work = config.WORK_DIR

    durations = [s.duration for s in scenes]
    transition = config.TRANSITION_SEC
    if len(scenes) > 1:
        transition = min(transition, min(durations) * 0.5)
    transition = max(0.05, transition)

    notes: list[str] = []
    clips = []
    for i, sc in enumerate(scenes):
        clip = work / f"clip_{i:03d}.mp4"
        ex = transition if len(scenes) > 1 else 0.0
        if getattr(sc, "is_cutaway", False):
            note = _cutaway_clip(sc, clip, extra=ex)
        elif is_video(sc.image):
            note = _video_scene_clip(sc, clip, extra=ex)
        else:
            note = _ken_burns_clip(sc, clip, extra=ex)
        notes.append(f"{i+1:02d}. {sc.image.name}: {note}")
        clips.append(clip)

    silent = work / "_video.mp4"
    _concat_xfade(clips, durations, silent, transition)

    # Build the narration track (with cutaway audio spliced in), then mix music.
    narration = _build_narration_audio(voice_wav, cutaways or [], work)
    total = (voice_duration or 0.0) + sum(c["duration"] for c in (cutaways or []))
    mixed = work / "_audio.m4a"
    _mix_audio(narration, music, mixed, total or voice_duration)

    _burn_and_mux(silent, mixed, ass_file, out_path)
    return notes