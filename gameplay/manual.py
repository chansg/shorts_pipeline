"""Manual-mode orchestrator: a pre-trimmed clip + an edited transcript ->
finished 9:16 Short.

    reframe (blur-pad) -> effects (optional) -> burn captions -> overlay (optional)
    -> export.

Stages are resumable: the blur-pad (which depends only on the source) is cached
across rebuilds, while the opts-dependent stages (effects/captions/overlay) are
rebuilt so changing a toggle takes effect. Yields {"msg": str} progress events
and finally {"done": True, "output": Path} — the same shape the lore
`orchestrator.stages.voice_assemble` uses, so the GUI streams it identically.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from modules.assemble import _run, _has_audio
from modules.karaoke_captions import CaptionStyle, build_ass
from orchestrator.errors import FriendlyError, ensure_ffmpeg
from gameplay import config as gconf
from gameplay import effects as fx_mod
from gameplay import reframe as reframe_mod
from gameplay.state import GameplayClip
from gameplay.transcript import Transcript


@dataclass
class ManualOptions:
    effects: list[str] = field(default_factory=list)        # e.g. ["punch_zoom", "shake"]
    overlay_name: str | None = None
    overlay_position: str = gconf.OVERLAY_DEFAULT_POSITION
    overlay_start: float = gconf.OVERLAY_DEFAULT_START
    overlay_duration: float = gconf.OVERLAY_DEFAULT_DURATION
    caption_font: str = gconf.CAPTION_FONT
    caption_pos_y_frac: float = gconf.CAPTION_POS_Y_FRAC
    speaker_colors: dict[str, tuple[int, int, int]] = field(default_factory=dict)


def caption_style(opts: ManualOptions) -> CaptionStyle:
    return CaptionStyle(
        font=opts.caption_font,
        fontsize=gconf.CAPTION_FONTSIZE,
        play_w=gconf.WIDTH,
        play_h=gconf.HEIGHT,
        pos_y_frac=opts.caption_pos_y_frac,
        words_per_cue=1,
        gap_fill=True,
        max_gap=0.8,    # clear a held word across a long pause (matches lore look)
        hold=0.4,
        speaker_colors=opts.speaker_colors or {},
    )


def write_captions(transcript: Transcript, opts: ManualOptions, ass_path: Path) -> Path:
    ass_path.write_text(build_ass(transcript.to_tuples(), caption_style(opts)),
                        encoding="utf-8")
    return ass_path


def burn_captions(video: Path, ass: Path, out: Path) -> Path:
    """Burn the .ass onto `video`, keeping the clip's own audio. Reuses the lore
    pipeline's fontsdir trick: run from the .ass folder, pass the bare filename,
    and give fontsdir as a relative (colon-free) path so the bundled font resolves
    on Windows without a system install."""
    video, out = video.resolve(), out.resolve()
    fonts_rel = os.path.relpath(gconf.FONTS_DIR, ass.parent).replace("\\", "/")
    cmd = ["ffmpeg", "-y", "-i", str(video),
           "-vf", f"ass={ass.name}:fontsdir={fonts_rel}", "-map", "0:v"]
    if _has_audio(video):
        cmd += ["-map", "0:a", "-c:a", "copy"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium",
            "-crf", "20", str(out)]
    _run(cmd, cwd=ass.parent)
    return out


def run_manual(clip: GameplayClip, transcript: Transcript, opts: ManualOptions,
               force: bool = False) -> Iterator[dict]:
    """Render the Short. `clip` must already have an imported source and an edited
    transcript. Yields progress events; the last one carries the output path."""
    ensure_ffmpeg()
    src = clip.source_path()
    if src is None:
        raise FriendlyError("No source clip imported for this gameplay clip.")
    if not transcript.words:
        raise FriendlyError(
            "The transcript is empty — run Transcribe (and check the audio has "
            "speech) before building.")

    # Opts-dependent outputs are always rebuilt; the blur-pad is cached unless force.
    for p in (clip.fx_path, clip.captioned_path, clip.final_path, clip.ass_path):
        p.unlink(missing_ok=True)
    if force:
        clip.reframed_path.unlink(missing_ok=True)

    yield {"msg": "1/4  Reframing to 9:16 (blur-pad)..."}
    reframe_mod.reframe(src, clip.reframed_path)
    stage_in = clip.reframed_path

    if opts.effects:
        yield {"msg": f"2/4  Applying effects: {', '.join(opts.effects)} "
                      f"(detecting loud beats)..."}
        _, beats = fx_mod.apply_effects(stage_in, clip.fx_path, opts.effects)
        yield {"msg": f"     {len(beats)} beat(s) detected."}
        stage_in = clip.fx_path
    else:
        yield {"msg": "2/4  Effects: none."}

    yield {"msg": "3/4  Burning captions..."}
    write_captions(transcript, opts, clip.ass_path)
    burn_captions(stage_in, clip.ass_path, clip.captioned_path)
    stage_in = clip.captioned_path

    if opts.overlay_name:
        yield {"msg": f"4/4  Compositing overlay '{opts.overlay_name}'..."}
        from gameplay import overlay as ov_mod
        ov_mod.composite(stage_in, opts.overlay_name, clip.final_path,
                         position=opts.overlay_position,
                         start=opts.overlay_start, duration=opts.overlay_duration)
    else:
        yield {"msg": "4/4  Overlay: none."}
        _run(["ffmpeg", "-y", "-i", str(stage_in), "-c", "copy",
              str(clip.final_path)])

    yield {"msg": f"Done -> {clip.final_path}", "done": True,
           "output": clip.final_path}
