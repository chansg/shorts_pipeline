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

from modules.assemble import _run, _has_audio, _probe_duration
from modules.karaoke_captions import CaptionStyle, build_ass
from orchestrator.errors import FriendlyError, ensure_ffmpeg
from gameplay import config as gconf
from gameplay import censor as censor_mod
from gameplay import effects as fx_mod
from gameplay import encode as enc
from gameplay import hook as hook_mod
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
    reframe_mode: str = gconf.REFRAME_MODE                  # fill | fit_crop | blur_pad | zoom_blur
    crop_x_offset: float = gconf.REFRAME_CROP_X_OFFSET      # fill: horizontal crop bias (0..1)
    crop_y_offset: float = gconf.REFRAME_CROP_Y_OFFSET      # fill: vertical crop bias (0..1)
    fill_fraction: float = gconf.REFRAME_FILL_FRACTION      # fill: zoom past cover (>=1.0)
    censor: str = "both"          # "both" | "audio" | "caption" | "off"
    censor_audio_mode: str = gconf.CENSOR_AUDIO_MODE        # bleep | mute | duck
    hook_enabled: bool = False    # narrate an opening hook line over the start
    hook_text: str = ""           # the hook line (read aloud + captioned in NARRATOR colour)
    hook_voice: str = ""          # ElevenLabs voice id (blank = gconf.HOOK_VOICE)
    caption_mode: str = gconf.CAPTION_CHUNK_MODE            # "phrase" (default) | "word"
    caption_offset: float = gconf.CAPTION_OFFSET_S          # global lead(-)/lag(+) nudge (s)


def _censor_audio(opts: "ManualOptions") -> bool:
    return opts.censor in ("both", "audio")


def _censor_caption(opts: "ManualOptions") -> bool:
    return opts.censor in ("both", "caption")


def _hook_active(opts: "ManualOptions") -> bool:
    return bool(opts.hook_enabled and (opts.hook_text or "").strip())


def caption_style(opts: ManualOptions) -> CaptionStyle:
    # The narrated hook caption gets a reserved colour so it reads as separate from
    # in-game speech; merged with any per-speaker colours the user picked.
    speaker_colors = dict(opts.speaker_colors or {})
    speaker_colors.setdefault("NARRATOR", tuple(gconf.NARRATOR_CAPTION_COLOR))
    return CaptionStyle(
        font=opts.caption_font,
        fontsize=gconf.CAPTION_FONTSIZE,
        play_w=gconf.WIDTH,
        play_h=gconf.HEIGHT,
        pos_y_frac=opts.caption_pos_y_frac,
        words_per_cue=1,            # cues are pre-grouped by gameplay.captioning
        gap_fill=True,
        max_gap=gconf.CAPTION_MAX_GAP_S,   # bridge small gaps; hold briefly across big ones
        hold=0.4,
        min_hold_s=gconf.CAPTION_MIN_DUR_S,  # minimum on-screen time (anti-flash)
        speaker_colors=speaker_colors,
        # gameplay defence in depth (noisy ASR): never wall the screen or overlap
        max_event_s=gconf.CAPTION_MAX_EVENT_S,
        max_line_chars=gconf.CAPTION_MAX_LINE_CHARS,
        prevent_overlap=True,
    )


def write_captions(transcript: Transcript, opts: ManualOptions, ass_path: Path,
                   hook: tuple | None = None) -> Path:
    """Build the .ass for the gameplay caption mode. "phrase" groups a few words per
    cue (forgiving of ASR drift); "word" keeps one-word karaoke. A global
    `caption_offset` nudges all transcript cues. `hook=(text, dur, lead)` prepends the
    NARRATOR-coloured hook words (placed at t=0, not offset)."""
    from gameplay import captioning
    tuples = captioning.caption_cues(
        transcript.words, mode=opts.caption_mode, offset=opts.caption_offset,
        mask=_censor_caption(opts), mask_style=gconf.CENSOR_CAPTION_STYLE)
    if hook:
        from gameplay.hook import hook_caption_tuples
        text, dur, lead = hook
        tuples = hook_caption_tuples(text, dur, lead) + list(tuples)
    ass_path.write_text(build_ass(tuples, caption_style(opts)), encoding="utf-8")
    return ass_path


def burn_captions(video: Path, ass: Path, out: Path, *,
                  effects_vf: str | None = None, final: bool = True,
                  audio_graph: str | None = None,
                  audio_inputs: list[str] | None = None) -> Path:
    """Burn the .ass onto `video` (optionally chaining an `effects_vf` zoompan filter
    BEFORE it, so effects + captions are ONE encode), keeping the clip's own audio.

    Reuses the lore pipeline's fontsdir trick: run from the .ass folder, pass the bare
    filename, and give fontsdir as a relative (colon-free) path so the bundled font
    resolves on Windows without a system install. `final=True` uses the quality-
    targeted final encode (CRF 18 + profile high + faststart); `final=False` uses a
    near-lossless intermediate (when an overlay pass still follows).

    `audio_graph` (censor and/or narrated-hook duck/mix) processes the audio in this
    SAME pass — no extra encode. When set, video + audio are composed in one
    filter_complex; otherwise the audio is stream-copied (lossless). `audio_inputs`
    (e.g. ["-i", hook_wav]) are extra `-i` inputs the graph references as [1:a], [2:a]…"""
    video, out = video.resolve(), out.resolve()
    fonts_rel = os.path.relpath(gconf.FONTS_DIR, ass.parent).replace("\\", "/")
    vchain = ",".join(([effects_vf] if effects_vf else [])
                      + [f"ass={ass.name}:fontsdir={fonts_rel}"])
    enc_args = enc.final_args() if final else enc.intermediate_args()
    if audio_graph and _has_audio(video):
        cmd = ["ffmpeg", "-y", "-i", str(video), *(audio_inputs or []),
               "-filter_complex", f"[0:v]{vchain}[v];{audio_graph}",
               "-map", "[v]", "-map", "[a]",
               *enc_args, "-c:a", "aac", "-b:a", "192k", str(out)]
    else:
        cmd = ["ffmpeg", "-y", "-i", str(video), "-vf", vchain, "-map", "0:v"]
        if _has_audio(video):
            cmd += ["-map", "0:a", "-c:a", "copy"]
        cmd += [*enc_args, str(out)]
    _run(cmd, cwd=ass.parent)
    return out


def preview_captions(clip: GameplayClip, transcript: Transcript,
                     opts: ManualOptions, seconds: float = 8.0) -> Path:
    """Re-render JUST the caption track onto the first `seconds` of the reframed
    clip — a fast 'did my edits land?' preview that never re-runs transcription.
    Reuses the cached blur-pad (no effects/overlay). Returns the preview mp4."""
    ensure_ffmpeg()
    src = clip.source_path()
    if src is None:
        raise FriendlyError("No source clip for this gameplay clip.")
    if not transcript.words:
        raise FriendlyError("The transcript is empty — nothing to preview.")
    reframe_mod.reframe(src, clip.reframed_path)          # cached
    preview_src = clip.dir / "_preview_src.mp4"
    preview_ass = clip.dir / "_preview.ass"
    preview_out = clip.dir / "_preview.mp4"
    for p in (preview_src, preview_out):
        p.unlink(missing_ok=True)
    # trim to the first N seconds; absolute caption times in the .ass still line up
    _run(["ffmpeg", "-y", "-t", f"{seconds:.3f}", "-i", str(clip.reframed_path),
          "-c", "copy", str(preview_src)])
    write_captions(transcript, opts, preview_ass)
    burn_captions(preview_src, preview_ass, preview_out, final=False)  # fast preview
    return preview_out


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

    # Opts-dependent outputs are always rebuilt; the reframe is cached unless force.
    for p in (clip.fx_path, clip.captioned_path, clip.final_path, clip.ass_path):
        p.unlink(missing_ok=True)
    if force:
        clip.reframed_path.unlink(missing_ok=True)

    has_overlay = bool(opts.overlay_name)
    # Encode passes: the cached reframe (near-lossless intermediate) + ONE final
    # encode (effects+captions composed), + an overlay encode only if an overlay is
    # used. Down from up to four (reframe/effects/captions/overlay each re-encoded).
    n_passes = 2 + (1 if has_overlay else 0)

    yield {"msg": f"1/3  Reframing to 9:16 ({opts.reframe_mode})..."}
    reframe_mod.reframe(src, clip.reframed_path, mode=opts.reframe_mode,
                        x_off=opts.crop_x_offset, y_off=opts.crop_y_offset,
                        fill_frac=opts.fill_fraction)

    # Effects (zoompan) are composed into the caption burn as ONE pass, so loud-beat
    # detection runs here but no separate effects encode happens.
    effects_vf = None
    if opts.effects:
        beats = fx_mod.detect_beats(src)
        effects_vf = fx_mod.build_effects_filter(beats, opts.effects,
                                                 gconf.WIDTH, gconf.HEIGHT)
        yield {"msg": f"2/3  Effects {', '.join(opts.effects)} + captions "
                      f"({len(beats)} beat(s)) — one encode..."}
    else:
        yield {"msg": "2/3  Burning captions (effects: none)..."}

    # ---- audio: narrated hook + v3 profanity censor, composed into ONE graph ----
    # (no extra pass — both ride this final encode). Order: censor the bed first,
    # then duck it under the narration and mix. Each piece degrades cleanly to a no-op.
    hook = None        # (wav_path, dur_s) when active AND the TTS succeeded
    if _hook_active(opts):
        yield {"msg": "     Narrated hook: synthesizing voice (ElevenLabs)..."}
        try:
            wav, hdur = hook_mod.synthesize_hook(opts.hook_text,
                                                 opts.hook_voice or None, clip.dir)
            hook = (wav, hdur)
            yield {"msg": f"     Hook: {hdur:.1f}s narration over the opening "
                          f"(game audio ducks to {gconf.DUCK_LEVEL})."}
        except Exception as e:        # noqa: BLE001 — degrade, never fail the render
            yield {"msg": f"     ⚠ Hook TTS failed ({type(e).__name__}: {e}); "
                          f"building WITHOUT narration."}

    clip_dur = _probe_duration(clip.reframed_path)
    bed, parts = "[0:a]", []
    if _censor_audio(opts):
        spans = censor_mod.merge_spans(transcript.censor_spans(),
                                       gconf.CENSOR_PAD_S, clip_dur)
        flagged = sum(1 for w in transcript.words if w.censor)
        if spans:
            cen_out = "[__cen]" if hook else "[a]"
            parts.append(censor_mod.audio_graph(spans, clip_dur,
                                                mode=opts.censor_audio_mode,
                                                src=bed, out=cen_out))
            bed = cen_out
            skipped = flagged - sum(1 for w in transcript.words
                                    if w.censor and w.end > w.start)
            note = f"     Censor: {len(spans)} span(s) [{opts.censor_audio_mode}]"
            if skipped:
                note += f"; {skipped} flagged word(s) had no timestamp (caption-only)"
            yield {"msg": note + "."}
        elif flagged:
            yield {"msg": f"     Censor: {flagged} flagged word(s) had no timestamp "
                          f"(caption masked, audio unchanged)."}
    if hook:
        parts.append(hook_mod.duck_mix_graph(bed, hook[1], lead=gconf.HOOK_LEAD_IN_S))

    audio_graph = ";".join(parts) if parts else None
    audio_inputs = ["-i", str(hook[0])] if hook else None
    cap_hook = (opts.hook_text, hook[1], gconf.HOOK_LEAD_IN_S) if hook else None

    write_captions(transcript, opts, clip.ass_path, hook=cap_hook)
    # The caption burn is the FINAL encode unless an overlay pass still follows.
    cap_out = clip.captioned_path if has_overlay else clip.final_path
    burn_captions(clip.reframed_path, clip.ass_path, cap_out,
                  effects_vf=effects_vf, final=not has_overlay,
                  audio_graph=audio_graph, audio_inputs=audio_inputs)

    if has_overlay:
        yield {"msg": f"3/3  Compositing overlay '{opts.overlay_name}' (final encode)..."}
        from gameplay import overlay as ov_mod
        ov_mod.composite(cap_out, opts.overlay_name, clip.final_path,
                         position=opts.overlay_position,
                         start=opts.overlay_start, duration=opts.overlay_duration)

    yield {"msg": f"Done -> {clip.final_path} "
                  f"(CRF {gconf.OUTPUT_CRF}, {n_passes} encode pass(es)).",
           "done": True, "output": clip.final_path}
