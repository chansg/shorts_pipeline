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
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from modules.assemble import _run, _has_audio, _probe_duration
from modules.karaoke_captions import CaptionStyle, build_ass, wrap_lines
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
    # Player captions on/off. Off = no player captions burned and NO transcript needed
    # (a narrated hook, if enabled, still shows its own caption).
    captions_enabled: bool = gconf.CAPTION_ON_BY_DEFAULT
    # Caption look — defaults reproduce the original style exactly.
    caption_size: int = gconf.CAPTION_FONTSIZE
    caption_outline: float = gconf.CAPTION_OUTLINE
    caption_uppercase: bool = gconf.CAPTION_UPPERCASE
    caption_animate: bool = gconf.CAPTION_ANIMATE           # pop-in scale animation
    caption_by_player: bool = gconf.CAPTION_BY_PLAYER       # colour each line by player
    caption_color: str = gconf.CAPTION_SINGLE_COLOR         # used when not by player


def _censor_audio(opts: "ManualOptions") -> bool:
    return opts.censor in ("both", "audio")


def _censor_caption(opts: "ManualOptions") -> bool:
    return opts.censor in ("both", "caption")


def _hook_active(opts: "ManualOptions") -> bool:
    return bool(opts.hook_enabled and (opts.hook_text or "").strip())


# ---- fitting caption lines to the frame -------------------------------------
# libass sizes a font so its OS/2 (winAscent + winDescent) spans the style's Fontsize,
# so one em is Fontsize * unitsPerEm / (winAscent + winDescent). Lines are measured
# with the real font file at that em size, so wrapping matches what libass draws
# (checked against burned frames: Anton 'UNBELIEVABLE' at 150 is 444px either way).
_FONT_EM_SCALE = {"Anton": 2048 / 3550, "Arial": 2048 / 2288}   # Arial: the Bold face
_DEFAULT_EM_SCALE = 0.9
_FONT_FILES = {"Anton": ["Anton-Regular.ttf"],
               # the caption style is bold; Liberation Sans is metric-compatible
               "Arial": ["arialbd.ttf", "Arial Bold.ttf", "Arial_Bold.ttf",
                         "LiberationSans-Bold.ttf"]}
_POP_OVERSHOOT = 1.12            # the pop-in animation peaks at 112% scale
_MEASURE_EM = 1000               # measure once at a big size, then scale linearly
_PROBE_EM = 48                   # small size for the glyph-coverage check
# A character the caption font doesn't have (emoji, CJK, …) is drawn by libass with a
# system fallback font — far wider than the font's own .notdef box that Pillow would
# measure. Budget a full Fontsize for each one (measured: ~0.8 for Korean, more for
# emoji).
_MISSING_GLYPH_W = 1.0
_NOTDEF_PROBE = "\U0010FFFD"     # a private-use code point no caption font covers
_measure_lock = threading.RLock()  # FreeType faces aren't safe to share across threads

# Fallback when the font file can't be found: a conservative average width of one
# character, as a fraction of Fontsize (90th percentile over common words, measured
# from burned frames). Arial lowercase is much narrower than its capitals; Anton's
# lowercase is about as wide as its capitals.
_CHAR_W = {("Anton", True): 0.30, ("Anton", False): 0.31,
           ("Arial", True): 0.67, ("Arial", False): 0.55}
_WIDE_CHAR_W = 0.67


def _font_dirs() -> list[Path]:
    dirs = [Path(gconf.FONTS_DIR)]
    if os.environ.get("WINDIR"):
        dirs.append(Path(os.environ["WINDIR"]) / "Fonts")
    dirs += [Path("/Library/Fonts"), Path("/System/Library/Fonts/Supplemental"),
             Path("/usr/share/fonts/truetype/msttcorefonts"),
             Path("/usr/share/fonts/truetype/liberation"),
             Path("/usr/share/fonts/truetype/liberation2")]
    return dirs


@lru_cache(maxsize=None)
def _load_font(font: str, em: int):
    """The caption font at `em` px/em, or None when its file can't be found (wrapping
    then falls back to the per-character estimate)."""
    try:
        from PIL import ImageFont
    except ImportError:                          # pragma: no cover - pillow is a dep
        return None
    for d in _font_dirs():
        for name in _FONT_FILES.get(font, [f"{font}.ttf"]):
            p = d / name
            if p.is_file():
                try:
                    return ImageFont.truetype(str(p), em)
                except OSError:
                    continue
    return None


def _measure_font(font: str):
    return _load_font(font, _MEASURE_EM)


def _glyph_sig(font: str, ch: str):
    with _measure_lock:
        m = _load_font(font, _PROBE_EM).getmask(ch)
        return m.size, bytes(m)


@lru_cache(maxsize=4096)
def _has_glyph(font: str, ch: str) -> bool:
    """Does the caption font itself draw `ch`? A character it lacks renders as the
    font's .notdef box, so compare against that (no font-table parsing needed)."""
    if ch.isspace():
        return True
    return _glyph_sig(font, ch) != _glyph_sig(font, _NOTDEF_PROBE)


@lru_cache(maxsize=8192)
def _advance(font: str, text: str) -> tuple[float, int]:
    """(advance of the characters the font has, at _MEASURE_EM; how many it lacks)."""
    covered = "".join(ch for ch in text if _has_glyph(font, ch))
    with _measure_lock:
        adv = float(_measure_font(font).getlength(covered)) if covered else 0.0
    return adv, len(text) - len(covered)


def line_max_px(animate: bool) -> float:
    """The widest a caption line may render. A static line keeps ~5% clear of each frame
    edge; an animated one is measured at the pop-in's 112% peak (it's only that big for
    ~0.1 s), which just has to stay inside the frame (~1% clear) — so a word that sits
    comfortably at rest isn't split just for the instant it overshoots."""
    return gconf.WIDTH * (0.98 if animate else 0.9)


def caption_text_px(font: str, size: float, *, outline: float, shadow: float,
                    animate: bool):
    """-> text_px(line): the widest a caption line renders, in frame pixels — the glyph
    advance at libass's em size (a character the font lacks counts a full Fontsize, for
    the wider fallback font libass draws it with), at the pop-in's 112% peak when
    animated, plus the outline on both sides and the shadow offset. None if the font
    file isn't found."""
    if _measure_font(font) is None:
        return None
    scale = _POP_OVERSHOOT if animate else 1.0
    k = float(size) * _FONT_EM_SCALE.get(font, _DEFAULT_EM_SCALE) / _MEASURE_EM * scale
    missing_px = float(size) * _MISSING_GLYPH_W * scale
    pad = 2 * float(outline) + float(shadow)

    def text_px(text: str) -> float:
        adv, missing = _advance(font, text)
        return adv * k + missing * missing_px + pad
    return text_px


def line_chars(font: str, size: int | float, *, uppercase: bool = True,
               animate: bool = True, outline: float = 0.0,
               shadow: float = 3.0) -> int:
    """Max characters per line when the font file can't be measured (the estimate
    fallback). Capped at CAPTION_MAX_LINE_CHARS (12) — exactly the old value for the
    default Anton at 150px, so an untouched build wraps identically; a big wide font
    wraps sooner instead of running off the frame edges."""
    per = _CHAR_W.get((font, bool(uppercase)), _WIDE_CHAR_W) * max(1.0, float(size))
    per *= _POP_OVERSHOOT if animate else 1.0
    usable = line_max_px(animate) - 2 * float(outline) - float(shadow)
    return max(4, min(gconf.CAPTION_MAX_LINE_CHARS, int(usable // per)))


def max_caption_lines(st: CaptionStyle) -> int:
    """How many lines one cue can have and still fit on screen. The block is centred on
    its height (\\an5), so it can reach min(y, H - y) each way; each line box is at
    most Fontsize tall (taller than the letters), x1.12 at the pop-in's peak."""
    y = st.play_h * st.pos_y_frac
    room = 2 * min(y, st.play_h - y) - 2 * st.outline
    line_h = st.fontsize * (_POP_OVERSHOOT if st.animate else 1.0)
    return max(1, int(room // max(1.0, line_h)))


def _fit_tokens(st: CaptionStyle, text: str) -> list[str]:
    """The cue's words, with any word too wide for one line replaced by the pieces the
    renderer would hard-split it into — so even a single long word can be spread over
    back-to-back cues when its pieces don't fit on screen together."""
    out: list[str] = []
    for w in text.split():
        pieces = wrap_lines(st, w)
        out.extend(pieces if len(pieces) > 1 else [w])
    return out


def fit_cues_to_height(tuples: list, st: CaptionStyle) -> list:
    """Split any cue that would wrap to more lines than fit on screen (a big font low
    down) into back-to-back cues that each fit, sharing its time span by length. A cue
    that already fits is passed through untouched."""
    max_lines = max_caption_lines(st)
    out = []
    for t in tuples:
        text = str(t[0])
        if len(wrap_lines(st, text)) <= max_lines:
            out.append(t)
            continue
        groups: list[list[str]] = []
        cur: list[str] = []
        for w in _fit_tokens(st, text):
            if cur and len(wrap_lines(st, " ".join(cur + [w]))) > max_lines:
                groups.append(cur)
                cur = [w]
            else:
                cur = cur + [w]
        if cur:
            groups.append(cur)
        start, end = float(t[1]), float(t[2])
        total = sum(len(" ".join(g)) for g in groups) or 1
        at = start
        for i, g in enumerate(groups):
            seg = " ".join(g)
            stop = end if i == len(groups) - 1 else at + (end - start) * len(seg) / total
            out.append((seg, at, stop, *t[3:]))
            at = stop
    return out


def caption_style(opts: ManualOptions) -> CaptionStyle:
    # The narrated hook caption gets a reserved colour so it reads as separate from
    # in-game speech; merged with any per-speaker colours the user picked.
    speaker_colors = dict(opts.speaker_colors or {})
    speaker_colors.setdefault("NARRATOR", tuple(gconf.NARRATOR_CAPTION_COLOR))
    extra = {}
    if not opts.caption_by_player:
        # one colour for every player line (write_captions drops the player label, so
        # the style fill applies); the hook keeps its NARRATOR colour.
        from gameplay.players import to_rgb
        extra["fill"] = to_rgb(opts.caption_color) or (255, 255, 255)
    st = CaptionStyle(
        font=opts.caption_font,
        fontsize=int(opts.caption_size),
        outline=float(opts.caption_outline),
        uppercase=bool(opts.caption_uppercase),
        animate=bool(opts.caption_animate),
        **extra,
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
    # Fit lines to the frame: measured with the real font when it can be found (a word
    # is then only split when it truly can't fit), else the per-character estimate.
    text_px = caption_text_px(st.font, st.fontsize, outline=st.outline,
                              shadow=st.shadow, animate=st.animate)
    if text_px is not None:
        st.text_px, st.max_line_px = text_px, line_max_px(st.animate)
    else:
        st.max_line_chars = line_chars(st.font, st.fontsize, uppercase=st.uppercase,
                                       animate=st.animate, outline=st.outline,
                                       shadow=st.shadow)
    return st


def write_captions(transcript: Transcript, opts: ManualOptions, ass_path: Path,
                   hook: tuple | None = None) -> Path:
    """Build the .ass for the gameplay caption mode. "phrase" groups a few words per
    cue (forgiving of ASR drift); "word" keeps one-word karaoke. A global
    `caption_offset` nudges all transcript cues. `hook=(text, dur, lead)` prepends the
    NARRATOR-coloured hook words (placed at t=0, not offset). With player captions
    off, only the hook (if any) is written."""
    from gameplay import captioning
    if opts.captions_enabled:
        tuples = captioning.caption_cues(
            transcript.words, mode=opts.caption_mode, offset=opts.caption_offset,
            mask=_censor_caption(opts), mask_style=gconf.CENSOR_CAPTION_STYLE)
        if not opts.caption_by_player:
            # drop the player label -> the renderer uses the single style fill
            tuples = [tuple(t[:3]) for t in tuples]
    else:
        tuples = []
    if hook:
        from gameplay.hook import hook_caption_tuples
        text, dur, lead = hook
        hook_end = lead + dur
        # The game bed is muted+muffled while she speaks, so its speech isn't audible —
        # drop any game-speech cue that overlaps the narration window so it doesn't
        # render ON TOP of the hook caption (the reported overlapping characters).
        tuples = [t for t in tuples if not (t[2] > lead and t[1] < hook_end)]
        tuples = hook_caption_tuples(text, dur, lead) + list(tuples)
    st = caption_style(opts)
    ass_path.write_text(build_ass(fit_cues_to_height(tuples, st), st), encoding="utf-8")
    return ass_path


def burn_captions(video: Path, ass: Path | None, out: Path, *,
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
    (e.g. ["-i", hook_wav]) are extra `-i` inputs the graph references as [1:a], [2:a]…

    `ass=None` burns no caption track (player captions off): the same single final
    encode still applies effects, censor and hook audio."""
    video, out = video.resolve(), out.resolve()
    vparts = [effects_vf] if effects_vf else []
    cwd = None
    if ass is not None:
        fonts_rel = os.path.relpath(gconf.FONTS_DIR, ass.parent).replace("\\", "/")
        vparts.append(f"ass={ass.name}:fontsdir={fonts_rel}")
        cwd = ass.parent
    vchain = ",".join(vparts) or "null"
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
    _run(cmd, cwd=cwd)
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
    if not opts.captions_enabled:
        raise FriendlyError("Player captions are off — turn them on to preview them.")
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
    """Render the Short. `clip` must already have an imported source. With player
    captions on it also needs an edited transcript; with them off the transcript is
    optional (if present it still drives the audio censor). Yields progress events;
    the last one carries the output path."""
    ensure_ffmpeg()
    src = clip.source_path()
    if src is None:
        raise FriendlyError("No source clip imported for this gameplay clip.")
    if opts.captions_enabled and not transcript.words:
        raise FriendlyError(
            "The transcript is empty — run Transcribe (and check the audio has "
            "speech) before building, or turn player captions off.")

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
        cap_word = "captions" if opts.captions_enabled else "no captions"
        yield {"msg": f"2/3  Effects {', '.join(opts.effects)} + {cap_word} "
                      f"({len(beats)} beat(s)) — one encode..."}
    elif opts.captions_enabled:
        yield {"msg": "2/3  Burning captions (effects: none)..."}
    else:
        yield {"msg": "2/3  Final encode — player captions off (effects: none)..."}

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
                          f"(game audio muted + muffled, captions paused until she "
                          f"finishes)."}
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

    # A caption track exists when player captions are on, or a hook line needs its
    # own caption; otherwise the final encode carries no subtitles at all.
    ass = None
    if opts.captions_enabled or cap_hook:
        ass = write_captions(transcript, opts, clip.ass_path, hook=cap_hook)
    # The caption burn is the FINAL encode unless an overlay pass still follows.
    cap_out = clip.captioned_path if has_overlay else clip.final_path
    burn_captions(clip.reframed_path, ass, cap_out,
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
