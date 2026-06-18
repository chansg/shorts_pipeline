"""16:9 YouTube export for the full-auto pipeline.

Cuts each chosen highlight window from the source at its NATIVE resolution (no
reframe, crop, or pad) and concatenates them into one landscape YouTube video. This
is the full-auto counterpart to the manual 9:16 Shorts backend — and deliberately
shares none of it (no blur-pad, no like/subscribe overlay, no karaoke captions).

One ffmpeg cut per window + a concat-demuxer join, reusing the lore pipeline's
runner (modules.assemble._run) so behaviour matches the rest of the project.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from modules.assemble import _run

Progress = Callable[[str], None]


def cut_segment(video: str | Path, start: float, end: float, out: Path,
                audio_graph: str | None = None) -> Path:
    """Cut [start, end] at the source's NATIVE resolution (no scale/crop/pad), CFR so
    a later concat is seamless, keeping the first video+audio and dropping any stray
    data/timecode track. Uses the shared quality-targeted encode (the cut is full-
    auto's only/own quality-governing pass; the concat is a stream copy).

    `audio_graph` (from gameplay.censor, spans rebased to this window) censors the
    audio in the same pass — full-auto has no captions, so this is audio-only."""
    from gameplay import encode as enc
    cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(video),
           "-dn", "-map_metadata", "-1", "-fps_mode", "cfr", *enc.final_args()]
    if audio_graph:
        cmd += ["-filter_complex", audio_graph, "-map", "0:v:0", "-map", "[a]"]
    else:
        cmd += ["-map", "0:v:0", "-map", "0:a:0?"]
    cmd += ["-c:a", "aac", "-b:a", "192k", str(out)]
    _run(cmd)
    return out


def _window_audio_graph(censor_spans, cstart: float, cend: float):
    """Rebase global censor spans into a cut window [cstart,cend] (local 0-based) and
    build the audio filtergraph for that segment, or None if no hit falls inside."""
    if not censor_spans:
        return None
    from gameplay import censor as cmod
    local = [(max(gs, cstart) - cstart, min(ge, cend) - cstart)
             for gs, ge in censor_spans if ge > cstart and gs < cend]
    return cmod.audio_graph(local, max(0.0, cend - cstart)) if local else None


def export_youtube(video: str | Path, candidates, out_path: str | Path,
                   progress: Progress | None = None, censor_spans=None) -> Path:
    """Assemble the candidate windows into one 16:9 YouTube video at native
    resolution. `candidates` are objects/tuples exposing .start/.end (or [0]/[1]).
    `censor_spans` (global `(start,end)` profanity hits) are bleeped per window.
    Returns `out_path`."""
    video, out_path = Path(video), Path(out_path)
    emit = (lambda m: progress(m)) if progress else (lambda m: None)
    work = out_path.parent
    work.mkdir(parents=True, exist_ok=True)

    spans: list[tuple[float, float]] = []
    for c in candidates or []:
        start = getattr(c, "start", None)
        end = getattr(c, "end", None)
        if start is None or end is None:        # tolerate (start, end, ...) tuples
            start, end = c[0], c[1]
        spans.append((float(start), float(end)))
    if not spans:
        raise ValueError("No candidates to export.")

    segs: list[Path] = []
    for i, (start, end) in enumerate(spans):
        emit(f"Cutting segment {i + 1}/{len(spans)} ({start:.0f}-{end:.0f}s)...")
        ag = _window_audio_graph(censor_spans, start, end)
        segs.append(cut_segment(video, start, end, work / f"_yt_seg_{i:03d}.mp4",
                                audio_graph=ag))

    if len(segs) == 1:
        emit("Single segment — finalising 16:9 video...")
        _run(["ffmpeg", "-y", "-i", str(segs[0]), "-c", "copy", str(out_path)])
    else:
        emit(f"Concatenating {len(segs)} segments into the 16:9 video...")
        listf = work / "_yt_concat.txt"
        listf.write_text("".join(f"file '{s.resolve().as_posix()}'\n" for s in segs),
                         encoding="utf-8")
        _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listf),
              "-c", "copy", str(out_path)])

    # TODO(16:9 captions): full-auto intentionally ships the YouTube cut WITHOUT the
    # 9:16 karaoke caption layer (a Shorts aesthetic). If landscape captions are
    # wanted later, burn a 16:9-appropriate .ass here from the per-window transcript
    # (fullauto.pipeline.slice_transcript) — keep it optional, do not reuse the Shorts
    # caption style by default.
    emit(f"Done -> {out_path}")
    return out_path
