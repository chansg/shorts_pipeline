"""Narrated opening hook for gameplay Shorts — a single line read aloud over the
start of the clip, with the game audio ducked under it (the TikTok "story-time"
format).

Reuses the lore ElevenLabs client (`modules.tts.synthesize`) — no second TTS client —
and the karaoke caption renderer (the hook is just more `(text,start,end,speaker)`
tuples). The TTS is cached by `(text, voice)` so re-builds don't re-bill. Pure helpers
(caption tuples + the ffmpeg audio sub-graph) have no network/ffmpeg dependency, so
they unit-test without ElevenLabs or a GPU.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from gameplay import config as gconf


def _cache_path(text: str, voice_id: str, dest_dir) -> Path:
    h = hashlib.sha1(f"{text}|{voice_id}".encode("utf-8")).hexdigest()[:16]
    return Path(dest_dir) / f"_hook_{h}.wav"


def synthesize_hook(text: str, voice_id: str | None, dest_dir) -> tuple[Path, float]:
    """Generate (and cache) the hook narration wav. Returns (path, duration_s).

    Cached by (text, voice): a repeat call with the same pair reuses the file and does
    NOT call ElevenLabs (no re-bill). Raises on a TTS failure — the caller decides to
    degrade gracefully (the build continues without narration)."""
    from modules.assemble import _probe_duration
    text = (text or "").strip()
    voice_id = voice_id or gconf.HOOK_VOICE
    out = _cache_path(text, voice_id, dest_dir)
    if not out.exists():
        from modules import tts
        tts.synthesize(text, out, voice_id=voice_id)
    return out, _probe_duration(out)


def hook_caption_tuples(text: str, dur: float, lead: float = 0.0,
                        speaker: str = "NARRATOR") -> list[tuple]:
    """Hook words as caption tuples, evenly distributed over [lead, lead+dur].
    ElevenLabs word timings aren't used — even distribution is the agreed fallback."""
    words = (text or "").split()
    if not words or dur <= 0:
        return []
    step = dur / len(words)
    out = []
    for i, w in enumerate(words):
        s = lead + i * step
        out.append((w, round(s, 3), round(s + step, 3), speaker))
    return out


def duck_mix_graph(bed_label: str, hook_dur: float, *, lead: float = 0.0,
                   duck: float | None = None, release: float | None = None,
                   muffle_hz: float | None = None, narr: str = "[1:a]",
                   out: str = "[a]") -> str:
    """ffmpeg audio sub-graph: MUTE + MUFFLE `bed_label` while the narration `narr`
    plays, swell it back after, and mix the narration on top.

    The bed is split into two legs (summed):
      - a DRY leg — full (1.0) BEFORE the line, silent DURING it, then a linear swell
        back to 1.0 over `release` seconds AFTER it;
      - a WET leg — the bed only DURING the line, at `duck` gain through a `muffle_hz`
        low-pass (a dull, quiet murmur), silent otherwise.
    Outside the line the bed is untouched; under it it's muted + muffled so the voice is
    clean. The narration is delayed by `lead` and summed at full level. Returns a graph
    fragment ending in `out`."""
    duck = gconf.HOOK_MUTE_GAIN if duck is None else duck
    release = gconf.DUCK_RELEASE_S if release is None else release
    muffle_hz = gconf.HOOK_MUFFLE_HZ if muffle_hz is None else muffle_hz
    rel = max(0.01, float(release))
    led = float(lead)
    end = led + float(hook_dur)
    # dry: 1 before the line, 0 during it, linear swell back to 1 over `rel` after.
    vol_dry = (f"if(lt(t,{led:.3f}),1,"
               f"if(lt(t,{end:.3f}),0,min(1,(t-{end:.3f})/{rel:.3f})))")
    # wet: `duck` gain only during the line (then low-passed), silent otherwise.
    vol_wet = f"if(lt(t,{led:.3f}),0,if(lt(t,{end:.3f}),{duck},0))"
    narr_chain = f"{narr}aresample=48000,aformat=channel_layouts=stereo"
    delay_ms = int(round(led * 1000))
    if delay_ms > 0:
        narr_chain += f",adelay={delay_ms}|{delay_ms}"
    return (
        f"{bed_label}aresample=48000,aformat=channel_layouts=stereo,asplit=2[__bdry][__bwet];"
        f"[__bdry]volume='{vol_dry}':eval=frame[__bd];"
        f"[__bwet]volume='{vol_wet}':eval=frame,lowpass=f={int(muffle_hz)}[__bw];"
        f"{narr_chain}[__narr];"
        f"[__bd][__bw][__narr]amix=inputs=3:duration=first:normalize=0{out}"
    )
