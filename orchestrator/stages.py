"""Stage runners: thin wrappers over the existing modules (wrap, don't rewrite).

i2v half:  generate_stills / animate  -> i2v.imagegen / i2v.pipeline + VeoProvider
handoff:   episodes/<name>/{stills,clips} -> assets/images as NN.png / NN.mp4
shorts half: voice_assemble -> modules.tts / captions / visuals / assemble

Everything is idempotent: existing outputs are skipped unless force=True, so a
re-run only fills gaps (this is what protects Veo spend).
"""
from __future__ import annotations

import datetime as _dt
import shutil
from pathlib import Path
from typing import Iterator

import config
from i2v import config as i2v_config
from i2v import imagegen
from i2v import pipeline as i2v_pipeline
from modules.visuals import VALID_EXT
from orchestrator import state as st
from orchestrator.errors import FriendlyError, ensure_ffmpeg, friendly

# Make sure the shared .env is loaded even when only this module is imported.
i2v_config.load_dotenv()


# --------------------------------------------------------------- stills ----

def generate_stills(ep: st.Episode, only: list[str] | None = None,
                    force: bool = False) -> list[dict]:
    """Run Nano Banana 2 for the episode (or a subset of scene names).
    Returns imagegen's per-scene results: generated / skipped / failed."""
    try:
        api_key = i2v_config.resolve_api_key()
        return imagegen.run(
            api_key=api_key,
            manifest_path=ep.manifest_path,
            images_dir=ep.stills_dir,
            refs_dir=st.REFS_DIR,
            only=only,
            force=force,
        )
    except Exception as exc:
        raise friendly(exc) from exc


# --------------------------------------------------------------- animate ----

def animate(ep: st.Episode, only: list[str], force: bool = False,
            model: str | None = None) -> list:
    """Render the given scene names with Veo. Idempotent via i2v.pipeline:
    an already-rendered clip is skipped (never re-billed) unless force=True."""
    if not only:
        return []
    try:
        from i2v.providers.veo import DEFAULT_MODEL, VeoProvider
        api_key = i2v_config.resolve_api_key()
        manifest = i2v_config.Manifest.load(ep.manifest_path)
        clips = manifest.select(only)
        if not clips:
            raise FriendlyError(f"No manifest scenes matched: {only}")
        provider = VeoProvider(api_key=api_key, model=model or DEFAULT_MODEL)
        return i2v_pipeline.run(
            provider=provider, clips=clips,
            images_dir=ep.stills_dir, out_dir=ep.clips_dir,
            force=force,
        )
    except Exception as exc:
        raise friendly(exc) from exc


# --------------------------------------------------------------- handoff ----

def handoff(ep: st.Episode) -> tuple[list[str], list[str]]:
    """Place this episode's media into assets/images as NN.mp4 / NN.png in
    manifest order — the step that used to be a manual rename-and-copy.

    Any media already in assets/images (e.g. a previous episode) is moved to
    assets/images/_prev_<timestamp>/ rather than deleted.
    Returns (placed_names, warnings)."""
    pairs = ep.expected_assets()
    if not pairs:
        raise FriendlyError("No manifest yet — generate prompts first.")

    warnings: list[str] = []
    for c in ep.clips():
        if not ep.still_path(c).exists():
            raise FriendlyError(
                f"Scene still missing: {ep.still_path(c).name}. Generate stills "
                "for every scene before the handoff."
            )
        if c.get("animate") and not ep.clip_path(c).exists():
            warnings.append(f"{c['name']}: marked animate but no Veo clip on "
                            "disk — handing off the still instead.")

    # Archive whatever is currently in assets/images (don't destroy it).
    existing = [p for p in config.IMAGES_DIR.iterdir()
                if p.suffix.lower() in VALID_EXT]
    expected = {dest.name: src for src, dest in pairs}
    stale = [p for p in existing if p.name not in expected]
    if stale:
        prev = config.IMAGES_DIR / f"_prev_{_dt.datetime.now():%Y%m%d_%H%M%S}"
        prev.mkdir(parents=True, exist_ok=True)
        for p in stale:
            shutil.move(str(p), str(prev / p.name))
        warnings.append(f"Moved {len(stale)} unrelated file(s) from assets/images "
                        f"to {prev.name}/ (nothing was deleted).")

    placed = []
    for src, dest in pairs:
        # NN.png and NN.mp4 must not coexist — the pipeline would pick up both.
        for twin in (dest.with_suffix(".png"), dest.with_suffix(".mp4")):
            if twin.exists() and twin.name != dest.name:
                twin.unlink()
        shutil.copy2(src, dest)
        placed.append(dest.name)
    # Renders that predate fingerprint recording can't prove they match the
    # new media — mark them stale so the build step re-runs instead of skipping.
    if ep.load_state().get("rendered_assets") is None:
        ep.save_state(rendered_assets=[])
    return placed, warnings


# ------------------------------------------------------ voice + assemble ----

def apply_voice_settings(voice_id: str, stability: float, similarity: float,
                         style: float, model: str | None = None) -> None:
    """The modules read config at call time, so overriding the module globals
    is the supported way to change voice settings per run."""
    config.ELEVENLABS_VOICE_ID = voice_id.strip() or config.ELEVENLABS_VOICE_ID
    config.ELEVENLABS_STABILITY = float(stability)
    config.ELEVENLABS_SIMILARITY = float(similarity)
    config.ELEVENLABS_STYLE = float(style)
    if model:
        config.ELEVENLABS_MODEL = model


def voice_assemble(ep: st.Episode, music_path: Path | None,
                   force: bool = False) -> Iterator[dict]:
    """TTS -> captions -> scenes -> assemble, mirroring pipeline.py's recipe
    (minus cutaways) with progress events the GUI can stream.

    Yields {"msg": str} progress events, then a final
    {"done": True, "output": Path, "timeline": [...], "notes": [...]}.
    The TTS result is cached: if the voice wav exists and the script hasn't
    changed, ElevenLabs is not called again (no re-billing) unless force.
    """
    from modules import assemble as asm_mod
    from modules import captions as cap_mod
    from modules import tts as tts_mod
    from modules import visuals as vis_mod
    from pipeline import get_audio_duration

    try:
        ensure_ffmpeg()
        text = ep.script_text()
        if not text:
            raise FriendlyError(f"Script is empty: {ep.script_path}")
        s_hash = st.script_hash(text)
        work = config.WORK_DIR
        voice_wav = work / f"{ep.name}_voice.wav"

        state = ep.load_state()
        if (voice_wav.exists() and not force
                and state.get("voice_script_hash") == s_hash
                and state.get("voice_settings") == _voice_snapshot()):
            yield {"msg": "Voiceover unchanged — reusing existing audio "
                          "(ElevenLabs not billed)."}
        else:
            yield {"msg": "Synthesizing voiceover (ElevenLabs)..."}
            tts_mod.synthesize(text, voice_wav)
            ep.save_state(voice_script_hash=s_hash,
                          voice_settings=_voice_snapshot())
        duration = get_audio_duration(voice_wav)
        yield {"msg": f"Voiceover ready: {duration:.1f}s of narration."}

        yield {"msg": "Transcribing word timings (Whisper — local, may take a "
                      "minute on first run while the model downloads)..."}
        words = cap_mod.transcribe_words(voice_wav)
        aligned, _exact = cap_mod.align_script_words(words, text)
        ass = cap_mod.write_ass(aligned, work / f"{ep.name}.ass")
        yield {"msg": f"Captions written ({len(aligned)} words, "
                      f"{config.CAPTION_MAX_WORDS} per line)."}

        images = vis_mod.collect_images()
        scenes = vis_mod.build_scenes(images, duration, words=aligned,
                                      script_text=text)
        n_sent = vis_mod.count_sentences(words, duration, script_text=text)
        if len(images) != n_sent:
            yield {"msg": f"Note: {len(images)} media files for {n_sent} "
                          "sentences — scenes will cover multiple sentences."}

        # Parse the optional SFX/audio layer from the manifest (clear errors if
        # malformed). Absent audio data -> empty spec -> legacy mix path.
        from orchestrator.audio_spec import parse_audio_spec
        audio_spec = parse_audio_spec(ep.load_manifest() or {})
        if not audio_spec.is_empty():
            yield {"msg": f"SFX layer: {len(audio_spec.all_cues())} cue(s) "
                          f"(ambient/motif/one-shot) will be mixed in."}

        yield {"msg": f"Assembling {len(scenes)} scenes (Ken Burns + clips, "
                      f"{config.TRANSITION_SEC}s crossfades)... this is the "
                      "long ffmpeg step."}
        music = music_path if music_path and Path(music_path).exists() else None
        notes = asm_mod.assemble(scenes, voice_wav, ass, ep.output_path, music,
                                 voice_duration=duration,
                                 audio_spec=audio_spec, words=aligned)

        timeline, t = [], 0.0
        for sc in scenes:
            timeline.append({"file": sc.image.name, "start": round(t, 2),
                             "duration": round(sc.duration, 2)})
            t += sc.duration
        ep.save_state(rendered_script_hash=s_hash, scene_timeline=timeline,
                      rendered_assets=st.assets_fingerprint(),
                      music=str(music) if music else "", approved=False)
        yield {"done": True, "output": ep.output_path, "timeline": timeline,
               "notes": notes}
    except FriendlyError:
        raise
    except Exception as exc:
        raise friendly(exc) from exc


def _voice_snapshot() -> dict:
    return {
        "voice_id": config.ELEVENLABS_VOICE_ID,
        "model": config.ELEVENLABS_MODEL,
        "stability": config.ELEVENLABS_STABILITY,
        "similarity": config.ELEVENLABS_SIMILARITY,
        "style": config.ELEVENLABS_STYLE,
    }


def list_voices() -> list[tuple[str, str]]:
    """(label, voice_id) pairs from the ElevenLabs account."""
    import os
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise FriendlyError("ELEVENLABS_API_KEY is not set — add it in Settings "
                            "or .env to list voices.")
    try:
        from elevenlabs.client import ElevenLabs
        voices = ElevenLabs(api_key=key).voices.get_all().voices
        return [(f"{v.name} ({v.voice_id})", v.voice_id) for v in voices]
    except Exception as exc:
        raise friendly(exc) from exc


def list_music() -> list[str]:
    exts = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
    return sorted(str(p) for p in config.MUSIC_DIR.iterdir()
                  if p.suffix.lower() in exts)
