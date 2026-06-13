"""
Shorts pipeline orchestrator.

Usage:
    python pipeline.py scripts/hollow_king.txt --no-rewrite
    python pipeline.py scripts/hollow_king.txt --music assets/music/Track.mp3

Steps: seed -> (rewrite) -> ElevenLabs TTS -> Whisper captions -> assembly -> mp4
Writes a detailed run log next to the output (output/<name>_log.txt) for review.
"""
from __future__ import annotations
import argparse
import time
import datetime as _dt
from pathlib import Path

import config
from modules import script as script_mod
from modules import tts as tts_mod
from modules import captions as cap_mod
from modules import visuals as vis_mod
from modules import assemble as asm_mod


def get_audio_duration(path: Path) -> float:
    import subprocess, json
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", str(path)],
        capture_output=True, text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def _filesize_kb(path: Path) -> int:
    try:
        return path.stat().st_size // 1024
    except OSError:
        return 0


class RunLog:
    """Collects a human-readable report of the run for later review."""
    def __init__(self, name: str):
        self.name = name
        self.lines: list[str] = []
        self.warnings: list[str] = []
        self.t0 = time.time()

    def add(self, line: str = ""):
        self.lines.append(line)

    def warn(self, line: str):
        self.warnings.append(line)
        self.lines.append(f"  [WARNING] {line}")

    def write(self, path: Path, error: str | None = None):
        header = [
            "=" * 60,
            "Ashen_Chan pipeline — run log",
            "=" * 60,
            f"When:     {_dt.datetime.now().isoformat(timespec='seconds')}",
            f"Script:   {self.name}",
            f"Elapsed:  {time.time() - self.t0:.1f}s",
            f"Status:   {'ERROR' if error else 'OK'}",
            "",
        ]
        body = header + self.lines
        if self.warnings:
            body += ["", "Warnings:"] + [f"  - {w}" for w in self.warnings]
        if error:
            body += ["", "ERROR:", error]
        path.write_text("\n".join(body), encoding="utf-8")


def run(seed_path: str, music: str | None = None, no_rewrite: bool = False):
    seed_path = Path(seed_path)
    name = seed_path.stem
    work = config.WORK_DIR
    log = RunLog(str(seed_path))
    log_path = config.OUTPUT_DIR / f"{name}_log.txt"

    try:
        print("1/5  Rewriting script...")
        seed = script_mod.load_seed(seed_path)
        if no_rewrite or config.REWRITE_BACKEND == "none":
            text = seed
            log.add(f"Rewrite:  skipped ({'--no-rewrite' if no_rewrite else 'REWRITE_BACKEND=none'})")
        else:
            text = script_mod.rewrite(seed)
            log.add(f"Rewrite:  via {config.REWRITE_BACKEND}")
        (work / f"{name}_script.txt").write_text(text, encoding="utf-8")
        hook = script_mod.hook_of(text)
        word_count = len(text.split())
        print(f"     Hook: {hook}")
        log.add(f"Hook:     \"{hook}\"")
        log.add(f"Words:    {word_count}")
        log.add(f"Script:   {text}")
        log.add("")

        print("2/5  Synthesizing voiceover (ElevenLabs)...")
        voice = tts_mod.synthesize(text, work / f"{name}_voice.wav")
        duration = get_audio_duration(voice)
        print(f"     {duration:.1f}s of narration")
        log.add("Voice:")
        log.add(f"  duration:   {duration:.1f}s")
        log.add(f"  voice_id:   {config.ELEVENLABS_VOICE_ID}")
        log.add(f"  model:      {config.ELEVENLABS_MODEL}")
        log.add(f"  stability/similarity/style: "
                f"{config.ELEVENLABS_STABILITY}/{config.ELEVENLABS_SIMILARITY}/{config.ELEVENLABS_STYLE}")
        log.add("")

        print("3/5  Transcribing captions (Whisper)...")
        words = cap_mod.transcribe_words(voice)
        # Drift-correct to the script FIRST: aligned has exactly the script's
        # words with corrected timing, so sentence boundaries land accurately
        # even when Whisper's word count differs.
        aligned, exact = cap_mod.align_script_words(words, text)
        n_sent = vis_mod.count_sentences(words, duration, script_text=text)
        sent_ends = vis_mod.sentence_end_times(aligned, duration, script_text=text)

        # Build scenes from the corrected timing so scene boundaries match.
        images = vis_mod.collect_images()
        scenes = vis_mod.build_scenes(images, duration, words=aligned, script_text=text)
        cum_ends, acc = [], 0.0
        for s in scenes:
            acc += s.duration
            cum_ends.append(acc)

        # Resolve cutaways. The narration pause and the cutaway VIDEO must land at
        # the SAME time, so we snap the insert to the scene boundary nearest the
        # end of the chosen sentence. With one image per sentence this is exact.
        cutaways = []
        for c in getattr(config, "CUTAWAYS", []):
            clip = config.CUTAWAY_DIR / c["clip"]
            after = c["after_sentence"]
            if not clip.exists():
                log.warn(f"Cutaway clip not found, skipping: {clip}")
                continue
            if after < 1 or after > len(sent_ends):
                log.warn(f"Cutaway after_sentence={after} out of range (1..{len(sent_ends)}); skipping.")
                continue
            target = sent_ends[after - 1]
            insert_idx = min(range(len(cum_ends)), key=lambda i: abs(cum_ends[i] - target)) + 1
            boundary = cum_ends[insert_idx - 1]
            d = asm_mod._probe_duration(clip)
            snapped = abs(boundary - target) > 0.3
            cutaways.append({"clip": clip, "duration": d, "narration_time": boundary,
                             "after_sentence": after, "insert_idx": insert_idx,
                             "target": target, "snapped": snapped})
            if snapped:
                log.warn(f"Cutaway requested after sentence {after} (@{target:.1f}s) but "
                         f"snapped to nearest scene boundary @{boundary:.1f}s. For an exact "
                         f"placement, use one image per sentence (currently {len(images)} "
                         f"images for {n_sent} sentences).")

        # Captions: exact spelling from script (robust align), then shift words
        # that fall after each cutaway, and break caption groups at the boundary.
        shifted = cap_mod.apply_cutaway_shifts(aligned, cutaways)
        break_times = [ca["narration_time"] for ca in cutaways]
        ass = cap_mod.write_ass(shifted, work / f"{name}.ass", script_text=None,
                                break_times=break_times)
        log.add("Captions:")
        log.add(f"  whisper_model: {config.WHISPER_MODEL}")
        log.add(f"  words_timed:   {len(words)}")
        log.add(f"  caption_text:  from script (robust align{'' if exact else ', with minor drift corrected'})")
        log.add(f"  words/line:    {config.CAPTION_MAX_WORDS}, font {config.CAPTION_FONTSIZE}")
        log.add(f"  sentences:     {n_sent}")
        log.add("")

        print("4/5  Timing visuals...")
        # Insert cutaway scenes at their snapped boundary (reverse keeps indices valid).
        for ca in sorted(cutaways, key=lambda c: c["insert_idx"], reverse=True):
            scenes.insert(ca["insert_idx"],
                          vis_mod.Scene(ca["clip"], ca["duration"], is_cutaway=True))
        print(f"     {len(images)} images, {len(cutaways)} cutaway(s), {len(scenes)} scenes")
        log.add("Visuals:")
        log.add(f"  images:     {len(images)} ({', '.join(p.name for p in images)})")
        log.add(f"  sentences:  {n_sent}")
        log.add(f"  ken_burns_zoom: {config.KEN_BURNS_ZOOM}")
        log.add(f"  transition:     {config.TRANSITION_SEC}s crossfade")
        if cutaways:
            log.add("  cutaways:")
            for ca in cutaways:
                log.add(f"    {ca['clip'].name} after sentence {ca['after_sentence']} "
                        f"@ {ca['narration_time']:.1f}s, {ca['duration']:.1f}s "
                        f"(narration pauses, clip audio plays)")
        if len(images) > n_sent and n_sent > 0:
            log.warn(f"More images ({len(images)}) than sentences ({n_sent}); "
                     f"fell back to even time split (no sentence alignment).")
        elif len(images) < n_sent:
            log.warn(f"Fewer images ({len(images)}) than sentences ({n_sent}); "
                     f"each image covers multiple sentences.")
        # per-scene timeline
        log.add("  scene timeline:")
        t = 0.0
        for i, sc in enumerate(scenes):
            tag = " [CUTAWAY]" if getattr(sc, "is_cutaway", False) else ""
            log.add(f"    {i+1:02d}. {sc.image.name:20s} {t:6.1f}-{t+sc.duration:6.1f}s "
                    f"({sc.duration:.1f}s){tag}")
            t += sc.duration
        log.add("")

        print("5/5  Assembling video...")
        out = config.OUTPUT_DIR / f"{name}.mp4"
        music_path = Path(music) if music else None
        if music_path is None and config.DEFAULT_MUSIC:
            cand = config.MUSIC_DIR / config.DEFAULT_MUSIC
            if cand.exists():
                music_path = cand
        if music_path and music_path.exists():
            log.add(f"Music:    {music_path.name} (vol {config.MUSIC_VOLUME}, looped + faded)")
        else:
            log.add("Music:    none")
            if music_path:
                log.warn(f"Music file not found: {music_path}")
        # Optional SFX/audio layer from manifests/<name>.json (if one exists).
        audio_spec = None
        manifest_path = config.ROOT / "manifests" / f"{name}.json"
        if manifest_path.exists():
            import json as _json
            from orchestrator.audio_spec import parse_audio_spec
            manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
            audio_spec = parse_audio_spec(manifest)
            if not audio_spec.is_empty():
                log.add(f"SFX:      {len(audio_spec.all_cues())} cue(s) mixed "
                        f"(ducking {'on' if audio_spec.duck_enabled else 'off'})")

        render_notes = asm_mod.assemble(scenes, voice, ass, out, music_path,
                                        voice_duration=duration, cutaways=cutaways,
                                        audio_spec=audio_spec, words=aligned)

        log.add("")
        log.add("Scene render (image=still+KenBurns, video=Higgsfield clip):")
        for n in render_notes:
            log.add(f"  {n}")
            if "slowed to" in n:
                try:
                    factor = float(n.split("slowed to")[1].split("x")[0])
                    if factor < 0.5:
                        log.warn(f"Clip heavily slowed ({factor:.2f}x) — may look like "
                                 f"slow-motion: {n.split(':')[0].strip()}. "
                                 f"Generate a longer source clip or shorten that sentence.")
                except (ValueError, IndexError):
                    pass

        log.add("")
        log.add("Output:")
        log.add(f"  file: {out}")
        log.add(f"  size: {_filesize_kb(out)} KB")
        log.add(f"  spec: {config.WIDTH}x{config.HEIGHT} @ {config.FPS}fps")
        log.write(log_path)
        print(f"\nDone -> {out}")
        print(f"Run log -> {log_path}")
        return out

    except Exception as e:
        import traceback
        log.write(log_path, error=traceback.format_exc())
        print(f"\nRun failed. Details written to: {log_path}")
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seed", help="path to script .txt")
    ap.add_argument("--music", help="background music file", default=None)
    ap.add_argument("--no-rewrite", action="store_true",
                    help="narrate the script verbatim (skip the rewrite step)")
    args = ap.parse_args()
    run(args.seed, args.music, args.no_rewrite)