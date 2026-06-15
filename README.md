# Ashen_Chan Shorts Pipeline

Turns a written script into a finished, vertical (9:16) YouTube Short — script →
AI stills (style-locked) → Veo motion on chosen scenes → ElevenLabs voiceover →
word-timed captions → music → **your approval** → `output/NAME.mp4`.

Built for faceless lore-narration Shorts (Dark Souls lore, dark folklore, etc.).
One script sentence = one scene; some scenes are animated Veo clips, the rest
stay Ken Burns stills (the cost-saving hybrid).

---

## The GUI (start here)

```
python app.py
```

One window, eight gated stages, end to end:

```
1 Script        pick/edit scripts/<name>.txt — sentence count = scene count
2 References    REQUIRED style lock (refs/style_*.png, every scene) +
                optional character refs — gates stage 3
3 Prompts       reference-aware auto-draft, one still prompt + motion prompt
                per sentence, into the i2v manifest — edit everything in-app
4 Stills        Nano Banana 2 with the locked refs; gallery; regenerate any
                single scene; approve to continue
5 Animate       choose which scenes become Veo clips (💰): test ONE clip
                first, confirm the batch, rendered clips are never re-billed
6 Handoff       automatic — clips/stills placed into assets/images/ as
                NN.mp4 / NN.png in manifest order (no manual renaming)
7 Voice & Build ElevenLabs voice + settings → Whisper captions (script-
                aligned, active-word style) → ffmpeg assemble + music
8 Review        the human gate: spec check (1080×1920/30fps/audio), per-scene
                contact sheet with near-black detection, then
                Approve → ready to publish (opens the output folder)
```

The stage order is **enforced**: no prompts without the style lock, no stills
without prompts, no animation without approved stills, no handoff without the
animate decision, no build without the handoff.

Everything is **idempotent / resumable** — stage completion is derived from
files on disk, so closing the app loses nothing and re-running a stage only
fills gaps (existing Veo clips and unchanged TTS audio are never re-billed).

Long stages stream progress into the window; the UI never freezes.

### Where things live

```
shorts_pipeline/
├─ app.py                 # the GUI (python app.py)
├─ pipeline.py            # CLI orchestrator (still works standalone)
├─ prompt_gen.py          # beat-sheet/prompt CLI + shared style/lint logic
├─ gen_images.py          # legacy Imagen candidate generator (CLI)
├─ config.py              # all pipeline settings (voice, captions, visuals)
├─ orchestrator/          # what the GUI calls: stages, state, manifest gen, QC
├─ i2v/                   # vendored aria-i2v: Nano Banana 2 + Veo 3.1
│  └─ (CLI still works:  python -m i2v.cli --dry-run)
├─ fonts/                 # bundled caption font (Anton, SIL OFL) — burned via fontsdir
├─ modules/               # script / tts / captions / visuals / assemble
├─ scripts/               # episode scripts (<name>.txt, one sentence per scene)
├─ refs/                  # style lock + character reference images
├─ manifests/             # <name>.json prompt manifests (the creative spec)
├─ episodes/<name>/       # generated stills/ and clips/ per episode
├─ assets/images/         # pipeline input after handoff (NN.png / NN.mp4)
├─ assets/music/          # background tracks
└─ output/                # finished <name>.mp4 + logs
```

### Keys & setup

```
pip install -r requirements.txt
copy .env.example .env          # then fill in both keys
```

One `.env` at the repo root holds both keys (gitignored, also editable in the
GUI's Settings tab):

```
GEMINI_API_KEY=...        # Google AI Studio: images (Nano Banana 2) + Veo + drafting
ELEVENLABS_API_KEY=...    # voiceover
```

- **Python 3.11+** (Windows 11 supported)
- **ffmpeg** on PATH — `winget install Gyan.FFmpeg`, then restart the terminal
- ElevenLabs free tier can't use *library* voices (HTTP 402) — use a premade
  voice ID or a paid tier.

### Cost notes

- Stills (Nano Banana 2): cheap per image; regenerate freely.
- **Veo bills per second of video** (~8s per clip). The GUI makes you test one
  clip before a batch, asks for explicit confirmation, and skips anything
  already rendered. Keep most scenes as stills.
- Prompt drafting / title drafting are tiny Gemini text calls (optional —
  uncheck "Draft with Gemini" for a free deterministic draft).

### Captions

Two renderers, switchable with `CAPTION_STYLE` in `config.py` (or the **Settings**
tab → *Caption style*, which applies to the next build):

- **`active_word`** (default) — one big bold uppercase word at a time, bright yellow
  with a thick black outline + shadow, centred low-middle, popping in synced to each
  spoken word. Tune it with the `CAPTION_AW_*` settings (font, size, fill colour,
  outline, vertical position, words-per-cue).
- **`classic`** — the older 3-words-per-line style (white text, yellow active word),
  tuned via the `CAPTION_*` settings.

Both reuse the word-level Whisper timings the pipeline already produces (no
re-transcription). The active-word font (**Anton**, SIL OFL) is bundled in `fonts/`
and burned via ffmpeg's `fontsdir`, so it works on any machine without installing the
font system-wide.

---

## Gameplay pipeline (second, parallel mode)

A separate pipeline for **gameplay footage**, alongside the lore wizard — its own
**🎮 Gameplay** tab in the GUI. It takes a pre-trimmed clip (any game, any length,
16:9 or other) and turns it into a 9:16 Short with per-speaker captions, optional
effects, and a like/subscribe overlay. The lore pipeline is untouched by it.

### Manual mode (the main flow)

1. **Upload** a pre-trimmed clip.
2. **① Transcribe** — WhisperX transcribes, word-aligns, and (with a token)
   diarizes the speakers in one GPU pass. No token / one voice → single-speaker.
3. **Transcript gate** — fix ASR errors in the editable grid, rename speakers
   (`SPEAKER_00` → "Chan") and pick their colours. *These rows are the captions.*
4. Choose **effects** (punch-zoom / shake, driven off the audio-energy envelope),
   a **caption** font/position, and a **like/subscribe overlay** (asset, position,
   start, duration).
5. **② Build Short** → reframe (blur-pad) → burn captions → effects → overlay →
   `output/gameplay/<clip>/<clip>_short.mp4`. Stages are resumable.

### Setup

```bash
pip install -r requirements.txt
pip install -r requirements-gameplay.txt   # WhisperX + pyannote + torch (heavy)
```

`torch` must match your CUDA — install the CUDA build first (see the top of
`requirements-gameplay.txt`). For diarization, add a free HuggingFace token to
`.env` and accept the model terms once:

```
HF_TOKEN=hf_xxx
```
- https://huggingface.co/settings/tokens
- accept: `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`

Without `HF_TOKEN` the gameplay pipeline still works — it just captions as a
single speaker (no per-speaker colour). The first real transcribe+diarize run is
yours to execute (it needs your GPU).

### Overlays

Like/subscribe assets live in `overlays/` — transparent `.png` or alpha video
(`.mov` ProRes 4444 / `.webm` VP9). A placeholder
(`like_subscribe_placeholder.png`) ships in the repo; drop your own in and click
**↻ overlays**. Assets without an alpha channel are rejected with a clear error.

### Captions / colours

Captions reuse `modules/karaoke_captions` — the same active-word renderer as the
lore path. The gameplay path passes 4-tuples `(text, start, end, speaker)`, which
drive per-speaker colour (explicit hex wins; otherwise a 6-colour palette is
auto-assigned in order of appearance). The bundled **Anton** font is used the same
way (via `fontsdir`).

### Full-auto (experimental)

Under the **⚠ Experimental** accordion: ingest a long (~1hr) video, auto-detect &
categorise highlights (an audio-energy spike pass + an LLM pass over the diarized
transcript → clutch / funny / rage / story), auto-cut each, and run them through
the manual backend. Compute-heavy and GPU-gated; per-candidate failures are
contained; the LLM backend reuses `REWRITE_BACKEND` (ollama/claude). It is
isolated from manual mode and cannot affect it.

---

## The manifest (what stage 3 produces)

`manifests/<name>.json`, aria-i2v schema — `defaults` (model, aspect, duration,
negative prompt, compositional style suffix), `style_refs` (applied to every
scene), `characters` (named ref groups), and per-clip `prompt` /
`motion_prompt` / `refs` / `animate`. Because prompts are generated *after* the
style lock is set, they describe composition and subject only — the look comes
from the reference images, and character refs are wired into exactly the
scenes that feature them.

---

## Sound effects & custom audio

The manifest carries an **optional** audio layer, separate from the narration.
A manifest with no `audio` key and no per-clip `sfx` key builds exactly as
before — this is fully backward compatible. **Cue tags live only in these
blocks, never in the script**, so the voiceover can physically never speak a
cue marker.

### The two commands you need

```
python app.py                              # local UI: import mp3, attach SFX, Build
python pipeline.py scripts/NAME.txt        # headless build (reads manifests/NAME.json)
```

In the UI, open **7 · Voice & Build → 🔊 Sound effects & custom audio**: upload
an mp3, pick a source + layer, set timing/gain, **Add cue**, then **Build**.
Everything is written into the manifest, so the headless build uses the same
cues.

### Three layers

| layer | what it is | placement |
|-------|------------|-----------|
| `ambient_bed` | one continuous track for the whole video | from 0, loops |
| `music_bed`   | a second continuous bed (e.g. an imported song) | from 0, loops |
| `motif`       | a recurring/loopable cue, placed one or more times | anchored |
| one-shots     | discrete cues under a clip's `sfx[]` array | anchored |

Every cue supports: `source`, `gain_db`, optional `pan` (-1..1), `fade_in`,
`fade_out`, `loop`, and an `at` anchor (not needed for beds).

### Anchors — `at`

```jsonc
{"scene": 2, "offset": 0.5}                 // 0.5s into scene 2
{"word": "knocking", "occurrence": 1, "offset": 0}  // on a spoken word
{"time": 4.2}                               // absolute seconds from the start
```

Word anchors reuse the Whisper word timings the build already computes — land a
knock exactly on the word "knocking".

### `source` — tags, not paths

A cue's `source` is a **library tag** (resolved via `assets/sfx/sfx_map.json`),
an `@import/<name>` alias for an imported track, or a raw path (escape hatch).
Bundled tags: `knock_wood`, `wind_hall`, `rot_shimmer`, `boom_low`.

### Example manifest with SFX

```jsonc
{
  "defaults": { /* … unchanged … */ },
  "audio": {
    "ambient_bed": { "source": "wind_hall", "gain_db": -22,
                     "loop": true, "fade_in": 2.0, "fade_out": 3.0 },
    "motifs": [
      { "source": "rot_shimmer", "at": { "scene": 6, "offset": 0.0 },
        "gain_db": -14, "pan": -0.3, "fade_in": 0.5, "loop": true }
    ],
    "ducking": { "enabled": true, "amount_db": 8, "threshold": 0.05 }
  },
  "clips": [
    { "image": "01.png", "name": "01_…", "narrates": "Something is knocking.",
      "sfx": [
        { "source": "knock_wood", "at": { "word": "knocking" },
          "gain_db": -6, "pan": 0.2, "fade_out": 0.3 }
      ] }
  ]
}
```

### Import your own mp3

```
python pipeline.py scripts/NAME.txt --sfx-import song.mp3 --sfx-as music_bed
python pipeline.py scripts/NAME.txt --sfx-import knock.mp3 thud.mp3   # just register tags
```

Import runs ffmpeg **`loudnorm`** (I=-16 / TP=-1.5 / LRA=11) so levels are sane,
copies the result into `assets/sfx/imported/`, and registers it so it's usable
as a bare tag or `@import/<name>` on any layer. The UI's **Import & normalize**
button does the same.

### Mixing

All layers render in a single ffmpeg `filter_complex`: `adelay` + `volume` +
`pan` + `afade` per cue → `amix=normalize=0` (no auto-normalization that would
wreck levels) → optional `sidechaincompress` so SFX dip under the voice when
they overlap → muxed onto the video. Cue ordering is stable, so the same
manifest always produces the same mix.

### Validation

Bad audio data fails **loudly** with every problem listed at once (unknown
tags, out-of-range scenes, missing anchor words, bad pan/gain) — both in the UI
and the headless build. There is no silent fallthrough.

---

## CLI reference (everything still works without the GUI)

```
python pipeline.py scripts/NAME.txt [--no-rewrite] [--music PATH]
python pipeline.py scripts/NAME.txt --sfx-import a.mp3 --sfx-as music_bed
python prompt_gen.py scripts/NAME.txt [--style dark_fantasy|folklore_horror]
python -m i2v.cli --manifest manifests/NAME.json --images episodes/NAME/stills \
                  --out episodes/NAME/clips [--only scene ...] [--dry-run]
python -m i2v.imagegen --manifest manifests/NAME.json \
                  --images episodes/NAME/stills --refs refs
python -m modules.tts voices        # list ElevenLabs voice IDs
python -m pytest tests/ -q          # timeline parse/validation + SFX timing tests
```

Key pipeline behaviours (unchanged):

- **Script is the source of truth.** Sentence count, caption spelling, and
  scene timing all come from the script — Whisper only supplies timing, so
  fantasy proper nouns (Oolacile, Artorias…) always spell correctly. A
  7-sentence script drives 7 scenes; an 8-sentence script drives 8.
- **No stretching.** Media is scaled-to-cover and cropped to true 9:16;
  clips are trimmed or gently slowed to their sentence's duration.
- Ken Burns zoom on stills, 0.4s crossfades, looped/faded music bed.
- Every render writes `output/NAME_log.txt` with the full scene timeline.

### Cutaways (cutscene inserts, CLI only)

Insert a clip that plays with its own audio while the narration pauses, then
resumes. In `config.py`:

```python
CUTAWAYS = [{"after_sentence": 5, "clip": "artorias_speech.mp4"}]
```

The clip lives in `assets/cutaways/`, is letterboxed to 9:16, and is snapped
to the nearest scene boundary. Set `CUTAWAYS = []` to disable.