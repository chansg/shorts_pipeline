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

1. **Upload** a pre-trimmed clip. (Ingest strips stray data/timecode tracks and
   forces CFR when needed, so burned captions don't drift on VFR captures.)
2. **① Transcribe** — WhisperX transcribes, word-aligns, and (with a token)
   diarizes the speakers in one pass. The device is auto-detected: **CUDA →
   large-v2; CPU → a smaller model** with a loud warning (see Setup). No token /
   one voice → single-speaker.
3. **Transcript gate** — fix ASR errors in the editable grid, rename speakers
   (`SPEAKER_00` → "Chan") and pick their colours. *These rows are the captions.*
   The **Bulk edits & caption preview** panel adds: multi-row speaker reassignment
   (fix a stretch the diariser mislabelled), find/replace (a misheard name, fixed
   once), merge/split rows, and a **Re-apply captions** preview that re-renders
   just the caption track — no re-transcription.
4. Choose **effects** (punch-zoom / shake, driven off the audio-energy envelope),
   a **caption** font/position (default `0.78` sits in the lower blur band, off the
   HUD), and a **like/subscribe overlay** (asset, position, start, duration).
5. **② Build Short** → reframe (blur-pad) → burn captions → effects → overlay →
   `output/gameplay/<clip>/<clip>_short.mp4`. Stages are resumable.

### Setup

```bash
pip install -r requirements.txt
# IMPORTANT: install the CUDA-matched torch FIRST, or you get the CPU-only build
# and WhisperX runs on CPU (slow; full-auto becomes impractical):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gameplay.txt   # WhisperX + pyannote (pinned)
```

The pipeline auto-detects the device: **CUDA → large-v2**, **CPU → `small`** with a
prominent warning in the transcribe log naming this exact fix. (The first real run
ran on CPU because the installed torch was the CPU-only build — `cu121` above is the
cure; pick the tag matching your driver.)

For **diarization** (per-speaker colour) you need a token **and** to accept the
pyannote model licences once — a common gotcha that otherwise fails with an auth
error:

```
HF_TOKEN=hf_xxx
```
- token: https://huggingface.co/settings/tokens
- click **Agree** on BOTH (whisperx 3.8 defaults to community-1):
  - https://huggingface.co/pyannote/speaker-diarization-community-1
  - https://huggingface.co/pyannote/segmentation-3.0

Without `HF_TOKEN` (or the licence) the pipeline still works — it just captions as a
single speaker (no per-speaker colour), and the log says which of the two is
missing. The pinned, tested stack is whisperx 3.8.6 / pyannote.audio 4.0.4 /
torch 2.8.0. **The first real transcribe+diarize run is yours to execute** (needs
your GPU, token, and accepted licence).

### Noisy game audio (transcript quality)

Loud game audio over voice chat used to break the transcript two ways: a **massive
dropout** (whole speech regions missing) and a **repetition collapse** (one "word"
of hundreds of repeated letters, e.g. `Naaaaaa…`). Both are handled in
`gameplay/transcribe.py` + `gameplay/config.py`:

- **Clean 16k-mono prep** — WhisperX downmixes to 16k mono but applies no filtering,
  so quiet voice stays buried. We add an explicit downmix + **high-pass**
  (`WHISPERX_AUDIO_HIGHPASS_HZ`, cuts explosion/footstep rumble) + **loudnorm**
  (`WHISPERX_AUDIO_LOUDNORM`, lifts the voice) so VAD finds the speech and ASR has
  SNR. The transcribe log prints the source vs prepped sample-rate/channels.
- **Repetition guard** — WhisperX runs *batched* inference, whose decode path honours
  only `WHISPERX_NO_REPEAT_NGRAM_SIZE` / `WHISPERX_REPETITION_PENALTY` (the
  temperature / compression-ratio / `condition_on_previous_text` knobs gate the
  *sequential* path only and are no-ops here — documented inline). These two are the
  real anti-loop levers.
- **VAD** is always on; its sensitivity is exposed as `WHISPERX_VAD_ONSET` /
  `WHISPERX_VAD_OFFSET` (lower onset = recover more speech, at the risk of decoding
  loud non-speech).
- **Post-guard** — `WHISPERX_MAX_WORD_CHARS` repairs a repetition token (collapses
  4+ char runs) or drops it if still absurd, *before* the editable grid, so a
  300-char wall can never reach captions. `WHISPERX_MAX_WORD_S` still clamps duration.

**Success signal on a re-run:** the log shows `… → 16000Hz/mono`, `VAD: pyannote on`,
a word count in a realistic range (dense chatter ⇒ well over 13 words for ~34s), and
**no** multi-second single-letter token and **no** 20+ second gap.

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

## Full-Auto Experiment (its own landing entry — 16:9 YouTube output)

The experimental long-form processor is its **own card on the landing screen**
(marked ⚗ EXPERIMENTAL), separate from Gaming. Its output is a **standard 16:9
YouTube video**, not a 9:16 Short — which is why it lives outside the Gaming Shorts
section. It lives in its own package (`fullauto/`) and **never calls the 9:16 Shorts
backend** (blur-pad reframe, like/subscribe overlay, karaoke captioner, vertical
export). It still reuses the shared, aspect-agnostic infra in `gameplay/`
(transcription, config, the audio-energy envelope).

A long video becomes a *reviewable* set of candidates rather than an unattended export:

1. **① Detect highlights** — transcribe + diarize the long video (staged progress;
   GPU-gated), then a **fused detector** finds & categorises candidates: audio-energy
   peaks + a reaction-keyword scan + an **LLM judge** over the transcript →
   clutch / funny / rage / hype / story, each with a time range, score, and hook
   caption. No building yet.
2. **Review gallery** — candidates appear as preview thumbnails + a table; tick the
   ones worth keeping.
3. **② Build selected → 16:9 YouTube video** — the ticked highlights are cut at the
   source's **native resolution** and assembled into one landscape video
   (`fullauto/export.py`). No reframe, overlay, or karaoke captions.

> **Captions on the YouTube cut are an open question.** The per-speaker karaoke style
> is a Shorts aesthetic, so full-auto ships the 16:9 video *without* it for now; a
> `TODO(16:9 captions)` hook in `fullauto/export.py` marks where optional landscape
> captioning could plug in later.

Compute-heavy and GPU-gated; detection is failure-contained (a bad LLM chunk, a CUDA
OOM, or zero candidates each degrade gracefully); the LLM backend reuses
`REWRITE_BACKEND` (ollama/claude). Isolated from manual mode and cannot affect it.

**Long videos (1–60 min) on a 10GB card.** The ASR model's VRAM is released before
the alignment model loads, and again before diarization, so the card never holds two
models at once; transcription uses `AUTO_TRANSCRIBE_BATCH` and retries once at a
smaller batch on a CUDA OOM; the LLM judge runs over ~2.5-min transcript chunks (never
the whole hour at once). Past `AUTO_MAX_MINUTES` you get a warning, not a block.

**Tuning the detector (first runs are calibration).** Every threshold/weight/lexicon
lives in the `AUTO_*` block of `gameplay/config.py`, commented inline:
- **Energy peaks** — `AUTO_ENERGY_K` (sensitivity), `AUTO_ENERGY_MIN_PROMINENCE`,
  `AUTO_ENERGY_MIN_SPACING_S`, `AUTO_ENERGY_ROLL_S` (local-normalisation window).
- **Reaction lexicon** — `AUTO_REACTION_LEXICON`, a per-category phrase dict; **add
  your group's slang and the games you play** here.
- **Window framing** — `AUTO_CLIP_MIN_S` / `AUTO_CLIP_MAX_S`, `AUTO_LEAD_IN_S`,
  `AUTO_LEAD_OUT_S`.
- **LLM chunking** — `AUTO_LLM_CHUNK_S`, `AUTO_LLM_CHUNK_OVERLAP_S`.
- **Score weights** (default LLM-led) — `AUTO_W_LLM` 0.50, `AUTO_W_ENERGY` 0.25,
  `AUTO_W_REACTION` 0.15, `AUTO_W_OVERLAP` 0.10; `AUTO_TOP_N` caps survivors.

Detection is deterministic given the same transcript + config, so tuning is
reproducible (the LLM's own output is the only non-deterministic input).

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