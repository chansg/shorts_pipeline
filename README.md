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
                aligned, 3 words/line) → ffmpeg assemble + music
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

## CLI reference (everything still works without the GUI)

```
python pipeline.py scripts/NAME.txt [--no-rewrite] [--music PATH]
python prompt_gen.py scripts/NAME.txt [--style dark_fantasy|folklore_horror]
python -m i2v.cli --manifest manifests/NAME.json --images episodes/NAME/stills \
                  --out episodes/NAME/clips [--only scene ...] [--dry-run]
python -m i2v.imagegen --manifest manifests/NAME.json \
                  --images episodes/NAME/stills --refs refs
python -m modules.tts voices        # list ElevenLabs voice IDs
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
