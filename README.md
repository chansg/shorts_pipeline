# Ashen_Chan Shorts Pipeline

Turns a written script into a finished, vertical (9:16) YouTube Short:
**voiceover → word-timed captions → motion video → music → cutscene inserts**,
all aligned to the script automatically.

Built for faceless lore-narration Shorts (Dark Souls lore, dark folklore, etc.).
Visuals can be AI-generated stills (animated with a Ken Burns zoom) and/or AI
video clips (e.g. from Veo / Higgsfield), mixed freely — one media file per
sentence.

---

## What it does

```
script.txt
   │
   ├─ ElevenLabs ──────────► voiceover (.wav)
   │
   ├─ Whisper ─────────────► word timings ──► script-accurate captions (.ass)
   │
   ├─ your media (01..NN) ──► scenes, one per sentence
   │                          • image → Ken Burns zoom
   │                          • video → cover-cropped to 9:16, fit to scene
   │
   ├─ cutaways (optional) ──► cutscene plays with its OWN audio; narration pauses
   │
   └─ ffmpeg ──────────────► crossfades + music ──► output/NAME.mp4  (+ run log)
```

Key behaviours:
- **Script is the source of truth.** Sentence count, caption spelling, and image
  timing all come from the script — Whisper only supplies *timing*. Fantasy
  proper nouns (Oolacile, Artorias…) always spell correctly.
- **One media file per sentence.** Images switch at sentence boundaries.
- **No stretching.** Every image/clip is scaled-to-cover and cropped to true
  9:16 (cutaways are letterboxed to preserve the full widescreen frame).
- **Self-documenting.** Every run writes `output/NAME_log.txt` with the full
  scene timeline, per-scene render notes, warnings, and output spec.

---

## Requirements

- **Python 3.11+** (Windows 11 supported)
- **ffmpeg** on PATH — `winget install Gyan.FFmpeg`, then restart the terminal
  (verify with `ffprobe -version`)
- Python packages: `pip install -r requirements.txt`
  (includes `elevenlabs`, `faster-whisper`, `python-dotenv`)
- **ElevenLabs API key** in a `.env` file at the project root:
  ```
  ELEVENLABS_API_KEY=your_key_here
  ```
  > Note: free-tier API **cannot** use ElevenLabs *library* voices (HTTP 402).
  > Use a premade voice ID, or a paid tier for library voices.

`.env` holds a billable secret — keep it out of git (it's in `.gitignore`).

---

## Project structure

```
shorts_pipeline/
├─ pipeline.py            # main orchestrator (run this)
├─ prompt_gen.py          # generate Gemini/Veo prompts from a script
├─ config.py              # all settings
├─ requirements.txt
├─ modules/
│  ├─ script.py           # load seed + optional rewrite
│  ├─ tts.py              # ElevenLabs voiceover
│  ├─ captions.py         # Whisper timing → script-accurate .ass captions
│  ├─ visuals.py          # collect media, sentence-aligned scene timing
│  └─ assemble.py         # ffmpeg: Ken Burns / clips / cutaways / music / mux
├─ scripts/               # your script .txt files (one sentence per beat)
├─ assets/
│  ├─ images/             # scene media: 01.png, 02.mp4, 03.png … (mix freely)
│  ├─ music/              # background tracks
│  └─ cutaways/           # cutscene clips (play with their own audio)
├─ prompts/               # prompt_gen output (beat sheets + ready prompts)
└─ output/                # finished NAME.mp4 + NAME_log.txt
```

---

## Workflow

### 1. Write the script
Create `scripts/NAME.txt`. **One sentence per visual beat.** Keep sentences
even (~15–20 words) so scenes are evenly paced; the number of sentences is the
number of media files you'll need.

### 2. (Optional) Generate image/motion prompts
```
python prompt_gen.py scripts/NAME.txt --style folklore_horror
```
- First run writes `prompts/NAME.beats.txt` — one beat per sentence. Fill in a
  `scene:` line for each and set `type: still` or `type: clip`.
- Run again to write `prompts/NAME_prompts.txt` — copy-paste-ready Gemini image
  prompts (with the master style block attached) and filter-safe Veo motion
  prompts. It also **warns on words that tend to trip Gemini/Veo filters.**

Styles: `dark_fantasy`, `folklore_horror` (add more in `prompt_gen.py`).

### 3. Make the media
Generate one media file per sentence and drop them in `assets/images/`, named in
order to match the beats: `01`, `02`, … (`.png`/`.jpg` for stills, `.mp4`/`.mov`
for clips). **Tips that avoid the common problems:**
- Generate clips **6–8s long**, not 4s — short clips get slowed to fill their
  scene (slow-motion). The log reports each clip's speed.
- Use **stills for calm beats** — Ken Burns animates them and they never slow.
- Put your **best image on the emotional/retention beat.**

### 4. (Optional) Add music and a cutscene
- Drop a track in `assets/music/` and set `DEFAULT_MUSIC` in `config.py` (or pass
  `--music path`).
- For a cutscene insert, drop a clip in `assets/cutaways/` and set `CUTAWAYS` in
  `config.py` (see below).

### 5. Render
```
python pipeline.py scripts/NAME.txt --no-rewrite
```
Output: `output/NAME.mp4` and `output/NAME_log.txt`.

---

## Command reference

```
python pipeline.py scripts/NAME.txt [--no-rewrite] [--music PATH]
```
- `--no-rewrite` — narrate the script verbatim (recommended for polished scripts;
  skips the local-LLM rewrite step).
- `--music PATH` — background track for this run (overrides `DEFAULT_MUSIC`).

```
python prompt_gen.py scripts/NAME.txt [--style dark_fantasy|folklore_horror]
```

---

## Cutaways (cutscene inserts)

Insert a clip that **plays with its own audio while the narration pauses**, then
resumes — e.g. a boss-fight cutscene. In `config.py`:

```python
CUTAWAYS = [
    {"after_sentence": 5, "clip": "artorias_speech.mp4"},
]
```
- The clip lives in `assets/cutaways/`.
- It's inserted at the end of the given sentence (snapped to the matching scene
  boundary so video and audio stay in sync).
- It's **letterboxed** to 9:16 to preserve the full widescreen frame.
- Set `CUTAWAYS = []` for videos with no cutscene.

---

## Key settings (`config.py`)

| Setting | What it does |
|---|---|
| `WIDTH`, `HEIGHT`, `FPS` | output format (1080×1920 @ 30) |
| `ELEVENLABS_VOICE_ID` | voice (premade ID for free tier) |
| `ELEVENLABS_STABILITY/SIMILARITY/STYLE` | voice expressiveness |
| `WHISPER_MODEL` | caption timing model (`small`/`base`) |
| `CAPTION_MAX_WORDS`, `CAPTION_FONTSIZE`, `CAPTION_MARGIN_*` | caption look/fit |
| `KEN_BURNS_ZOOM` | still zoom amount (1.18) |
| `TRANSITION_SEC` | crossfade length between scenes |
| `MUSIC_VOLUME`, `DEFAULT_MUSIC` | background music |
| `CUTAWAYS` | cutscene inserts |
| `REWRITE_BACKEND` | `ollama` / `claude` / `none` |

---

## The run log

`output/NAME_log.txt` records, every run: the hook, word count, full script,
voice settings, caption source, sentence count, the per-scene timeline (with
`[CUTAWAY]` tags), per-scene render notes (still vs clip, and any clip slowdown),
music status, output spec, and warnings. On failure it captures the traceback.
Attach it when reviewing a render.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Changes don't take effect | Stale files — make sure you replaced the file and re-ran. |
| `ffmpeg`/`ffprobe` not found | Install ffmpeg and restart the terminal. |
| ElevenLabs `402 paid_plan_required` | You used a *library* voice on free tier — switch to a premade voice ID. |
| A clip looks like slow-motion | Source clip too short for its scene; generate it longer (6–8s) or use a still. The log flags clips slowed past ~0.5×. |
| Captions misspell names | Should be fixed (captions come from the script); if the log says "from Whisper", your script and audio diverged. |
| Gemini/Veo rejects a prompt | Content filter — rephrase to physical/atmospheric description (no distress/harm/child words). `prompt_gen.py` warns on likely triggers. |
| `Cutaway clip not found` | The `CUTAWAYS` entry points to a file not in `assets/cutaways/`; fix the name or set `CUTAWAYS = []`. |
| Images look stretched | Shouldn't happen (scale-to-cover + crop); keep the subject roughly centered since edges are cropped to 9:16. |

---

## Notes

- This is a personal content tool. When using third-party game footage or IP in
  cutaways/clips, be aware of copyright and platform policies; narration over
  original art is the safest footing.
- Tooling by Chan (github.com/chansg), built iteratively with Claude.