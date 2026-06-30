# Ashen_Chan Shorts Pipeline

A desktop tool for making vertical **9:16 YouTube Shorts** for the **GamerChans**
channel. It does three different jobs, all from one window:

1. **Music montage** — drop in a few gameplay clips + one music track → one Short.
2. **Lore narration** — turn a written script into a fully AI-generated narrated Short.
3. **Gameplay clips** — turn a recorded gameplay clip into a captioned Short.

Plus a **Full-Auto** helper that scans long recordings and pulls out the best moments
for you.

---

## Getting started

You need **Python 3.11+** and **ffmpeg** (the video tool the whole app relies on).

```bash
# 1. install ffmpeg (Windows) and restart your terminal
winget install Gyan.FFmpeg

# 2. install the Python dependencies
pip install -r requirements.txt

# 3. add your API keys
copy .env.example .env        # then open .env and fill it in
```

The `.env` file (kept out of git) holds two keys — you only need these for the
**lore narration** flow:

```
GEMINI_API_KEY=...        # Google AI Studio — AI images + video + text drafting
ELEVENLABS_API_KEY=...    # the voiceover
```

Then launch the app:

```bash
python app.py
```

It opens a **landing screen** with cards for each mode: **Shorts**, **Gameplay**,
**Full-Auto**, and **Settings** (where you can also edit the keys).

> The **music montage** and **gameplay** flows don't need the API keys — they only use
> ffmpeg. The lore-narration flow is the one that calls (and bills) the AI services.

---

## The three things it makes

### 🎬 Music montage (Shorts → first tab)

Pick several gameplay clips + one royalty-free MP3 (e.g. from YouTube's Audio Library)
and it builds a single 9:16 Short: the clips are reframed and joined with smooth
crossfades, the game audio is cleaned up and turned down **under** the music, and the
**GamerChans** like/subscribe overlay is added. You set where in the song to start (skip to the drop), hit
**Build montage**, and watch the progress — the finished file is saved to your chosen
output folder.

### 📜 Lore narration (Shorts → the numbered stages)

A guided, step-by-step wizard that turns a script into a narrated Short: pick a script →
generate style-locked AI images → animate chosen scenes → add an ElevenLabs voiceover →
auto-caption it → **you approve it** → done. Each step unlocks the next, and nothing is
re-generated (or re-billed) if you re-run — it picks up where you left off.

### 🎮 Gameplay clips (Gameplay tab)

Upload a recorded gameplay clip and it becomes a captioned 9:16 Short:

1. **Transcribe** — speech is auto-transcribed and, optionally, split by speaker
   (each speaker gets their own caption colour).
2. **Edit** — a fast keyboard-driven editor lets you fix the words and speakers quickly.
3. **Build** — captions are burned in, optional zoom/shake effects and a like/subscribe
   overlay are added, and swear words are auto-bleeped.

---

## 🤖 Full-Auto (find the good moments in long recordings)

For when you have long VODs and don't want to scrub through them by hand:

- **Highlight clips** — point it at a long gameplay video and it finds the funny / hype
  moments (from loud voice reactions, plus on-screen kill banners in League/ARAM) and
  hands you ranked candidate clips to finish in the Gameplay tab.
- **Candidate export (batch)** — point it at a *folder* of recordings and it exports the
  top ~5 raw highlight trims from each one, ready for manual editing.

Both run in the background so the window stays responsive, and a bad file just gets
skipped with a warning instead of crashing the run.

---

## Where things live

```
shorts_pipeline/
├─ app.py            # the GUI — run this
├─ config.py         # lore-narration settings
├─ gameplay/         # gameplay clips, music montage, captions, overlays
│  └─ config.py      # gameplay + montage settings
├─ fullauto/         # the long-video highlight finder + batch export
├─ modules/          # shared building blocks (script, voice, captions, audio)
├─ orchestrator/     # the lore wizard's step-by-step logic
├─ scripts/          # your episode scripts (one sentence = one scene)
├─ overlays/         # like/subscribe banner images
└─ output/           # finished videos
```

---

## Good to know

- **Costs:** only the lore-narration flow bills anything. AI images are cheap; the
  **video animation (Veo) bills per second**, so the app makes you test one clip and
  confirm before a batch, and never re-renders what's done. The montage and gameplay
  flows are free to run.
- **Speaker labels (optional):** to colour gameplay captions by speaker you need a free
  HuggingFace token (`HF_TOKEN` in `.env`) and to accept the model licence once. Without
  it, captions still work — just in a single colour.
- **GPU (optional):** gameplay transcription is much faster on an NVIDIA GPU. See
  `requirements-gameplay.txt` for the extra install; on CPU it still runs, just slowly.
- **Everything resumes:** progress is saved to disk, so closing the app mid-way loses
  nothing.

## Want the full details?

This README is an overview. The deep settings, tuning knobs, and design notes all live
**inline in the code** — start with `config.py` and `gameplay/config.py` (every option is
commented), and each module has a docstring explaining what it does. The pipelines also
run from the command line; see the `python -m ...` entry points in `app.py`, `pipeline.py`,
and `fullauto/candidates.py`.
