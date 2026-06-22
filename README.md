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
   one voice → single-speaker. A **Speakers** dropdown pins the exact count
   (`Auto`/1–5 → `DIARIZE_NUM_SPEAKERS`): pyannote's auto-clustering is unreliable on
   short, cross-talky clips, so telling it the number is the strongest lever when it
   mislabels. Word→speaker assignment is by **summed overlap per speaker** (robust to
   the tiny alternating turns pyannote emits during overlap).

   > **Cross-talk is a hard limit, not a bug.** When two people talk *at the same time*,
   > Whisper transcribes the mixed audio as **one** stream of words, so those words can
   > only be given to **one** speaker — and on short clips pyannote may even cluster a
   > whole utterance to the wrong person. Pinning the speaker count and the summed-overlap
   > assignment reduce errors, but they **cannot separate simultaneous voices**. Truly
   > automating overlap needs a **source-separation** stage (split the mix into one
   > waveform per speaker, transcribe each, then merge) — a heavier model not yet wired
   > in. Until then, fix the residual mislabels in the transcript grid (rename a speaker,
   > or **Bulk edits → Assign speaker to rows** for a whole stretch).
3. **Transcript gate** — fix ASR errors in the editable grid, rename speakers
   (`SPEAKER_00` → "Chan") and pick their colours (shown as inline swatches).
   *These rows are the captions.* The grid gives the words a **wide, wrapped** column
   with the numeric/flag columns kept tight, and a bounded scroll height so a long
   transcript doesn't push the build controls off-screen. Each row has a **censor**
   checkbox (auto-ticked from the word-list — see Profanity censor); right-click
   inserts/deletes rows (new rows with blank timing are timed automatically), and **↺
   Revert grid** reloads the saved transcript to discard accidental edits. The **Bulk
   edits & caption preview** panel
   adds: multi-row speaker reassignment (fix a stretch the diariser mislabelled),
   find/replace (a misheard name, fixed once), merge/split rows, and a **Re-apply
   captions** preview that re-renders just the caption track — no re-transcription.
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
- **VAD merge window** — `WHISPERX_CHUNK_SIZE` (seconds) is the **biggest lever**
  against dropout. WhisperX merges contiguous speech into windows decoded in one
  batched pass; its default (30s) lets a long talking run become a single window the
  model abandons after a few words (measured: 26s of chatter → **5 words**). A smaller
  window forces several independent decodes. On a dense 28.6s ARAM clip — VAD already
  covering 92% — the decoder still gave up *inside* the covered windows at 8s (**30
  words**); **6s** recovered **64 words** (2.1×), 4s over-fragmented back to 49. So the
  default is **6**. Lower = more coverage, more fragmentation. (Re-transcribe an
  existing clip to benefit — a cached `transcript.json` is not re-cut automatically.)
- **VAD** sensitivity is exposed as `WHISPERX_VAD_ONSET` / `WHISPERX_VAD_OFFSET`
  (lower onset = recover more speech, at the risk of decoding loud non-speech).
- **Language** — `WHISPERX_LANGUAGE` (default `"en"`) pins the transcription language.
  WhisperX's per-clip auto-detect mis-fires on noisy gameplay intros (English audio
  came out as **Danish**, which also loads the wrong alignment model). Set another
  whisper code (`"es"`, `"de"`, …) or `None` to restore auto-detection.
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

### Output quality & layout

Built Shorts are encoded for phone sharpness via one shared helper
(`gameplay/encode.py`, used by manual + full-auto):

- **Quality-targeted final encode** — `libx264` at constant quality **`OUTPUT_CRF`
  (default 18)**, `OUTPUT_PRESET` (`slow`), H.264 High profile, `yuv420p`, and
  `+faststart` for mobile streaming. CRF 18 lands well above 10 Mbps on motion content
  by itself (set 16 for more headroom).
- **No compounding loss** — a clip now goes through **2 encodes** (a cached
  near-lossless reframe at `INTERMEDIATE_CRF` 14 + one final encode; +1 only if a
  like/subscribe overlay is used), down from up to four. Effects + captions are
  composed into a single pass. The build log reports the pass count.
- **Layout modes** (`REFRAME_MODE`, also a GUI dropdown) reclaim the bitrate the
  blur-pad spends on blurred bars:
  - `fill` *(default, recommended)* — cover + crop so the gameplay fills the whole
    1080×1920 at full resolution (**sharpest**, no wasted blur). Loses the far
    horizontal edges; bias the crop with `REFRAME_CROP_X_OFFSET` (0=left, 0.5=centre,
    1=right) — e.g. nudge to keep the ARAM minimap. `REFRAME_CROP_Y_OFFSET` and
    `REFRAME_FILL_FRACTION` (zoom past cover) tune it; all three are GUI sliders.
  - `fit_crop` — `fill` at fraction 1.0, centred.
  - `blur_pad` — full frame centred over a blurred fill (no crop; the old default).
  - `zoom_blur` — blur-pad with the gameplay band enlarged by `ZOOM_BLUR_SCALE`.

  With `fill`, captions default to `CAPTION_POS_Y_FRAC` 0.72 — a readable lower band
  that sits **above** the centred (~0.84) like/subscribe overlay, so caption and banner
  never fight the game's centre HUD or each other. It's a GUI slider per build.

### Caption timing & chunking

Captions can only be as good as the transcript, and strict one-word-at-a-time karaoke
*magnifies* any residual ASR drift — sparse words flash in and out and read as broken.
Gameplay captions are therefore **phrase-chunked** by `gameplay/captioning.py`, a pure
transform over the word list (the renderer and the `(text,start,end,speaker)` contract
are unchanged):

- **`CAPTION_CHUNK_MODE`** — `"phrase"` *(default)* groups consecutive same-speaker words
  into one cue that holds for the span of its words; `"word"` keeps the classic one-word
  karaoke. A GUI **Caption style** dropdown picks per build.
- A phrase is capped at **`CAPTION_CHUNK_MAX_WORDS`** (4), **`CAPTION_CHUNK_MAX_WINDOW_S`**
  (1.2s span), and **`CAPTION_CHUNK_MAX_CHARS`** (22) so it stays readable on a phone.
- **`CAPTION_MIN_DUR_S`** (0.4) is the minimum on-screen time per cue (anti-flash) and
  **`CAPTION_MAX_GAP_S`** (0.8) bridges small gaps so a cue doesn't blink off between
  words — both applied by the renderer via `CaptionStyle` (`min_hold_s` / `max_gap`).
- **`CAPTION_OFFSET_S`** (0, a GUI slider) is a global lead(−)/lag(+) nudge — a
  calibration safety valve if a small systematic drift remains after the ASR fixes.

Timing itself comes straight from WhisperX **word-level alignment** (wav2vec2), not
segment times distributed evenly — verified by test. There is no trim between transcribe
and burn (source and reframe share the timeline), so captions carry no constant offset.
Lore captions are untouched (they use the renderer directly, not this transform).

### Profanity censor

Curse words are bleeped in the audio **and** masked in the caption, driven off one
editable word-list (`gameplay/censor.py` + `gameplay/config.py`) and the WhisperX word
spans — so audio and caption censor the same moment. **On by default.**

- **Detection** is case-insensitive: a token is censored if it's in `CENSOR_WORDLIST`
  **or contains a `CENSOR_STEM`** (e.g. `fuck`, `shit`, `bitch`) as a substring — so
  variants and compounds (`fucking`, `bullshit`, `wankers`, `dipshit`) are caught
  automatically without listing each. `CENSOR_ALLOWLIST` guards the clean words a stem
  would otherwise hit ("Shaco", "assassin", "Scunthorpe", "niggle", fire-"retardant").
  Add your slang to the stems/word-list.
- **Audio** `CENSOR_AUDIO_MODE`: `bleep` (1 kHz tone, default — reads as intentional),
  `mute`, or `duck`. Applied in the *same* final encode (no extra pass); for full-auto
  it's applied per cut window (audio-only — no captions there).
- **Caption** `CENSOR_CAPTION_STYLE`: `stars` (`f***`) or `block` (`[bleep]`).
- **Toggle** per build in the GUI: bleep+mask / audio-only / caption-only / off.
- **Editor:** profane text is **auto-censored at build** (typed, edited, or added),
  so a right-click-added row is censored without even ticking the box. The per-row
  **censor** checkbox can ADD censor to a non-listed word (e.g. a name); to KEEP a
  flagged word uncensored, add it to `CENSOR_ALLOWLIST`.
- **Added rows are tickable.** Gradio renders a new row's censor cell as a text box
  (it holds `""` until it carries a real bool), so the checkbox was missing — the grid
  now coerces that column to a real bool on every change, so the checkbox appears and
  auto-ticks profane text. A new row's default `0`/`0` timing (or blank) is treated as
  "added" and **inferred** to sit right after the previous word, so its bleep/mask lands
  in place instead of at `t=0`. Adjacent hits merge; a flagged word with no timestamp is
  masked in the caption but skipped for audio (logged).

### Narrated hook (story-time opener)

Optionally read **one opening hook line** aloud over the start of the Short (the TikTok
"story-time" format — e.g. *"The time I got ganked by 3 people playing Yone"*). Off by
default; per-build toggle in the gameplay tab.

- Type the line, tick **Narrated hook**, optionally set a voice id (defaults to
  `HOOK_VOICE` = the lore pipeline's ElevenLabs voice) and **Preview voice**. The TTS
  reuses the lore ElevenLabs client and is **cached by (line, voice)** — re-builds and
  repeat previews don't re-bill.
- The narration plays from the start; while she speaks the **game bed is muted +
  muffled** (gain `HOOK_MUTE_GAIN` 0.05 through a `HOOK_MUFFLE_HZ` 500 Hz low-pass — a
  dull, quiet murmur, not just a duck) so the voice is clean, then it **swells back** to
  full over `DUCK_RELEASE_S` (0.3 s) once the line ends.
- The hook is captioned in its own reserved **NARRATOR colour** (`NARRATOR_CAPTION_COLOR`,
  cyan), and **game-speech captions are paused** for the narration window — a word that
  falls under the hook is dropped so nothing renders on top of the hook caption (the bed
  is inaudible there anyway). Normal transcript captions resume the moment she finishes.
  Hook caption timings are distributed evenly across the narration (no per-word
  ElevenLabs timings).
- It rides the **same final encode** as captions/effects/censor — **zero extra passes**
  (censor + duck/mix compose into one audio graph). If ElevenLabs errors, the build
  completes **without** narration and logs why. Manual mode only.

### Captions / colours

Captions reuse `modules/karaoke_captions` — the same active-word renderer as the
lore path. The gameplay path passes 4-tuples `(text, start, end, speaker)`, which
drive per-speaker colour (explicit hex wins; otherwise a 6-colour palette is
auto-assigned in order of appearance). The bundled **Anton** font is used the same
way (via `fontsdir`).

## Full-Auto: Highlight clips → manual (audio-reaction + HUD)

The headline full-auto flow takes a **lengthy gameplay VOD** (hours of League/ARAM,
NVIDIA captures) and returns **ranked candidate clips**, each a **generous raw 9:16
segment** you then refine in the Gaming tab. It targets **funny / rage** moments scored
from **voice reactions + HUD events**, and needs **no transcript or LLM** — so it's
robust and fast. (`fullauto/reaction.py`, `fullauto/hud.py`, `fullauto/clips.py`.)

**Design: audio reaction LEADS, HUD CONFIRMS.**

1. **Audio reaction (robust core).** The mixed mono track is **streamed** through ffmpeg
   (never fully decoded — handles hours within bounded memory), band-passed to the
   **vocal range** (`REACTION_BAND_HZ` ~300–3400 Hz), and folded to a per-window RMS.
   Each window is scored by **suddenness above a rolling baseline** — an **onset/attack**
   term (`REACTION_ONSET_WEIGHT`) dominates a sustained-energy term, so a sharp “WHAT?!”
   outscores a long teamfight roar (gradual onset; the baseline catches up). Peaks above
   `REACTION_THRESHOLD`, spaced so one reaction = one peak.
2. **Generous windows.** Each peak frames `[t − PRE_ROLL_S, t + POST_ROLL_S]` — **anchored
   before the spike** (setup + payoff) — and overlapping windows **merge** (so a long fight
   is one candidate, not five). Capped to `MAX_CANDIDATES` (recall-biased: surface more,
   you're the final judge).
3. **HUD scan (isolated, fail-safe booster).** *Within each window only* (cheap), frames
   are sampled and League HUD ROIs (`HUD_ROIS`) read for kill-feed / multikill / ace
   banners. Events add a boost (`final = audio_score × (1 + hud_boost)`; penta > quad >
   … > single). The **entire scan is wrapped** so any failure (no OCR backend, an ffmpeg
   hiccup) yields no events and the audio candidate stands — and `HUD_SCAN_ENABLED=False`
   skips it. A multikill with no reaction never outranks a reaction with no HUD.
4. **Rank → export → manifest.** Candidates are cut + reframed to 9:16 reusing the shared
   `gameplay.reframe`/`encode` (one source of truth — no captions/effects/overlay; those
   come later in manual), and each is dropped in as a **GameplayClip** the Gaming tab can
   open. A `candidates.json` records rank, score, audio-score, HUD events, window, and
   paths so the GUI lists each candidate **with why it was picked**.

In the GUI: upload the VOD, tune the knobs (threshold, pre/post-roll, max, HUD on/off),
**① Detect highlight clips** (the log prints the **score curve** for calibration), then
pick a candidate and **② Refine in manual mode →** loads it straight into the Gaming
uploader.

> **Calibration WILL be needed on real footage.** The detector is tuned on synthetic
> audio; the live thresholds are yours to dial. If candidates are **sparse**, lower
> `REACTION_THRESHOLD` (watch the logged score curve — aim for the threshold to sit
> below the reaction peaks but above chatter). If **teamfights leak in**, raise
> `REACTION_ONSET_WEIGHT` or the threshold. Widen clips with `PRE_ROLL_S`/`POST_ROLL_S`.
> HUD ROIs are 1080p defaults — adjust per capture resolution. The GPU/long-run pass on
> real captures is yours; the unit tests cover the detection logic on synthetic signals.

## Full-Auto Experiment (16:9 YouTube output — the older flow)

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