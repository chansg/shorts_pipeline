"""Full-Auto Experiment — the experimental long-form processor.

Drop in raw YouTube footage or a long gameplay VOD; parse → categorise → auto-cut
highlights → assemble into a **16:9 YouTube video** (NOT a 9:16 Short). This is
distinct from the manual Gaming pipeline (gameplay/), which turns a pre-trimmed clip
into a vertical Short.

Relocated out of gameplay/ so it stands on its own and does NOT call the 9:16 Shorts
backend (blur-pad reframe, like/subscribe overlay, karaoke captioner, 9:16 export).
It still reuses shared, aspect-agnostic infra from gameplay/ (transcription, config,
the audio-energy envelope) — the dependency is one-directional: fullauto -> gameplay
shared utils; the manual gameplay pipeline never imports fullauto.
"""
