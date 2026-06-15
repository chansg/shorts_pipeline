"""Gameplay-to-Shorts pipeline — a second, parallel pipeline alongside the lore
Short builder.

Manual mode (build fully): a pre-trimmed gameplay clip ->
  WhisperX transcribe + diarize + word-align (single-speaker fallback)
  -> editable transcript gate (fix ASR, name/recolour speakers)
  -> reframe to 9:16 (blur-pad)
  -> burn per-speaker captions (modules.karaoke_captions, 4-tuples)
  -> optional effects (punch-zoom, shake)
  -> like/subscribe overlay
  -> export 9:16 Short.

Full-auto mode (experimental, isolated): ingest a long video, detect + categorise
highlight moments, auto-cut candidates, feed each through the manual backend.

The lore pipeline (pipeline.py / app.py wizard) is untouched by this package.
"""
