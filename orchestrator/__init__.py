"""Orchestration layer: drives the i2v half (stills + Veo) and the shorts half
(TTS + captions + assemble) end to end, with per-episode state so the GUI can
gate stages, resume runs, and never re-bill work that already exists on disk.
"""
