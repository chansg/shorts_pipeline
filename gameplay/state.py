"""Per-clip state for the gameplay pipeline.

Like `orchestrator.state.Episode`, stage completion is DERIVED FROM DISK so a
re-run resumes where it left off and never redoes a finished stage. Everything
for one clip lives under output/gameplay/<name>/:

    source.<ext>      the imported (copied) source clip
    transcript.json   cached WhisperX result (the editable transcript)
    captions.ass      burned subtitle file
    reframed.mp4       9:16 blur-pad
    fx.mp4             after effects (only if effects were enabled)
    captioned.mp4      after caption burn
    <name>_short.mp4   final Short (also copied to output/gameplay/)
"""
from __future__ import annotations

import re
from pathlib import Path

from gameplay import config as gconf


def slugify(text: str, fallback: str = "clip") -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return s[:40] or fallback


class GameplayClip:
    def __init__(self, name: str):
        self.name = slugify(name)
        self.dir = gconf.GAMEPLAY_DIR / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.transcript_path = self.dir / "transcript.json"
        self.ass_path = self.dir / "captions.ass"
        self.reframed_path = self.dir / "reframed.mp4"
        self.fx_path = self.dir / "fx.mp4"
        self.captioned_path = self.dir / "captioned.mp4"
        self.final_path = self.dir / f"{self.name}_short.mp4"

    def source_path(self) -> Path | None:
        for p in sorted(self.dir.glob("source.*")):
            return p
        return None

    # ---- stage completion (derived from disk) ----

    def has_source(self) -> bool:
        return self.source_path() is not None

    def has_transcript(self) -> bool:
        return self.transcript_path.exists()

    def is_built(self) -> bool:
        return self.final_path.exists()

    def status(self) -> dict:
        return {
            "name": self.name,
            "source": self.has_source(),
            "transcript": self.has_transcript(),
            "built": self.is_built(),
        }
