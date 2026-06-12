"""Provider interface.

Every backend takes an image + ClipSpec and produces an mp4 on disk. Keeping
this contract narrow is what lets you swap Veo for a local model on the 3080,
or Runway/Kling, without touching the orchestrator.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from i2v.config import ClipSpec


class VideoProvider(ABC):
    name: str = "base"

    @abstractmethod
    def render(self, image_path: Path, spec: ClipSpec, out_path: Path) -> Path:
        """Generate a video from a still image and write it to out_path.

        Implementations should be blocking (poll until done) and return the
        path actually written. Raise on unrecoverable failure so the
        orchestrator can decide whether to retry or skip.
        """
        raise NotImplementedError
