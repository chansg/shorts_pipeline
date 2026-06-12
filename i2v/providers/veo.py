"""Veo backend (Gemini Developer API, google-genai SDK).

Implements image-to-video via client.models.generate_videos, polls the
long-running operation, downloads the result and saves it.

Verified against the google-genai SDK image-to-video flow:
  - types.Image.from_file(location=...) to load the still
  - client.models.generate_videos(model, prompt, image, config=...)
  - poll operation.done / client.operations.get(operation)
  - client.files.download(file=video) then video.save(path)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from i2v.config import ClipSpec
from i2v.providers.base import VideoProvider

log = logging.getLogger("i2v.veo")

# veo-3.1 supports native portrait (9:16) and image-to-video. Override via env/CLI
# if a newer/cheaper variant is preferred (e.g. a "fast" tier).
DEFAULT_MODEL = "veo-3.1-generate-preview"


class VeoProvider(VideoProvider):
    name = "veo"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        poll_interval: int = 15,
        timeout: int = 600,
        enhance_prompt: bool = False,  # veo-3.1-generate-preview rejects this param
    ) -> None:
        # Imported lazily so the rest of the pipeline (manifest validation,
        # stitching) works without the SDK installed.
        from google import genai  # noqa: WPS433

        self._genai = genai
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.enhance_prompt = enhance_prompt

    def _build_config(self, spec: ClipSpec):
        from google.genai import types

        # Only set keys that are present, so unsupported kwargs never get sent.
        cfg: dict = {
            "number_of_videos": 1,
            "aspect_ratio": spec.aspect_ratio,
        }
        if self.enhance_prompt:  # omitted by default — not supported on veo-3.1
            cfg["enhance_prompt"] = self.enhance_prompt
        if spec.negative_prompt:
            cfg["negative_prompt"] = spec.negative_prompt
        if spec.duration_seconds:
            cfg["duration_seconds"] = spec.duration_seconds
        return types.GenerateVideosConfig(**cfg)

    def render(self, image_path: Path, spec: ClipSpec, out_path: Path) -> Path:
        from google.genai import types

        image = types.Image.from_file(location=str(image_path))
        # Animate with the motion-specific prompt if the manifest provides one;
        # otherwise reuse the still-generation prompt.
        anim_prompt = spec.motion_prompt or spec.prompt
        log.info("Submitting %s (%s, %ss, %s)%s", spec.name, self.model,
                 spec.duration_seconds, spec.aspect_ratio,
                 " [motion_prompt]" if spec.motion_prompt else "")

        operation = self.client.models.generate_videos(
            model=self.model,
            prompt=anim_prompt,
            image=image,
            config=self._build_config(spec),
        )

        deadline = time.monotonic() + self.timeout
        while not operation.done:
            if time.monotonic() > deadline:
                raise TimeoutError(f"{spec.name}: generation exceeded {self.timeout}s")
            time.sleep(self.poll_interval)
            operation = self.client.operations.get(operation)
            log.debug("polling %s ... done=%s", spec.name, operation.done)

        # Surface API-side errors explicitly rather than crashing on attribute access.
        if getattr(operation, "error", None):
            raise RuntimeError(f"{spec.name}: Veo returned an error: {operation.error}")

        generated = operation.response.generated_videos[0]
        self.client.files.download(file=generated.video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        generated.video.save(str(out_path))
        log.info("Saved %s", out_path)
        return out_path
