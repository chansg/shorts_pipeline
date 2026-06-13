"""Configuration and manifest handling.

Loads the prompt manifest (prompts.json), merges per-clip overrides on top of
defaults, and resolves the API key from the environment. No provider-specific
logic lives here so the same manifest can drive any backend.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ClipSpec:
    """A single image-to-video job after defaults have been merged in."""

    image: str                      # filename, resolved against the images dir
    name: str                       # used for the output filename
    prompt: str                     # full prompt (style suffix already appended); used for still gen
    aspect_ratio: str
    duration_seconds: int
    negative_prompt: str
    motion_prompt: str | None = None  # if set, the i2v stage animates with THIS instead of `prompt`

    def output_filename(self) -> str:
        return f"{self.name}.mp4"


@dataclass
class Manifest:
    defaults: dict
    clips: list[ClipSpec] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        defaults = data.get("defaults", {})
        style_suffix = defaults.get("style_suffix", "").strip()

        clips: list[ClipSpec] = []
        for raw in data["clips"]:
            base_prompt = raw["prompt"].strip()
            full_prompt = f"{base_prompt} {style_suffix}".strip() if style_suffix else base_prompt
            # motion_prompt is used verbatim by the i2v stage (no style suffix — the
            # style already lives in the generated still; this prompt is about motion).
            motion = raw.get("motion_prompt")
            motion = motion.strip() if isinstance(motion, str) and motion.strip() else None
            clips.append(
                ClipSpec(
                    image=raw["image"],
                    name=raw["name"],
                    prompt=full_prompt,
                    aspect_ratio=raw.get("aspect_ratio", defaults.get("aspect_ratio", "9:16")),
                    duration_seconds=int(raw.get("duration_seconds", defaults.get("duration_seconds", 8))),
                    negative_prompt=raw.get("negative_prompt", defaults.get("negative_prompt", "")),
                    motion_prompt=motion,
                )
            )
        return cls(defaults=defaults, clips=clips)

    def select(self, names: list[str] | None) -> list[ClipSpec]:
        """Return clips filtered by name or image filename; all if names is None/empty."""
        if not names:
            return self.clips
        wanted = {n.lower().removesuffix(".png").removesuffix(".mp4") for n in names}
        return [
            c for c in self.clips
            if c.name.lower() in wanted or c.image.lower().removesuffix(".png") in wanted
        ]


# The single source of truth for keys: the .env at the repo root (one level up
# from this package), shared with the shorts_pipeline modules.
REPO_ENV = Path(__file__).resolve().parents[1] / ".env"


def load_dotenv(path: Path | str | None = None) -> None:
    """Minimal .env loader (no python-dotenv dependency). Existing env vars win.
    Defaults to the repo-root .env so both halves of the pipeline share one file."""
    p = Path(path) if path else REPO_ENV
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def resolve_api_key(explicit: str | None = None) -> str:
    """Resolve the Gemini API key from an explicit value or the environment."""
    key = explicit or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "No API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment "
            "or .env file, or pass --api-key."
        )
    return key
