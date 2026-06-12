"""Image generation stage with reference-image conditioning.

Runs BEFORE the i2v (image-to-video) stage. For each scene it builds a
Nano Banana 2 request as [ ...reference images..., prompt ] so the model
keeps a consistent art style across every frame and a consistent character
(e.g. Manus) on the scenes where he appears.

Reference model: gemini-3.1-flash-image (Nano Banana 2) — accepts up to 14
reference images and maintains style/character consistency. Falls back
cleanly to gemini-2.5-flash-image if you set --model.

Two kinds of reference, both optional:
  - style_refs (manifest-level): applied to EVERY scene, locks the look.
  - character refs (per-scene): named entries from `characters`, applied only
    to the scenes that list them, locks who someone is across frames.

Manifest additions (see manus_prompts.json):
  {
    "defaults": { "image_model": "gemini-3.1-flash-image",
                  "image_aspect_ratio": "9:16" },
    "style_refs": ["refs/style_dark_fantasy.png"],
    "characters": { "manus": ["refs/manus_ref.png"] },
    "clips": [ { "image": "02.png", "prompt": "...", "refs": ["manus"] }, ... ]
  }
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("i2v.imagegen")

DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image"  # Nano Banana 2

# Prepended to every prompt when references are attached, so the model knows
# what the attached images are FOR (style vs identity).
REF_PREAMBLE = (
    "Use the attached reference image(s) as visual guidance. "
    "Match their art style, palette, lighting and rendering exactly. "
    "Where a character reference is attached, keep that character's design, "
    "silhouette and features consistent. Then create the following scene:\n"
)


@dataclass
class GenSpec:
    image: str                 # output filename, e.g. "02.png"
    name: str
    prompt: str
    aspect_ratio: str
    ref_paths: list[Path]      # fully-resolved style + character refs


def _load_pil(path: Path):
    from PIL import Image
    img = Image.open(path)
    img.load()
    return img


def _resolve_manifest(manifest_path: Path, refs_dir: Path) -> tuple[str, list[GenSpec]]:
    """Parse the manifest and resolve every reference to an on-disk path."""
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    defaults = data.get("defaults", {})
    model = defaults.get("image_model", DEFAULT_IMAGE_MODEL)
    aspect = defaults.get("image_aspect_ratio", defaults.get("aspect_ratio", "9:16"))
    style_suffix = defaults.get("style_suffix", "").strip()

    # Manifest-level style refs (applied to every scene).
    style_refs = [_resolve_ref(r, refs_dir) for r in data.get("style_refs", [])]

    # Named character ref groups.
    characters: dict[str, list[Path]] = {
        name: [_resolve_ref(r, refs_dir) for r in paths]
        for name, paths in data.get("characters", {}).items()
    }

    specs: list[GenSpec] = []
    for raw in data["clips"]:
        # Per-scene refs: each entry is either a character name or a direct path.
        scene_refs: list[Path] = []
        for ref in raw.get("refs", []):
            if ref in characters:
                scene_refs.extend(characters[ref])
            else:
                scene_refs.append(_resolve_ref(ref, refs_dir))

        all_refs = style_refs + scene_refs
        if len(all_refs) > 14:
            log.warning("%s has %d refs; Nano Banana 2 caps at 14, trimming.",
                        raw["name"], len(all_refs))
            all_refs = all_refs[:14]

        base_prompt = raw["prompt"].strip()
        full = f"{base_prompt} {style_suffix}".strip() if style_suffix else base_prompt
        specs.append(GenSpec(
            image=raw["image"], name=raw["name"], prompt=full,
            aspect_ratio=raw.get("image_aspect_ratio", aspect), ref_paths=all_refs,
        ))
    return model, specs


def _resolve_ref(ref: str, refs_dir: Path) -> Path:
    p = Path(ref)
    if not p.is_absolute() and not p.exists():
        p = refs_dir / Path(ref).name if (refs_dir / Path(ref).name).exists() else Path(ref)
    if not p.exists():
        raise FileNotFoundError(f"Reference image not found: {ref} (looked at {p})")
    return p


class NanoBananaProvider:
    """Gemini image generation with reference-image conditioning."""

    def __init__(self, api_key: str, model: str = DEFAULT_IMAGE_MODEL) -> None:
        from google import genai  # lazy import
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _config(self, aspect_ratio: str):
        """Build a config that requests the right aspect ratio, degrading
        gracefully if the installed SDK doesn't expose image_config."""
        from google.genai import types
        try:
            return types.GenerateContentConfig(
                image_config=types.ImageConfig(aspect_ratio=aspect_ratio)
            )
        except (AttributeError, TypeError):
            return None  # older SDK: aspect is steered via the prompt text instead

    def generate(self, spec: GenSpec, out_path: Path) -> Path:
        # contents = [ ...reference images..., prompt ]
        contents: list = [_load_pil(p) for p in spec.ref_paths]
        prompt = spec.prompt
        if spec.ref_paths:
            prompt = REF_PREAMBLE + prompt
        if "9:16" in spec.aspect_ratio:
            prompt += "\nVertical 9:16 portrait composition, full-frame."
        contents.append(prompt)

        kwargs = {"model": self.model, "contents": contents}
        cfg = self._config(spec.aspect_ratio)
        if cfg is not None:
            kwargs["config"] = cfg

        log.info("Generating %s (%s, %d ref%s)", spec.image, self.model,
                 len(spec.ref_paths), "" if len(spec.ref_paths) == 1 else "s")
        response = self.client.models.generate_content(**kwargs)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        for part in response.parts:
            if getattr(part, "inline_data", None) is not None:
                part.as_image().save(str(out_path))
                log.info("Saved %s", out_path)
                return out_path
        raise RuntimeError(f"{spec.name}: no image returned (prompt may have been filtered)")


def run(
    api_key: str,
    manifest_path: Path,
    images_dir: Path,
    refs_dir: Path,
    model: str | None = None,
    only: list[str] | None = None,
    force: bool = False,
    retries: int = 2,
) -> list[dict]:
    resolved_model, specs = _resolve_manifest(manifest_path, refs_dir)
    provider = NanoBananaProvider(api_key=api_key, model=model or resolved_model)

    if only:
        wanted = {o.lower().removesuffix(".png") for o in only}
        specs = [s for s in specs if s.name.lower() in wanted
                 or s.image.lower().removesuffix(".png") in wanted]

    images_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for spec in specs:
        out = images_dir / spec.image
        if out.exists() and not force:
            log.info("Skipping %s (exists). Use --force to regenerate.", spec.image)
            results.append({"name": spec.name, "status": "skipped", "output": str(out)})
            continue
        results.append(_gen_with_retries(provider, spec, out, retries))

    rendered = sum(r["status"] == "generated" for r in results)
    log.info("Image gen done: %d generated, %d skipped, %d failed",
             rendered, sum(r["status"] == "skipped" for r in results),
             sum(r["status"] == "failed" for r in results))
    return results


def _gen_with_retries(provider: NanoBananaProvider, spec: GenSpec,
                      out: Path, retries: int) -> dict:
    last = None
    for attempt in range(1, retries + 2):
        try:
            provider.generate(spec, out)
            return {"name": spec.name, "status": "generated", "output": str(out)}
        except Exception as exc:  # noqa: BLE001
            last = exc
            log.warning("Attempt %d failed for %s: %s", attempt, spec.image, exc)
            time.sleep(min(20, 4 * attempt))
    return {"name": spec.name, "status": "failed", "error": str(last)}


def _cli() -> int:
    import argparse
    from i2v import config

    p = argparse.ArgumentParser(prog="i2v.imagegen",
                                description="Reference-conditioned image generation (Nano Banana 2).")
    p.add_argument("--manifest", type=Path, default=Path("manus_prompts.json"))
    p.add_argument("--images", type=Path, default=Path("images"))
    p.add_argument("--refs", type=Path, default=Path("refs"))
    p.add_argument("--model", help="Override image model (e.g. gemini-2.5-flash-image).")
    p.add_argument("--only", nargs="*", help="Generate only these scene names/filenames.")
    p.add_argument("--force", action="store_true", help="Regenerate even if the PNG exists.")
    p.add_argument("--api-key")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S",
    )
    config.load_dotenv()  # read GEMINI_API_KEY from .env — this is what was missing
    api_key = config.resolve_api_key(args.api_key)
    run(api_key=api_key, manifest_path=args.manifest, images_dir=args.images,
        refs_dir=args.refs, model=args.model, only=args.only, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
