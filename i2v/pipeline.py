"""Pipeline orchestration.

Walks the selected clips, hands each to the provider, and records what
happened. Idempotent: an existing output is skipped unless force=True, so
re-runs only fill in the gaps (Veo costs money per clip — you don't want to
re-render the seven that already succeeded).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from i2v.config import ClipSpec
from i2v.providers.base import VideoProvider

log = logging.getLogger("i2v.pipeline")


@dataclass
class ClipResult:
    name: str
    image: str
    status: str            # "rendered" | "skipped" | "failed"
    output: str | None = None
    error: str | None = None
    seconds: float | None = None


def run(
    provider: VideoProvider,
    clips: list[ClipSpec],
    images_dir: Path,
    out_dir: Path,
    force: bool = False,
    retries: int = 2,
) -> list[ClipResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[ClipResult] = []

    for spec in clips:
        image_path = images_dir / spec.image
        out_path = out_dir / spec.output_filename()

        if not image_path.exists():
            log.error("Missing image: %s", image_path)
            results.append(ClipResult(spec.name, spec.image, "failed",
                                      error=f"image not found: {image_path}"))
            continue

        if out_path.exists() and not force:
            log.info("Skipping %s (already rendered). Use --force to overwrite.", spec.name)
            results.append(ClipResult(spec.name, spec.image, "skipped", output=str(out_path)))
            continue

        result = _render_with_retries(provider, image_path, spec, out_path, retries)
        results.append(result)

    _write_run_manifest(out_dir, results)
    return results


def _render_with_retries(
    provider: VideoProvider,
    image_path: Path,
    spec: ClipSpec,
    out_path: Path,
    retries: int,
) -> ClipResult:
    last_err: Exception | None = None
    for attempt in range(1, retries + 2):  # initial try + `retries` retries
        start = time.monotonic()
        try:
            provider.render(image_path, spec, out_path)
            return ClipResult(spec.name, spec.image, "rendered",
                              output=str(out_path), seconds=round(time.monotonic() - start, 1))
        except Exception as exc:  # noqa: BLE001 - we want to retry on anything transient
            last_err = exc
            wait = min(30, 5 * attempt)
            log.warning("Attempt %d/%d failed for %s: %s (retrying in %ds)",
                        attempt, retries + 1, spec.name, exc, wait)
            time.sleep(wait)

    log.error("Giving up on %s: %s", spec.name, last_err)
    return ClipResult(spec.name, spec.image, "failed", error=str(last_err))


def _write_run_manifest(out_dir: Path, results: list[ClipResult]) -> None:
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": [asdict(r) for r in results],
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rendered = sum(r.status == "rendered" for r in results)
    skipped = sum(r.status == "skipped" for r in results)
    failed = sum(r.status == "failed" for r in results)
    log.info("Done: %d rendered, %d skipped, %d failed", rendered, skipped, failed)
