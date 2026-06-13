"""Command-line entry point.

Examples
--------
# Render everything from prompts.json using PNGs in ./images, output to ./output
python -m i2v.cli --images ./images --manifest prompts.json --out ./output

# Render only two clips, then stitch the whole output dir into a reel
python -m i2v.cli --only gothic_city throne_room
python -m i2v.cli --stitch --stitch-mode crossfade

# Dry run: validate the manifest and show what would render, no API calls
python -m i2v.cli --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from i2v import config, pipeline, stitch
from i2v.providers.veo import DEFAULT_MODEL


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="i2v", description="Animate stills with Veo (image-to-video).")
    p.add_argument("--images", type=Path, default=Path("images"), help="Directory of source PNGs.")
    p.add_argument("--manifest", type=Path, default=Path("prompts.json"), help="Prompt manifest.")
    p.add_argument("--out", type=Path, default=Path("output"), help="Output directory.")
    p.add_argument("--only", nargs="*", help="Render only these clip names (or image filenames).")
    p.add_argument("--force", action="store_true", help="Re-render even if output exists.")
    p.add_argument("--retries", type=int, default=2, help="Retries per clip on failure.")
    p.add_argument("--api-key", help="Gemini API key (else GEMINI_API_KEY / GOOGLE_API_KEY).")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Veo model id.")
    p.add_argument("--poll-interval", type=int, default=15, help="Seconds between status polls.")
    p.add_argument("--timeout", type=int, default=600, help="Per-clip timeout in seconds.")
    p.add_argument("--dry-run", action="store_true", help="Validate + list clips, no API calls.")
    p.add_argument("--stitch", action="store_true", help="Stitch outputs into a single reel.")
    p.add_argument("--stitch-mode", choices=["concat", "crossfade"], default="concat")
    p.add_argument("--reel-name", default="reel.mp4", help="Filename for the stitched reel.")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("i2v")

    config.load_dotenv()  # repo-root .env, shared with the shorts pipeline
    manifest = config.Manifest.load(args.manifest)
    clips = manifest.select(args.only)
    if not clips:
        log.error("No clips matched %s", args.only)
        return 2

    order = [c.name for c in manifest.clips]  # canonical reel order = manifest order

    if args.dry_run:
        log.info("Dry run — %d clip(s) selected:", len(clips))
        for c in clips:
            target = args.images / c.image
            mark = "ok" if target.exists() else "MISSING"
            log.info("  [%s] %-18s <- %-8s %ds %s", mark, c.name, c.image,
                     c.duration_seconds, c.aspect_ratio)
        if args.stitch:
            log.info("Would stitch (%s) -> %s", args.stitch_mode, args.out / args.reel_name)
        return 0

    if not args.stitch or args.only:  # render unless this is a pure stitch pass
        from i2v.providers.veo import VeoProvider
        api_key = config.resolve_api_key(args.api_key)
        provider = VeoProvider(
            api_key=api_key, model=args.model,
            poll_interval=args.poll_interval, timeout=args.timeout,
        )
        results = pipeline.run(
            provider=provider, clips=clips, images_dir=args.images,
            out_dir=args.out, force=args.force, retries=args.retries,
        )
        if any(r.status == "failed" for r in results) and not args.stitch:
            return 1

    if args.stitch:
        dest = args.out / args.reel_name
        if args.stitch_mode == "crossfade":
            stitch.crossfade(args.out, order, dest)
        else:
            stitch.concat(args.out, order, dest)
        log.info("Reel written: %s", dest)

    return 0


if __name__ == "__main__":
    sys.exit(main())
