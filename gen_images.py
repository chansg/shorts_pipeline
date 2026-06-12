"""
Image generator — calls the Gemini (Imagen) image API to create scene images
from your filled beat sheet, saving candidates for you to choose from.

Why this exists:
  - Images come out CLEAN (no app sparkle) because they're generated via the API.
  - They're generated NATIVELY at 9:16, so subjects aren't cropped out later.
  - It automates the tedious fetch, while keeping the creative pick in your hands:
    it saves 2-3 CANDIDATES per beat to a review folder, and you copy the best
    one into assets/images/ as 01.png, 02.png, ...

Workflow:
  1. python prompt_gen.py scripts/NAME.txt --style ...   (fill the beat sheet)
  2. Get a Gemini API key from https://aistudio.google.com/apikey
     and add it to .env:   GEMINI_API_KEY=your_key_here
  3. python gen_images.py scripts/NAME.txt --style ... [--candidates 3]
     -> writes prompts/NAME_candidates/01_a.png, 01_b.png, ...
  4. Review the folder, copy your favourite of each beat into assets/images/
     named 01.png .. NN.png (clip beats: still generate a frame here for
     reference, then animate separately in Veo).

Notes:
  - This is pay-per-image on your Google Cloud billing (~$0.04/image), SEPARATE
    from a Gemini app subscription. Eight beats x3 candidates ~= $1 per video.
  - Model IDs and SDK details move; if a call fails, check the current model
    name at https://ai.google.dev/gemini-api/docs/imagen and pass --model.
  - Requires: pip install google-genai python-dotenv
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# Reuse the SAME prompt-building logic as prompt_gen so images match the prompts.
import prompt_gen

ROOT = Path(__file__).parent
PROMPTS_DIR = ROOT / "prompts"

DEFAULT_MODEL = "imagen-4.0-generate-001"   # override with --model if it changes
DEFAULT_ASPECT = "9:16"                      # vertical: no post-crop needed


def _load_api_key() -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except Exception:
        pass
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("No API key found. Add GEMINI_API_KEY=... to .env "
                 "(get one at https://aistudio.google.com/apikey).")
    return key


def main():
    ap = argparse.ArgumentParser(description="Generate scene image candidates via the Gemini/Imagen API.")
    ap.add_argument("script", help="path to the script .txt (to locate its beat sheet)")
    ap.add_argument("--style", default="dark_fantasy", choices=sorted(prompt_gen.STYLES))
    ap.add_argument("--candidates", type=int, default=3, help="images per beat (default 3)")
    ap.add_argument("--aspect", default=DEFAULT_ASPECT, help="aspect ratio (default 9:16)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Imagen model id")
    ap.add_argument("--only", default="", help="comma-separated beat numbers to (re)generate, e.g. 3,7")
    args = ap.parse_args()

    name = Path(args.script).stem
    beats_path = PROMPTS_DIR / f"{name}.beats.txt"
    if not beats_path.exists():
        sys.exit(f"No beat sheet at {beats_path}. Run: python prompt_gen.py {args.script} --style {args.style}")

    beats = prompt_gen.parse_beats(beats_path)
    if not beats:
        sys.exit("Beat sheet has no beats — fill in the scene lines first.")
    only = {int(x) for x in args.only.split(",") if x.strip().isdigit()} if args.only else None

    style_block = prompt_gen.STYLES[args.style]

    # Import + client created lazily so --help works without the SDK installed.
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        sys.exit("google-genai not installed. Run: pip install google-genai python-dotenv")

    client = genai.Client(api_key=_load_api_key())

    out_dir = PROMPTS_DIR / f"{name}_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    letters = "abcdefghij"
    made, failed = 0, []

    for b in beats:
        n = b["n"]
        if only and n not in only:
            continue
        scene = (b.get("scene") or "").strip()
        if not scene:
            print(f"  Beat {n:02d}: no scene filled in — skipping.")
            continue
        full_prompt = f"{style_block} {scene}"
        print(f"Beat {n:02d} [{b['type']}]: generating {args.candidates} candidate(s)...")
        try:
            resp = client.models.generate_images(
                model=args.model,
                prompt=full_prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=args.candidates,
                    aspect_ratio=args.aspect,
                    # Leave provenance (SynthID) at the API default; do not disable.
                ),
            )
        except Exception as e:
            print(f"  [ERROR] Beat {n:02d} request failed: {e}")
            failed.append((n, str(e)[:120]))
            continue

        imgs = getattr(resp, "generated_images", None) or []
        if not imgs:
            # Usually a content-filter block: report the reason if present.
            reason = ""
            for ga in (imgs or []):
                reason = getattr(ga, "rai_filtered_reason", "") or reason
            print(f"  [FILTERED] Beat {n:02d}: no images returned (likely content filter). "
                  f"Rephrase the scene to physical/atmospheric description.")
            failed.append((n, "filtered / no images"))
            continue

        for i, ga in enumerate(imgs):
            if getattr(ga, "image", None) is None:
                continue
            path = out_dir / f"{n:02d}_{letters[i] if i < len(letters) else i}.png"
            try:
                ga.image.save(str(path))
                made += 1
            except Exception:
                # Fallback: write raw bytes.
                data = getattr(ga.image, "image_bytes", None)
                if data:
                    path.write_bytes(data); made += 1
        print(f"  saved {len(imgs)} -> {out_dir.name}/")

    print(f"\nDone: {made} image(s) in {out_dir}")
    print("Review them and copy your pick of each beat into assets/images/ as 01.png .. NN.png")
    if failed:
        print("\nBeats needing attention:")
        for n, why in failed:
            print(f"  Beat {n:02d}: {why}")


if __name__ == "__main__":
    main()
