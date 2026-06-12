"""
Prompt generator — turns a script into ready-to-paste Gemini image prompts and
filter-safe Veo motion prompts, one per sentence (one per beat).

It automates the boring, repeatable parts of asset prep:
  - the master style block (kept identical across all beats for visual cohesion)
  - consistent formatting and beat -> filename mapping (01..NN)
  - a filter-safe Veo motion line for any beat you mark as a clip
  - a linter that flags words likely to trip Gemini/Veo content filters

It deliberately does NOT invent what each scene depicts — that creative call is
yours, and it's where the videos get their quality. You write one short scene
line per beat; the module handles all the boilerplate around it.

Workflow:
  1. python prompt_gen.py scripts/wendigo.txt --style folklore_horror
     -> first run writes prompts/wendigo.beats.txt (a template, one beat per
        sentence, each pre-filled with its sentence for context).
  2. Fill in a `scene:` line for each beat (and set `type:` to still or clip).
  3. Run the same command again
     -> writes prompts/wendigo_prompts.txt with the full prompts to paste.

Styles match what the channel already uses; add your own in STYLES below.
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

ROOT = Path(__file__).parent
PROMPTS_DIR = ROOT / "prompts"
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Master style blocks (identical across all 8 beats = cohesive look) ---
STYLES = {
    "dark_fantasy": (
        "Dark fantasy digital painting, painterly semi-realistic style, "
        "melancholic FromSoft-inspired mood. Muted palette of charcoal black, "
        "ash grey, deep browns and dim ember orange. Cinematic lighting, "
        "volumetric haze, drifting ash and cinders. Vertical 9:16 composition, "
        "highly detailed, atmospheric, no text, no watermark. Leave the lower "
        "third darker and uncluttered for subtitles. Scene:"
    ),
    "folklore_horror": (
        "Dark folklore horror digital painting, painterly semi-realistic style, "
        "bleak and atmospheric. Muted palette of frozen blue-grey, black, bone "
        "white, with faint cold moonlight. Volumetric mist, falling snow, deep "
        "shadow. Vertical 9:16 composition, highly detailed, eerie, no text, no "
        "watermark. Keep the main subject centered with the lower third darker "
        "for subtitles. Scene:"
    ),
}

# Filter-safe default motion (pure physics — no emotion/distress words).
DEFAULT_MOTION = (
    "Subtle cinematic motion. Slow, steady camera push-in. Gentle atmospheric "
    "movement such as drifting particles and soft haze. The subject stays "
    "stable; do not distort faces, bodies, or objects. Calm, no fast motion."
)

# Words that have tripped Gemini/Veo filters — warn (don't block) so you can
# rephrase before wasting a generation. These are SUGGESTIONS, not bans.
RISKY_WORDS = [
    "kneeling", "huddled", "collapsed", "cowering", "crying", "weeping",
    "wound", "wounded", "bleeding", "blood", "bloody", "gore", "gory",
    "corpse", "corpses", "body", "bodies", "flesh", "mutilated", "severed",
    "child", "children", "kid", "infant", "baby",
    "starving", "suffering", "agony", "screaming", "terror", "pleading",
    "devastating", "sorrowful", "dying", "torture",
]


def split_sentences(text: str) -> list[str]:
    """Same sentence splitter the pipeline uses, so beats line up 1:1 with the
    scenes the pipeline will build."""
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def write_beats_template(sentences: list[str], path: Path) -> None:
    lines = [
        "# Beat sheet — fill in a `scene:` line for each beat below.",
        "# - type:  still  (Ken Burns animates it; free, never slows) ",
        "#          clip   (generate motion in Veo; use for dynamic beats only)",
        "# - scene: a short visual description of what to depict (your creative call).",
        "# - motion (optional, clips only): override the default Veo motion line.",
        "# The sentence each beat narrates is shown for context. Keep scenes",
        "# atmospheric, not graphic, and avoid the filter-trigger words the",
        "# generator will warn you about.",
        "",
    ]
    for i, s in enumerate(sentences, 1):
        lines += [
            f"# sentence {i}: {s}",
            f"[{i}] type: still",
            "scene: ",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_beats(path: Path) -> list[dict]:
    beats: list[dict] = []
    cur: dict | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("#") or not line:
            continue
        m = re.match(r"^\[(\d+)\]\s*type:\s*(still|clip)\s*$", line, re.I)
        if m:
            if cur:
                beats.append(cur)
            cur = {"n": int(m.group(1)), "type": m.group(2).lower(),
                   "scene": "", "motion": ""}
            continue
        if cur is None:
            continue
        if line.lower().startswith("scene:"):
            cur["scene"] = line[len("scene:"):].strip()
        elif line.lower().startswith("motion:"):
            cur["motion"] = line[len("motion:"):].strip()
    if cur:
        beats.append(cur)
    return beats


def lint(text: str) -> list[str]:
    low = text.lower()
    return [w for w in RISKY_WORDS if re.search(rf"\b{re.escape(w)}\b", low)]


def build_prompts(beats: list[dict], style_block: str, sentences: list[str]) -> str:
    out = ["=" * 64,
           "GEMINI / VEO PROMPTS — paste each into the matching tool.",
           "Name your outputs 01..NN to match the beat numbers.",
           "=" * 64, ""]
    warnings: list[str] = []
    for b in beats:
        n = b["n"]
        tag = "CLIP" if b["type"] == "clip" else "STILL"
        scene = b["scene"].strip() or "(scene not filled in yet)"
        out.append(f"----- Beat {n:02d}  [{tag}]  -> file {n:02d} -----")
        sent = sentences[n - 1] if n - 1 < len(sentences) else ""
        if sent:
            out.append(f"# narrates: {sent}")
        out.append("")
        out.append("GEMINI IMAGE PROMPT:")
        out.append(f"{style_block} {scene}")
        out.append("")
        if b["type"] == "clip":
            out.append("VEO MOTION PROMPT (filter-safe):")
            out.append(b["motion"].strip() or DEFAULT_MOTION)
            out.append("")
        else:
            out.append("(still — the pipeline's Ken Burns zoom animates this; no Veo needed)")
            out.append("")
        # lint
        hits = lint(scene + " " + b.get("motion", ""))
        if hits:
            warnings.append(f"  Beat {n:02d}: possible filter triggers -> {', '.join(hits)}")
        out.append("")

    if warnings:
        out += ["=" * 64,
                "FILTER WARNINGS (these words have tripped Gemini/Veo before;",
                "consider rephrasing to physical/atmospheric description):",
                *warnings,
                "=" * 64]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Generate Gemini/Veo prompts from a script.")
    ap.add_argument("script", help="path to a script .txt (one sentence per beat)")
    ap.add_argument("--style", default="dark_fantasy", choices=sorted(STYLES),
                    help="master style block to use")
    args = ap.parse_args()

    script_path = Path(args.script)
    name = script_path.stem
    sentences = split_sentences(script_path.read_text(encoding="utf-8"))
    beats_path = PROMPTS_DIR / f"{name}.beats.txt"

    if not beats_path.exists():
        write_beats_template(sentences, beats_path)
        print(f"Wrote beat-sheet template: {beats_path}")
        print(f"  {len(sentences)} beats (one per sentence).")
        print("  Fill in each `scene:` line, set type: still/clip, then re-run.")
        return

    beats = parse_beats(beats_path)
    if len(beats) != len(sentences):
        print(f"  [WARNING] beat sheet has {len(beats)} beats but script has "
              f"{len(sentences)} sentences — they should match 1:1.")
    out_path = PROMPTS_DIR / f"{name}_prompts.txt"
    out_path.write_text(build_prompts(beats, STYLES[args.style], sentences), encoding="utf-8")
    n_clip = sum(1 for b in beats if b["type"] == "clip")
    print(f"Wrote prompts: {out_path}")
    print(f"  {len(beats)} beats ({n_clip} clips, {len(beats) - n_clip} stills), style '{args.style}'.")


if __name__ == "__main__":
    main()
