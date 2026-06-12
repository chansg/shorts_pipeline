"""Reference-aware scene-prompt generation -> aria-i2v manifest.

Built on prompt_gen.py (same sentence splitter as the pipeline, same filter
linter, same filter-safe default motion). One script sentence = one scene.

Reference-aware by construction: this only runs AFTER the style lock and
character refs are set, so
  (a) each character ref is wired into the scenes that feature it via `refs`,
  (b) prompts describe COMPOSITION and SUBJECT, not the art style — the style
      is locked by the attached style_refs, so re-describing it just fights
      the reference. The style_suffix kept in defaults is compositional only.

Drafting uses Gemini text (cheap, one call per episode) when a key is present;
otherwise a deterministic fallback turns each sentence into a literal scene
prompt the user can edit in the GUI. Either way the user edits every prompt
in-app before anything is generated.
"""
from __future__ import annotations

import json
import os
import re

import prompt_gen
from orchestrator import state as st
from orchestrator.errors import friendly

# Compositional-only suffix: the LOOK comes from the style refs.
COMPOSITION_SUFFIX = (
    "Vertical 9:16 composition, cinematic framing, no text, no watermark, "
    "keep the lower third darker and uncluttered for subtitles."
)

DEFAULT_NEGATIVE = (
    "text, watermark, logo, distorted anatomy, morphing, warping limbs, "
    "flickering geometry, extra limbs"
)

TEXT_MODEL = os.environ.get("GEMINI_TEXT_MODEL", "gemini-2.5-flash")

DRAFT_INSTRUCTIONS = """\
You write image-generation scene prompts for a dark-fantasy lore Short.
For EACH numbered narration sentence below, write one scene.

Rules:
- Describe ONLY composition, subject, action and lighting. Do NOT describe art
  style, palette or rendering — the style is locked by reference images.
- Filter-safe vocabulary: physical/atmospheric description, never graphic.
  Avoid these words entirely: {risky}.
- Characters available (attach by listing their name in "characters"):
  {characters}. Only list a character for scenes where they visibly appear.
- "animate": true for the 2-4 scenes where motion adds the most (reveals,
  battles, transformations); false for static/establishing beats.
- "motion_prompt": only for animate=true scenes — pure physical camera/motion
  description (slow push-in, drifting embers...), stable forms, no morphing.

Return ONLY a JSON array, one object per sentence, in order:
[{{"prompt": "...", "motion_prompt": "..." or null, "animate": true/false,
   "characters": ["name", ...]}}]

Sentences:
{sentences}
"""


def _deterministic_drafts(sentences: list[str], characters: list[str]) -> list[dict]:
    """No-LLM fallback: the sentence itself becomes the scene prompt (the user
    edits it in-app), character refs wired in by name match, nothing animated
    by default (the hybrid choice stays with the user, and stills are cheap)."""
    drafts = []
    for s in sentences:
        low = s.lower()
        drafts.append({
            "prompt": s.strip(),
            "motion_prompt": None,
            "animate": False,
            "characters": [c for c in characters if c.lower() in low],
        })
    return drafts


def _llm_drafts(sentences: list[str], characters: list[str]) -> list[dict]:
    from google import genai
    from i2v.config import resolve_api_key

    client = genai.Client(api_key=resolve_api_key())
    prompt = DRAFT_INSTRUCTIONS.format(
        risky=", ".join(prompt_gen.RISKY_WORDS),
        characters=", ".join(characters) if characters else "(none)",
        sentences="\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1)),
    )
    resp = client.models.generate_content(model=TEXT_MODEL, contents=prompt)
    text = resp.text.strip()
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        raise ValueError(f"model returned no JSON array: {text[:200]}")
    drafts = json.loads(m.group(0))
    if len(drafts) != len(sentences):
        raise ValueError(f"model returned {len(drafts)} scenes for "
                         f"{len(sentences)} sentences")
    known = {c.lower(): c for c in characters}
    out = []
    for d in drafts:
        out.append({
            "prompt": str(d.get("prompt", "")).strip(),
            "motion_prompt": (str(d["motion_prompt"]).strip()
                              if d.get("motion_prompt") else None),
            "animate": bool(d.get("animate")),
            "characters": [known[c.lower()] for c in d.get("characters", [])
                           if c.lower() in known],
        })
    return out


def generate_manifest(ep: st.Episode, use_llm: bool = True,
                      duration_seconds: int = 8) -> tuple[dict, list[str]]:
    """Build the aria-i2v manifest for an episode. Returns (manifest, notes).

    Gated on references: raises if the style lock isn't set — prompts must be
    generated WITH the references in hand so refs wire into scenes and prompts
    stay compositional.
    """
    style_refs = ep.style_refs()
    if not style_refs:
        from orchestrator.errors import FriendlyError
        raise FriendlyError(
            "The style lock is not set. Pick at least one style reference image "
            "in the References tab first — prompt generation is intentionally "
            "blocked until the look is locked."
        )

    script = ep.script_text()
    sentences = prompt_gen.split_sentences(script)
    characters = ep.characters()
    char_names = [c for c in characters if characters[c]]

    notes: list[str] = []
    drafts: list[dict] | None = None
    if use_llm:
        try:
            drafts = _llm_drafts(sentences, char_names)
            notes.append(f"Scene prompts drafted by {TEXT_MODEL} — edit below.")
        except Exception as exc:  # fall back rather than block the wizard
            notes.append(f"LLM drafting unavailable ({friendly(exc)}); "
                         "using sentences as draft prompts — edit below.")
    if drafts is None:
        drafts = _deterministic_drafts(sentences, char_names)
        if not use_llm:
            notes.append("Drafted from sentences (no LLM) — edit below.")

    clips = []
    for i, (sent, d) in enumerate(zip(sentences, drafts), 1):
        clips.append({
            "image": f"{i:02d}.png",
            "name": f"{i:02d}_{st.slugify(d['prompt'] or sent, f'scene_{i:02d}')}",
            "narrates": sent,                       # context only; i2v ignores it
            "refs": d["characters"],
            "prompt": d["prompt"] or sent,
            "motion_prompt": d["motion_prompt"] or prompt_gen.DEFAULT_MOTION,
            "animate": d["animate"],
        })
        # Lint user-facing text only — the stock DEFAULT_MOTION contains the
        # word "bodies" in its own safety instruction and must not self-flag.
        motion_to_lint = d["motion_prompt"] or ""
        if motion_to_lint == prompt_gen.DEFAULT_MOTION:
            motion_to_lint = ""
        hits = prompt_gen.lint((d["prompt"] or "") + " " + motion_to_lint)
        if hits:
            notes.append(f"Scene {i:02d}: possible filter triggers -> {', '.join(hits)}")

    manifest = {
        "defaults": {
            "image_model": "gemini-3.1-flash-image",
            "image_aspect_ratio": "9:16",
            "aspect_ratio": "9:16",
            "duration_seconds": duration_seconds,
            "negative_prompt": DEFAULT_NEGATIVE,
            "style_suffix": COMPOSITION_SUFFIX,
        },
        "style_refs": style_refs,
        "characters": {k: v for k, v in characters.items() if v},
        "clips": clips,
    }
    return manifest, notes
