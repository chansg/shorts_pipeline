"""Per-episode state and stage gating.

One episode = one script (scripts/<name>.txt). Everything the wizard produces
lives in predictable places so stage completion is DERIVED FROM DISK — re-runs
resume where they left off and never redo (or re-bill) finished work:

    manifests/<name>.json         aria-i2v prompt manifest (the creative spec)
    manifests/<name>.state.json   wizard choices: refs, voice, approvals, timeline
    episodes/<name>/stills/       NN.png from Nano Banana 2
    episodes/<name>/clips/        <scene_name>.mp4 from Veo
    assets/images/                NN.png / NN.mp4 after handoff (pipeline input)
    output/<name>.mp4             final render
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS_DIR = ROOT / "manifests"
EPISODES_DIR = ROOT / "episodes"
REFS_DIR = ROOT / "refs"
for _d in (MANIFESTS_DIR, EPISODES_DIR, REFS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

STAGES = ["script", "references", "prompts", "stills", "animate",
          "handoff", "assemble", "qc"]


def script_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def slugify(text: str, fallback: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:24] or fallback


class Episode:
    def __init__(self, name: str):
        self.name = name
        self.script_path = config.SCRIPTS_DIR / f"{name}.txt"
        self.manifest_path = MANIFESTS_DIR / f"{name}.json"
        self.state_path = MANIFESTS_DIR / f"{name}.state.json"
        self.stills_dir = EPISODES_DIR / name / "stills"
        self.clips_dir = EPISODES_DIR / name / "clips"
        self.output_path = config.OUTPUT_DIR / f"{name}.mp4"

    # ---- state json (wizard choices that aren't derivable from disk) ----

    def load_state(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        return {}

    def save_state(self, **updates) -> dict:
        state = self.load_state()
        state.update(updates)
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return state

    # ---- manifest ----

    def load_manifest(self) -> dict | None:
        if not self.manifest_path.exists():
            return None
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def save_manifest(self, data: dict) -> None:
        self.manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                                      encoding="utf-8")

    def clips(self) -> list[dict]:
        m = self.load_manifest()
        return list(m.get("clips", [])) if m else []

    def animated_clips(self) -> list[dict]:
        return [c for c in self.clips() if c.get("animate")]

    # ---- per-stage completion, derived from disk ----

    def script_text(self) -> str:
        return (self.script_path.read_text(encoding="utf-8").strip()
                if self.script_path.exists() else "")

    def style_refs(self) -> list[str]:
        state = self.load_state()
        return [r for r in state.get("style_refs", []) if Path(r).exists()]

    def characters(self) -> dict[str, list[str]]:
        state = self.load_state()
        return {k: [p for p in v if Path(p).exists()]
                for k, v in state.get("characters", {}).items()}

    def still_path(self, clip: dict) -> Path:
        return self.stills_dir / clip["image"]

    def clip_path(self, clip: dict) -> Path:
        return self.clips_dir / f"{clip['name']}.mp4"

    def stills_done(self) -> bool:
        clips = self.clips()
        return bool(clips) and all(self.still_path(c).exists() for c in clips)

    def animate_done(self) -> bool:
        """Animation is complete once the user confirmed the still/clip split AND
        every scene marked animate has a rendered Veo clip on disk."""
        if not self.load_state().get("animate_confirmed"):
            return False
        return all(self.clip_path(c).exists() for c in self.animated_clips())

    def expected_assets(self) -> list[tuple[Path, Path]]:
        """(source, destination) pairs for the handoff, in manifest order.
        Animated scenes hand off the Veo clip as NN.mp4, the rest NN.png."""
        pairs: list[tuple[Path, Path]] = []
        for idx, c in enumerate(self.clips(), 1):
            if c.get("animate") and self.clip_path(c).exists():
                pairs.append((self.clip_path(c), config.IMAGES_DIR / f"{idx:02d}.mp4"))
            else:
                pairs.append((self.still_path(c), config.IMAGES_DIR / f"{idx:02d}.png"))
        return pairs

    def handoff_done(self) -> bool:
        pairs = self.expected_assets()
        if not pairs:
            return False
        if not all(dest.exists() for _, dest in pairs):
            return False
        # the handoff must be THIS episode's: no stray extra media in the dir
        from modules.visuals import VALID_EXT
        present = {p.name for p in config.IMAGES_DIR.iterdir()
                   if p.suffix.lower() in VALID_EXT}
        return present == {dest.name for _, dest in pairs}

    def assemble_done(self) -> bool:
        """Rendered AND still current: same script and the same media files in
        assets/images (so swapping a still for a Veo clip forces a re-render)."""
        if not self.output_path.exists():
            return False
        state = self.load_state()
        if state.get("rendered_script_hash") != script_hash(self.script_text()):
            return False
        recorded = state.get("rendered_assets")
        return recorded is None or recorded == assets_fingerprint()

    def status(self) -> dict[str, bool]:
        return {
            "script": bool(self.script_text()),
            "references": bool(self.style_refs()),
            "prompts": self.load_manifest() is not None,
            "stills": self.stills_done(),
            "animate": self.animate_done(),
            "handoff": self.handoff_done(),
            "assemble": self.assemble_done(),
            "qc": bool(self.load_state().get("approved")),
        }

    def first_incomplete(self) -> str:
        st = self.status()
        for stage in STAGES:
            if not st[stage]:
                return stage
        return "done"


def assets_fingerprint() -> list:
    """Identity of the current pipeline input media (name + mtime), JSON-safe.
    Recorded at render time so a later handoff invalidates the old render."""
    from modules.visuals import VALID_EXT
    return [[p.name, int(p.stat().st_mtime)]
            for p in sorted(config.IMAGES_DIR.iterdir())
            if p.suffix.lower() in VALID_EXT]


def list_scripts() -> list[str]:
    return sorted(p.stem for p in config.SCRIPTS_DIR.glob("*.txt"))


def list_ref_images() -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(p for p in REFS_DIR.iterdir() if p.suffix.lower() in exts)
