"""QC gate helpers: spec check, per-scene contact sheet, bad-frame detection,
and title/description drafting for the publish step.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import config
from orchestrator import state as st
from orchestrator.errors import FriendlyError, ensure_ffmpeg, friendly

NEAR_BLACK_MEAN = 14  # 0-255 mean luminance below this = suspicious frame


def spec_check(video: Path) -> dict:
    """Probe the final mp4 against the publish spec (1080x1920 / 30fps / audio)."""
    ensure_ffmpeg()
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(video)],
        capture_output=True, text=True,
    ).stdout
    data = json.loads(out or "{}")
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    a = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    num, _, den = (v.get("avg_frame_rate") or "0/1").partition("/")
    try:
        fps = round(float(num) / float(den or 1), 2)
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    duration = float(data.get("format", {}).get("duration", 0.0) or 0.0)
    checks = {
        "resolution": (v.get("width"), v.get("height")) == (config.WIDTH, config.HEIGHT),
        "fps": abs(fps - config.FPS) < 0.6,
        "audio": a is not None,
        "duration": 10.0 <= duration <= 180.0,
    }
    return {
        "width": v.get("width"), "height": v.get("height"), "fps": fps,
        "duration": round(duration, 1), "has_audio": a is not None,
        "checks": checks, "ok": all(checks.values()),
    }


def contact_sheet(ep: st.Episode) -> tuple[list[tuple[str, str]], list[str]]:
    """First frame of each scene from the FINAL video (what the viewer sees).
    Returns (gallery_items=(path,label), issues). Near-black frames are flagged
    so a failed generation or wrong shot is visible at a glance."""
    ensure_ffmpeg()
    timeline = ep.load_state().get("scene_timeline") or []
    if not timeline:
        raise FriendlyError("No scene timeline recorded — run Voice & Assemble first.")
    if not ep.output_path.exists():
        raise FriendlyError(f"Final video not found: {ep.output_path}")

    sheet_dir = config.WORK_DIR / f"{ep.name}_contact"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    items: list[tuple[str, str]] = []
    issues: list[str] = []
    try:
        from PIL import Image
        for i, scene in enumerate(timeline, 1):
            t = scene["start"] + min(0.3, scene["duration"] / 2)
            frame = sheet_dir / f"scene_{i:02d}.jpg"
            subprocess.run(
                ["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", str(ep.output_path),
                 "-frames:v", "1", "-q:v", "3", str(frame)],
                capture_output=True, text=True, check=True,
            )
            label = f"{i:02d} · {scene['file']} · {scene['start']:.1f}s"
            mean = _mean_luma(Image.open(frame))
            if mean < NEAR_BLACK_MEAN:
                label += "  ⚠ near-black"
                issues.append(f"Scene {i:02d} ({scene['file']}) looks near-black "
                              f"(mean luma {mean:.0f}) — check for a failed shot.")
            items.append((str(frame), label))
    except subprocess.CalledProcessError as exc:
        raise friendly(RuntimeError(exc.stderr[-500:] if exc.stderr else str(exc)))
    return items, issues


def _mean_luma(img) -> float:
    g = img.convert("L")
    g.thumbnail((160, 160))
    hist = g.histogram()
    total = sum(hist)
    return sum(i * c for i, c in enumerate(hist)) / max(1, total)


def open_output_folder(path: Path) -> None:
    folder = path.parent if path.is_file() else path
    if sys.platform == "win32":
        os.startfile(str(folder))  # noqa: S606
    else:
        subprocess.Popen(["xdg-open", str(folder)])


def draft_metadata(script_text: str) -> tuple[str, str]:
    """Draft a title + description from the script via Gemini; deterministic
    fallback (hook + template) if no key / API failure. Cheap text call."""
    hook = next((ln.strip() for ln in script_text.splitlines() if ln.strip()), "")
    fallback_title = (hook.rstrip(".!?")[:90]) or "Untitled Short"
    fallback_desc = (f"{hook}\n\nDark Souls lore, retold.\n"
                     "#darksouls #lore #shorts #gaming")
    try:
        from google import genai
        from i2v.config import resolve_api_key
        from orchestrator.manifest_gen import TEXT_MODEL
        client = genai.Client(api_key=resolve_api_key())
        resp = client.models.generate_content(
            model=TEXT_MODEL,
            contents=("Write YouTube Shorts metadata for this dark-fantasy lore "
                      "narration. Return EXACTLY two lines:\n"
                      "line 1: a click-curiosity title under 90 chars (no quotes)\n"
                      "line 2: a 1-2 sentence description ending with 4-6 hashtags\n\n"
                      + script_text),
        )
        lines = [ln.strip() for ln in resp.text.strip().splitlines() if ln.strip()]
        if len(lines) >= 2:
            return lines[0][:100], " ".join(lines[1:])
    except Exception:
        pass
    return fallback_title, fallback_desc
