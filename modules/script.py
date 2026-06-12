"""
Step 1 — Script.

Takes a raw story seed (Reddit post, Dark Souls lore notes, anything) and
produces an ORIGINAL, rewritten script. The rewrite is what keeps you clear
of copyright: change names, phrasing, structure; keep only the idea.

The first line of the returned script is treated as the HOOK and should earn
the first 2 seconds. Retention on shorts lives or dies here.
"""
from __future__ import annotations
import subprocess
from pathlib import Path
import config

REWRITE_SYSTEM = """You are a short-form video scriptwriter.
Rewrite the user's story seed into an ORIGINAL narration script for a 30-60s
vertical video. Rules:
- Do NOT copy phrasing from the seed. Reword everything. Change names.
- Open with a 1-sentence hook that creates an open loop (curiosity/tension).
- Short, punchy sentences. Spoken-word rhythm, not written prose.
- 110-160 words total (≈ 35-55 seconds narrated).
- No emojis, no stage directions, no "in this video". Just the narration.
Return ONLY the script text."""


def load_seed(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def rewrite(seed: str) -> str:
    """Rewrite a seed into an original script via the configured backend."""
    backend = config.REWRITE_BACKEND
    if backend == "none":
        return seed
    if backend == "ollama":
        return _rewrite_ollama(seed)
    if backend == "claude":
        return _rewrite_claude(seed)
    raise ValueError(f"Unknown REWRITE_BACKEND: {backend}")


def _rewrite_ollama(seed: str) -> str:
    """Local Mistral via Ollama — same engine as Aria's intent router tier 2."""
    prompt = f"{REWRITE_SYSTEM}\n\n--- STORY SEED ---\n{seed}\n\n--- SCRIPT ---\n"
    result = subprocess.run(
        ["ollama", "run", config.OLLAMA_MODEL, prompt],
        capture_output=True, text=True, timeout=120,
        encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Ollama failed: {result.stderr}")
    return result.stdout.strip()


def _rewrite_claude(seed: str) -> str:
    """Anthropic API — reuse Aria's brain client if you prefer higher quality."""
    import anthropic  # pip install anthropic ; key via ANTHROPIC_API_KEY
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=400,
        system=REWRITE_SYSTEM,
        messages=[{"role": "user", "content": seed}],
    )
    return msg.content[0].text.strip()


def hook_of(script: str) -> str:
    """First sentence = the hook. Used for the thumbnail / title later."""
    for line in script.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


if __name__ == "__main__":
    import sys
    s = load_seed(sys.argv[1])
    out = rewrite(s)
    print("HOOK:", hook_of(out), "\n")
    print(out)
