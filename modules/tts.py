"""
Step 2 — Voiceover (TTS) via ElevenLabs.

ElevenLabs ONLY. There is deliberately no local/offline fallback: for polished
videos we want one consistent, high-quality voice. If the API key is missing or
the call fails, the pipeline raises and stops rather than silently degrading.

Auth: set ELEVENLABS_API_KEY in your .env (loaded automatically by config.py).
Never hard-code the key in this file.
"""
from __future__ import annotations
import os
from pathlib import Path
import wave
import numpy as np
import config


def synthesize(text: str, out_path: str | Path) -> Path:
    """Generate a voiceover WAV from text using ElevenLabs. Raises on any failure."""
    out_path = Path(out_path)

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ELEVENLABS_API_KEY not set. Put it in your .env (loaded by config.py). "
            "This pipeline is ElevenLabs-only and will not fall back to another voice."
        )

    from elevenlabs.client import ElevenLabs
    client = ElevenLabs(api_key=api_key)

    # pcm_24000 = raw 16-bit PCM @ 24 kHz, wrapped into WAV for Whisper + ffmpeg.
    audio = client.text_to_speech.convert(
        voice_id=config.ELEVENLABS_VOICE_ID,
        model_id=config.ELEVENLABS_MODEL,
        text=text,
        output_format="pcm_24000",
        voice_settings={
            "stability": config.ELEVENLABS_STABILITY,
            "similarity_boost": config.ELEVENLABS_SIMILARITY,
            "style": config.ELEVENLABS_STYLE,
            "use_speaker_boost": True,
        },
    )

    pcm = b"".join(audio)  # convert() yields byte chunks
    if not pcm:
        raise RuntimeError(
            "ElevenLabs returned empty audio. Check your ELEVENLABS_VOICE_ID and "
            "that your account has remaining character quota."
        )
    samples = np.frombuffer(pcm, dtype=np.int16)
    _write_wav(out_path, samples, sample_rate=24000)
    return out_path


def _write_wav(path: Path, samples, sample_rate: int) -> None:
    samples = np.asarray(samples)
    if samples.dtype != np.int16:
        samples = np.clip(samples, -1.0, 1.0)
        samples = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


def list_elevenlabs_voices() -> None:
    """List available voices + IDs. Self-diagnoses common setup problems."""
    # 1. Is python-dotenv installed? Without it, the .env file is never read.
    try:
        import dotenv  # noqa: F401
    except ImportError:
        print("[!] python-dotenv is NOT installed, so your .env file isn't being read.")
        print("    Fix:  pip install python-dotenv\n")

    # 2. Did the key actually load?
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        print("[!] ELEVENLABS_API_KEY not found in environment or .env.")
        print("    - Make sure a file named exactly '.env' sits in the project root")
        print("      (same folder as config.py), containing:")
        print("        ELEVENLABS_API_KEY=sk_your_real_key")
        return

    print(f"[ok] Key loaded (starts with '{key[:6]}...'). Contacting ElevenLabs...\n")

    # 3. Try the API and report clearly on success OR failure.
    try:
        from elevenlabs.client import ElevenLabs
        client = ElevenLabs(api_key=key)
        voices = client.voices.get_all().voices
    except Exception as e:
        print("[!] Couldn't fetch voices from ElevenLabs.")
        print(f"    {type(e).__name__}: {e}")
        print("    If this is an auth error, the API key is likely wrong or expired.")
        return

    if not voices:
        print("[!] Connected fine, but the account returned no voices.")
        return

    print(f"{'NAME':24s}  VOICE_ID")
    print(f"{'-'*24}  {'-'*22}")
    for v in voices:
        print(f"{v.name:24s}  {v.voice_id}")
    print(f"\n{len(voices)} voices found. "
          f"Copy a VOICE_ID into ELEVENLABS_VOICE_ID in config.py.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "voices":
        list_elevenlabs_voices()
