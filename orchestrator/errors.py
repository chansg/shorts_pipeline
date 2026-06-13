"""Friendly error handling.

Maps the failure modes actually hit in production (missing API keys, missing
reference images, Veo rejecting a config field, ffmpeg not on PATH) to clear,
actionable messages instead of tracebacks. Anything unrecognized is passed
through with its type so it's still debuggable.
"""
from __future__ import annotations

import shutil


class FriendlyError(Exception):
    """An error with a user-facing message. The GUI shows .args[0] verbatim."""


def ensure_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise FriendlyError(
            "ffmpeg/ffprobe not found on PATH. Install it with:\n"
            "    winget install Gyan.FFmpeg\n"
            "then restart the app (a new terminal is needed to pick up PATH)."
        )


def friendly(exc: Exception) -> FriendlyError:
    """Translate a raised exception into a FriendlyError with a clear message."""
    if isinstance(exc, FriendlyError):
        return exc
    msg = str(exc)
    low = msg.lower()

    if "elevenlabs_api_key" in low:
        return FriendlyError(
            "ELEVENLABS_API_KEY is not set. Add it to the .env file at the repo "
            "root (see .env.example), or paste it in the Settings tab, then retry."
        )
    if "gemini_api_key" in low or "no api key found" in low:
        return FriendlyError(
            "GEMINI_API_KEY is not set. Add it to the .env file at the repo root "
            "(get a key at https://aistudio.google.com/apikey), or paste it in "
            "the Settings tab, then retry."
        )
    if "api key not valid" in low or "api_key_invalid" in low:
        return FriendlyError(
            "Google rejected the GEMINI_API_KEY (invalid or expired). Check the "
            "key in .env / the Settings tab."
        )
    if "reference image not found" in low:
        return FriendlyError(
            f"{msg}\n\nPut the file in the refs/ folder (or fix the path in the "
            "References tab) and retry."
        )
    if isinstance(exc, FileNotFoundError) and ("ffmpeg" in low or "ffprobe" in low
                                               or "winerror 2" in low):
        return FriendlyError(
            "ffmpeg/ffprobe not found on PATH. Install it with:\n"
            "    winget install Gyan.FFmpeg\n"
            "then restart the app."
        )
    if "invalid_argument" in low or "unsupported" in low or "is not supported" in low:
        return FriendlyError(
            "Veo rejected the request — usually an unsupported config field or "
            "model/parameter mismatch for this Veo version.\n\nDetails: " + msg
        )
    if "quota" in low or "resource_exhausted" in low or "429" in low:
        return FriendlyError(
            "The API reported a quota/rate limit. Wait a minute and retry; "
            "check your plan's limits if it persists.\n\nDetails: " + msg
        )
    return FriendlyError(f"{type(exc).__name__}: {msg}")
