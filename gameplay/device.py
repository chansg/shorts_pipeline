"""Single source of truth for compute device selection (CUDA vs CPU) and the
WhisperX model size that goes with it.

Shared by the manual transcribe path and the full-auto path so device logic lives
in exactly one place. torch is imported lazily (and defensively) so importing this
module never requires torch — the rest of the gameplay package stays GPU-free.

The first real run shipped on CPU because the installed torch was the CPU-only
build (`torch.version.cuda is None`). We can't fix that for the user, but we detect
it correctly, pick a CPU-appropriate model so the mode stays usable, and surface a
loud, actionable warning telling them how to enable the GPU.
"""
from __future__ import annotations

from dataclasses import dataclass

from gameplay import config as gconf

_CPU_WARNING = (
    "⚠ Running WhisperX on CPU — no CUDA GPU detected by torch. This is slow; a "
    "1-hour video is impractical (full-auto especially). Almost always the cause is "
    "a CPU-only torch build. Install the CUDA-matched torch (see the header of "
    "requirements-gameplay.txt), e.g.:\n"
    "    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121\n"
    f"Using the smaller '{gconf.WHISPERX_MODEL_CPU}' model on CPU so it stays usable."
)


@dataclass
class DevicePlan:
    device: str            # "cuda" | "cpu"
    compute_type: str      # faster-whisper compute type for this device
    model: str             # WhisperX model size for this device
    on_cpu_fallback: bool  # True when we wanted CUDA but fell back to CPU
    warning: str | None    # user-facing warning (CPU fallback), else None

    def describe(self) -> str:
        return f"{self.model} on {self.device}"


def _cuda_available() -> bool:
    """Whether torch reports a usable CUDA GPU. Isolated so tests can monkeypatch it,
    and so a missing/broken torch degrades to CPU rather than raising."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:       # noqa: BLE001 — no torch / broken install -> CPU
        return False


def free_vram() -> None:
    """Release cached CUDA memory + run a GC pass. Called between the transcription
    and diarization models so a 10GB card never holds both at once. No-op (never
    raises) when torch is missing or CPU-only."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:       # noqa: BLE001 — best-effort; never fatal
        pass


def plan_device() -> DevicePlan:
    """Pick the device + matching model size. CUDA when available, else CPU with a
    smaller model and a prominent warning explaining the fix."""
    if _cuda_available():
        return DevicePlan(
            device="cuda",
            compute_type=gconf.WHISPERX_COMPUTE_CUDA,
            model=gconf.WHISPERX_MODEL_CUDA,
            on_cpu_fallback=False,
            warning=None,
        )
    return DevicePlan(
        device="cpu",
        compute_type=gconf.WHISPERX_COMPUTE_CPU,
        model=gconf.WHISPERX_MODEL_CPU,
        on_cpu_fallback=True,
        warning=_CPU_WARNING,
    )
