"""Long-video VRAM hardening (no GPU): free_vram no-ops without torch, and the
ASR OOM-retry falls back to a smaller batch once before giving up."""
import pytest

from gameplay import device as device_mod
from gameplay import transcribe as tx


def test_free_vram_is_noop_without_torch(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)   # import torch -> ImportError
    device_mod.free_vram()                             # must not raise


class _FakeModel:
    def __init__(self, fail_batches):
        self.fail_batches = set(fail_batches)
        self.calls = []

    def transcribe(self, audio, batch_size):
        self.calls.append(batch_size)
        if batch_size in self.fail_batches:
            raise RuntimeError("CUDA out of memory. Tried to allocate ...")
        return {"segments": [{"start": 0, "end": 1, "words": []}], "language": "en"}


def test_oom_retry_falls_back_to_smaller_batch(monkeypatch):
    monkeypatch.setattr(device_mod, "free_vram", lambda: None)
    logs = []
    model = _FakeModel(fail_batches={8})              # OOM at 8, ok at 4
    out = tx._transcribe_with_oom_retry(model, "audio", batch=8, oom_batch=4,
                                        progress=logs.append)
    assert out["language"] == "en"
    assert model.calls == [8, 4]                       # retried once, smaller
    assert any("OOM" in m and "batch=4" in m for m in logs)


def test_oom_retry_reraises_if_still_oom(monkeypatch):
    monkeypatch.setattr(device_mod, "free_vram", lambda: None)
    model = _FakeModel(fail_batches={8, 4})            # OOM at both
    with pytest.raises(RuntimeError, match="out of memory"):
        tx._transcribe_with_oom_retry(model, "audio", batch=8, oom_batch=4,
                                      progress=lambda m: None)
    assert model.calls == [8, 4]


def test_non_oom_error_not_retried(monkeypatch):
    monkeypatch.setattr(device_mod, "free_vram", lambda: None)

    class _Boom:
        def transcribe(self, audio, batch_size):
            raise ValueError("something else")

    with pytest.raises(ValueError, match="something else"):
        tx._transcribe_with_oom_retry(_Boom(), "audio", batch=8, oom_batch=4,
                                      progress=lambda m: None)
