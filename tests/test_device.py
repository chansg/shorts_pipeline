"""Device/model selection — the CPU-vs-CUDA decision that the first real run got
wrong (it ran large-v2 on a CPU-only torch). No GPU needed: we mock detection."""
import sys
import types

from gameplay import config as gconf
from gameplay import device as device_mod


def test_plan_cuda(monkeypatch):
    monkeypatch.setattr(device_mod, "_cuda_available", lambda: True)
    plan = device_mod.plan_device()
    assert plan.device == "cuda"
    assert plan.model == gconf.WHISPERX_MODEL_CUDA
    assert plan.compute_type == gconf.WHISPERX_COMPUTE_CUDA
    assert plan.on_cpu_fallback is False
    assert plan.warning is None


def test_plan_cpu_fallback_warns(monkeypatch):
    monkeypatch.setattr(device_mod, "_cuda_available", lambda: False)
    plan = device_mod.plan_device()
    assert plan.device == "cpu"
    assert plan.model == gconf.WHISPERX_MODEL_CPU      # smaller model on CPU
    assert plan.compute_type == gconf.WHISPERX_COMPUTE_CPU
    assert plan.on_cpu_fallback is True
    assert plan.warning and "CPU" in plan.warning
    assert "cu121" in plan.warning or "index-url" in plan.warning   # names the fix


def _stub_torch(cuda_available: bool):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    return torch


def test_cuda_available_true_via_torch_stub(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _stub_torch(True))
    assert device_mod._cuda_available() is True


def test_cuda_available_false_via_torch_stub(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _stub_torch(False))
    assert device_mod._cuda_available() is False


def test_cuda_available_no_torch_is_false(monkeypatch):
    # A missing/broken torch must degrade to CPU, not raise.
    monkeypatch.setitem(sys.modules, "torch", None)   # import torch -> ImportError
    assert device_mod._cuda_available() is False
