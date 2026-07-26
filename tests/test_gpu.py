"""
Tests for core/gpu.py -- device_count()/resolve_devices()/
ensure_cuda_visible_devices() are all pure or near-pure (device_count()'s
only side effect is a subprocess call, fully mockable), so none of this
needs a real GPU, torch, or nvidia-smi to test.
"""

import subprocess
from unittest.mock import MagicMock

import pytest

from core.gpu import (
    DeviceAssignment,
    device_count,
    ensure_cuda_visible_devices,
    resolve_devices,
)


# ── device_count() ───────────────────────────────────────────────────────────


def test_device_count_parses_two_gpu_nvidia_smi_output(monkeypatch):
    fake_stdout = (
        "GPU 0: NVIDIA A100 80GB PCIe (UUID: GPU-aaaa)\n"
        "GPU 1: NVIDIA A100 80GB PCIe (UUID: GPU-bbbb)\n"
    )
    monkeypatch.setattr(
        "core.gpu.subprocess.run",
        lambda *a, **k: MagicMock(returncode=0, stdout=fake_stdout),
    )
    assert device_count() == 2


def test_device_count_parses_single_gpu_nvidia_smi_output(monkeypatch):
    fake_stdout = "GPU 0: NVIDIA A100 80GB PCIe (UUID: GPU-aaaa)\n"
    monkeypatch.setattr(
        "core.gpu.subprocess.run",
        lambda *a, **k: MagicMock(returncode=0, stdout=fake_stdout),
    )
    assert device_count() == 1


def test_device_count_zero_when_nvidia_smi_missing(monkeypatch):
    def _raise(*a, **k):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr("core.gpu.subprocess.run", _raise)
    assert device_count() == 0


def test_device_count_zero_when_nvidia_smi_times_out(monkeypatch):
    def _raise(*a, **k):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10)

    monkeypatch.setattr("core.gpu.subprocess.run", _raise)
    assert device_count() == 0


def test_device_count_zero_on_nonzero_returncode(monkeypatch):
    monkeypatch.setattr(
        "core.gpu.subprocess.run",
        lambda *a, **k: MagicMock(returncode=1, stdout=""),
    )
    assert device_count() == 0


# ── resolve_devices() ────────────────────────────────────────────────────────


def test_resolve_devices_single_gpu_is_contended():
    a = resolve_devices(count=1)
    assert a.frontend_cuda_visible_devices == "0"
    assert a.conditioner_cuda_visible_devices == "0"
    assert a.contended is True


def test_resolve_devices_zero_gpus_is_contended():
    a = resolve_devices(count=0)
    assert a.frontend_cuda_visible_devices == "0"
    assert a.conditioner_cuda_visible_devices == "0"
    assert a.contended is True


def test_resolve_devices_two_gpus_splits_and_is_not_contended():
    a = resolve_devices(count=2)
    assert a.frontend_cuda_visible_devices == "0"
    assert a.conditioner_cuda_visible_devices == "1"
    assert a.contended is False


def test_resolve_devices_more_than_two_gpus_still_uses_first_two():
    a = resolve_devices(count=4)
    assert a.frontend_cuda_visible_devices == "0"
    assert a.conditioner_cuda_visible_devices == "1"
    assert a.contended is False


def test_resolve_devices_defaults_to_real_device_count(monkeypatch):
    monkeypatch.setattr("core.gpu.device_count", lambda: 2)
    a = resolve_devices()
    assert a.contended is False


# ── ensure_cuda_visible_devices() ────────────────────────────────────────────


def test_ensure_cuda_visible_devices_sets_frontend_when_unset(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assignment = ensure_cuda_visible_devices("frontend", count=2)
    assert assignment.frontend_cuda_visible_devices == "0"
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_ensure_cuda_visible_devices_sets_conditioner_when_unset(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assignment = ensure_cuda_visible_devices("conditioner", count=2)
    assert assignment.conditioner_cuda_visible_devices == "1"
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_ensure_cuda_visible_devices_respects_existing_override(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    assignment = ensure_cuda_visible_devices("conditioner", count=2)
    import os

    # Override is untouched, even though auto-detection would have picked "1".
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"
    # But contended still reflects what auto-detection concluded, not the
    # override -- see ensure_cuda_visible_devices()'s own docstring for why.
    assert assignment.contended is False


def test_ensure_cuda_visible_devices_respects_empty_string_override(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    ensure_cuda_visible_devices("frontend", count=1)
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_ensure_cuda_visible_devices_invalid_role_raises(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError):
        ensure_cuda_visible_devices("gpu_zero", count=2)
