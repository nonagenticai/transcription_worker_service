"""Backend selection + device detection."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from src.config import TranscriptionSettings
from src.services.backend_selector import _default_model_size, _detect_device, build_backend
from src.services.backends.base import BackendUnavailableError


def test_default_model_size() -> None:
    assert _default_model_size("cpu") == "base"
    assert _default_model_size("cuda") == "large-v3"


def test_detect_device_explicit() -> None:
    assert _detect_device("cpu") == "cpu"
    assert _detect_device("cuda") == "cuda"


def test_detect_device_auto_no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """When torch is unavailable, auto should fall back to cpu."""
    monkeypatch.setitem(sys.modules, "torch", None)
    assert _detect_device("auto") == "cpu"


def test_detect_device_auto_with_fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake torch.cuda.is_available() → True should resolve to cuda."""
    fake_torch = types.ModuleType("torch")
    cuda_mod = types.SimpleNamespace(is_available=lambda: True)
    setattr(fake_torch, "cuda", cuda_mod)  # noqa: B010 -- dynamic patching
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert _detect_device("auto") == "cuda"


def test_build_backend_faster() -> None:
    """build_backend instantiates FasterWhisperBackend without loading model."""
    settings = TranscriptionSettings(backend="faster", device="cpu", model_size="tiny")
    backend = build_backend(settings)
    assert backend.name == "faster"
    assert backend.model_size == "tiny"
    assert backend.device == "cpu"
    assert not backend.is_loaded  # lazy load; model not loaded yet


def test_build_backend_openai() -> None:
    settings = TranscriptionSettings(backend="openai", device="cpu", model_size="tiny")
    backend = build_backend(settings)
    assert backend.name == "openai"
    assert backend.model_size == "tiny"
    assert backend.device == "cpu"
    assert not backend.is_loaded


def test_build_backend_auto_model_size_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """When model_size is None, CPU backend gets 'base'."""
    monkeypatch.setitem(sys.modules, "torch", None)  # type: ignore[arg-type]
    settings = TranscriptionSettings(backend="faster", device="auto")
    backend = build_backend(settings)
    assert backend.model_size == "base"
    assert backend.device == "cpu"


def test_unknown_backend_raises() -> None:
    settings = TranscriptionSettings(backend="faster")
    # Force the Literal check out of the way by direct assignment.
    object.__setattr__(settings, "backend", "mojo")
    with pytest.raises(BackendUnavailableError, match="Unknown backend"):
        build_backend(settings)


def test_detect_device_cuda_probe_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """If torch.cuda.is_available() raises, fall back to cpu."""
    fake_torch = types.ModuleType("torch")

    def _raise() -> Any:
        raise RuntimeError("no cuda")

    setattr(fake_torch, "cuda", types.SimpleNamespace(is_available=_raise))  # noqa: B010
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert _detect_device("auto") == "cpu"
