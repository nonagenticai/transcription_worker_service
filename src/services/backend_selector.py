"""Backend selection helper for transcription_worker_service."""

from __future__ import annotations

import logging

from src.config import TranscriptionSettings
from src.services.backends.base import BackendUnavailableError, WhisperBackend

logger = logging.getLogger(__name__)

_DEFAULT_CPU_MODEL = "base"
_DEFAULT_GPU_MODEL = "large-v3"


def _detect_device(requested: str) -> str:
    """Resolve ``auto`` to ``cuda`` when torch reports an available GPU."""
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except (RuntimeError, AssertionError) as exc:
        from src.services.metrics import cuda_fallback_total

        cuda_fallback_total.inc()
        logger.warning("CUDA probe failed, falling back to CPU: %s", exc)
    return "cpu"


def _default_model_size(device: str) -> str:
    return _DEFAULT_GPU_MODEL if device == "cuda" else _DEFAULT_CPU_MODEL


def build_backend(settings: TranscriptionSettings) -> WhisperBackend:
    """Instantiate the configured WhisperBackend without loading the model."""
    device = _detect_device(settings.device)
    model_size = settings.model_size or _default_model_size(device)

    if settings.backend == "faster":
        from src.services.backends.faster_whisper_backend import FasterWhisperBackend

        return FasterWhisperBackend(
            model_size=model_size, device=device, compute_type=settings.compute_type
        )
    if settings.backend == "openai":
        from src.services.backends.openai_whisper_backend import OpenAIWhisperBackend

        return OpenAIWhisperBackend(
            model_size=model_size, device=device, compute_type=settings.compute_type
        )
    raise BackendUnavailableError(f"Unknown backend: {settings.backend}")
