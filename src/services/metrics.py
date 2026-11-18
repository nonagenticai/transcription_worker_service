"""Prometheus metrics for transcription_worker_service.

Centralizes counters/histograms so other modules import a single registry-bound
metric rather than constructing ad-hoc Counter() instances (which would error
on re-import / test collection).
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY, Counter, generate_latest

cuda_fallback_total: Counter = Counter(
    "transcription_cuda_fallback_total",
    "Number of times the CUDA device probe failed and the worker fell back to CPU.",
)


def render_metrics() -> tuple[bytes, str]:
    """Render the default registry to Prometheus text exposition format.

    Returns the body bytes and the Content-Type header value.
    """
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


__all__ = ["cuda_fallback_total", "render_metrics"]
