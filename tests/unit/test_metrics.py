"""Tests for Prometheus metrics exporter and CUDA fallback counter."""

from __future__ import annotations

from unittest.mock import patch

import pytest


def _read_counter_value(name: str) -> float:
    """Look up a counter's value from the default registry."""
    from prometheus_client import REGISTRY

    value = REGISTRY.get_sample_value(name + "_total")
    return float(value) if value is not None else 0.0


def test_render_metrics_returns_text_format():
    """render_metrics must return Prometheus exposition bytes + content-type."""
    from src.services.metrics import render_metrics

    body, content_type = render_metrics()
    assert isinstance(body, bytes)
    assert "text/plain" in content_type
    assert b"transcription_cuda_fallback_total" in body


def test_cuda_fallback_counter_increments_on_probe_failure():
    """When torch.cuda.is_available raises, the counter must increment."""
    from src.services import backend_selector

    before = _read_counter_value("transcription_cuda_fallback")

    class _BoomTorch:
        class cuda:  # noqa: N801 - mimic torch namespace
            @staticmethod
            def is_available() -> bool:
                raise RuntimeError("driver not loaded")

    with patch.dict("sys.modules", {"torch": _BoomTorch}):
        result = backend_selector._detect_device("auto")

    after = _read_counter_value("transcription_cuda_fallback")
    assert result == "cpu"
    assert after == pytest.approx(before + 1)


def test_cuda_fallback_counter_unchanged_on_cpu_only():
    """If torch import fails (no GPU stack), counter must NOT increment."""
    from src.services import backend_selector

    before = _read_counter_value("transcription_cuda_fallback")
    # Simulate ImportError by removing torch from sys.modules and blocking import.
    import builtins

    real_import = builtins.__import__

    def _block_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=_block_torch):
        result = backend_selector._detect_device("auto")

    after = _read_counter_value("transcription_cuda_fallback")
    assert result == "cpu"
    assert after == pytest.approx(before)


@pytest.mark.asyncio
async def test_metrics_endpoint_exposes_counter():
    """GET /metrics returns 200 and includes our custom metric name."""
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from src.services.admission import build_gate

    # Build a tiny app mounting only the metrics handler to avoid lifespan side effects.
    app = FastAPI()
    build_gate()  # registers the transcribe lane's gpu_admission_* series

    from src.services.metrics import render_metrics

    @app.get("/metrics")
    async def _m() -> object:
        from fastapi import Response

        body, ct = render_metrics()
        return Response(content=body, media_type=ct)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/metrics")

    assert resp.status_code == 200
    assert "transcription_cuda_fallback_total" in resp.text
    assert 'gpu_admission_active{lane="transcribe"}' in resp.text
    assert 'gpu_admission_rejected_total{lane="transcribe",reason="full"}' in resp.text
    # Forward-only: the hand-rolled shed counter is gone (gpu_admission_rejected_total).
    assert "transcription_shed_total" not in resp.text
