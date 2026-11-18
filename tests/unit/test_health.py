"""Health endpoint tests."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.health import router as health_router


def test_health_reports_backend(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["backend"] == "fake"
    assert body["model"] == "tiny"
    assert body["device"] == "cpu"
    assert body["loaded"] is False  # lazy; hasn't been invoked yet


def test_health_live(client: TestClient) -> None:
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_ready(client: TestClient) -> None:
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    assert body["backend"] == "fake"


def test_ready_is_false_when_backend_load_failed(client: TestClient, app: FastAPI) -> None:
    """Fail-closed readiness: a backend that cannot load must stop advertising."""
    app.state.backend_load_error = "faster-whisper is not installed"
    resp = client.get("/health/ready")
    assert resp.status_code == 503
    body = resp.json()
    assert body["ready"] is False
    assert "faster-whisper is not installed" in body["reason"]

    # readyz alias must agree — probes may point at either path.
    alias = client.get("/readyz")
    assert alias.status_code == 503
    assert alias.json()["ready"] is False


def test_liveness_stays_green_when_backend_load_failed(client: TestClient, app: FastAPI) -> None:
    """Liveness must NOT flip — a load failure should shed traffic, not
    put the process into a supervisor restart loop."""
    app.state.backend_load_error = "boom"
    assert client.get("/health/live").status_code == 200
    assert client.get("/healthz").status_code == 200
    assert client.get("/health").status_code == 200


def test_ready_is_false_without_backend() -> None:
    app = FastAPI()
    app.include_router(health_router)
    client = TestClient(app)
    resp = client.get("/health/ready")
    assert resp.status_code == 503
    assert resp.json()["ready"] is False


def test_ready_is_false_when_eager_preload_never_loaded(client: TestClient, app: FastAPI) -> None:
    """With load_on_startup=true an unloaded model is a genuine failure."""
    app.state.settings.transcription.load_on_startup = True
    resp = client.get("/health/ready")
    assert resp.status_code == 503
    assert resp.json()["ready"] is False


def test_health_without_backend() -> None:
    app = FastAPI()
    app.include_router(health_router)
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["backend"] == "unknown"
    assert body["loaded"] is False
