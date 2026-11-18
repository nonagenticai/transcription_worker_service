"""Additional error-path tests for transcription routes."""

from __future__ import annotations

import io
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.health import router as health_router
from src.api.routes import router as transcription_router
from src.config import get_settings
from src.services.backends.base import (
    BackendUnavailableError,
    TranscriptionResult,
    WhisperBackend,
)
from tests.conftest import stamp_test_tenant


class _RaisingBackend(WhisperBackend):
    name = "raising"

    def __init__(self, error: Exception) -> None:
        super().__init__(model_size="tiny", device="cpu", compute_type="int8")
        self.error = error

    @property
    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        pass

    def transcribe(
        self,
        audio_path: Path,
        language: str | None,
        word_timestamps: bool,
    ) -> TranscriptionResult:
        raise self.error


def _make_app(backend: WhisperBackend) -> FastAPI:
    app = FastAPI()
    app.state.backend = backend
    app.state.settings = get_settings()
    app.include_router(transcription_router)
    stamp_test_tenant(app)
    app.include_router(health_router)
    return app


def _make_client(backend: WhisperBackend) -> TestClient:
    return TestClient(_make_app(backend))


def test_transcribe_backend_unavailable_returns_503() -> None:
    client = _make_client(_RaisingBackend(BackendUnavailableError("not installed")))
    payload = {"file": ("a.wav", io.BytesIO(b"data"), "audio/wav")}
    resp = client.post("/api/transcribe", files=payload)
    assert resp.status_code == 503
    assert "not installed" in resp.json()["detail"]


def test_backend_unavailable_flips_readiness_to_false() -> None:
    """A lazy load that fails at request time must be visible to the probe —
    otherwise the service 503s every transcription while advertising ready."""
    app = _make_app(_RaisingBackend(BackendUnavailableError("not installed")))
    client = TestClient(app)
    assert client.get("/health/ready").status_code == 200

    payload = {"file": ("a.wav", io.BytesIO(b"data"), "audio/wav")}
    assert client.post("/api/transcribe", files=payload).status_code == 503

    ready = client.get("/health/ready")
    assert ready.status_code == 503
    assert ready.json()["ready"] is False


def test_transcribe_runtime_error_returns_500() -> None:
    client = _make_client(_RaisingBackend(RuntimeError("corrupt wav")))
    payload = {"file": ("a.wav", io.BytesIO(b"data"), "audio/wav")}
    resp = client.post("/api/transcribe", files=payload)
    assert resp.status_code == 500
    assert "corrupt wav" in resp.json()["detail"]


def test_transcribe_unlink_error_swallowed(monkeypatch) -> None:
    """If temp-file cleanup fails, the response still succeeds."""
    from tests.conftest import FakeBackend

    client = _make_client(FakeBackend())
    original_unlink = Path.unlink

    def fail_unlink(self: Path, *args, **kwargs) -> None:
        if (self.suffix == ".wav" and self.parent.as_posix().startswith("/var/folders")) or (
            "/tmp" in self.parent.as_posix()
        ):
            raise OSError("nope")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    payload = {"file": ("a.wav", io.BytesIO(b"RIFF"), "audio/wav")}
    resp = client.post("/api/transcribe", files=payload)
    assert resp.status_code == 200
