"""Shared test fixtures for transcription_worker_service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api.health import router as health_router
from src.api.routes import router as transcription_router
from src.config import TranscriptionSettings, clear_settings_cache, get_settings
from src.services.backends.base import (
    TranscriptionResult,
    TranscriptSegment,
    TranscriptWord,
    WhisperBackend,
)

# A REAL registered-looking tenant uuid. Product-owner ruling: a ``tenant_id``
# may never hold a sentinel, so tests must forward a real one exactly like a
# live caller does -- never rely on the service inventing a default.
TEST_TENANT_ID = "11111111-1111-4111-8111-111111111111"


def stamp_test_tenant(app: FastAPI, tenant_id: str = TEST_TENANT_ID) -> FastAPI:
    """Stand in for ``w2v_common.auth.middleware.AuthMiddleware``.

    In production AuthMiddleware sets ``request.state.tenant_id`` from the JWT /
    license token. Test apps build the router bare, so without this every route
    would (correctly) 401. Apps that deliberately omit this prove the fail-loud
    contract -- see ``tests/unit/test_tenant_required.py``.
    """

    @app.middleware("http")
    async def _stamp_tenant(request: Request, call_next: Any) -> Any:
        request.state.tenant_id = tenant_id
        return await call_next(request)

    return app


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop env vars that would bleed into Settings defaults during tests."""
    for key in list(os.environ):
        if key.startswith(("API_", "AUTH_", "TRANSCRIPTION_", "LOG_")):
            monkeypatch.delenv(key, raising=False)
    clear_settings_cache()


class FakeBackend(WhisperBackend):
    """Deterministic test double for WhisperBackend."""

    name = "fake"

    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str | None = "int8",
    ) -> None:
        super().__init__(model_size=model_size, device=device, compute_type=compute_type)
        self._loaded = False
        self.calls: list[dict[str, Any]] = []

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def transcribe(
        self,
        audio_path: Path,
        language: str | None,
        word_timestamps: bool,
    ) -> TranscriptionResult:
        self._loaded = True
        self.calls.append(
            {
                "audio_path": audio_path,
                "language": language,
                "word_timestamps": word_timestamps,
            }
        )
        words: list[TranscriptWord] = []
        if word_timestamps:
            words = [
                TranscriptWord(start=0.0, end=0.5, word="Hello", probability=0.95),
                TranscriptWord(start=0.5, end=1.0, word="world", probability=0.9),
            ]
        segments = [
            TranscriptSegment(
                id=0,
                start=0.0,
                end=1.0,
                text="Hello world",
                words=words,
                avg_logprob=-0.2,
                no_speech_prob=0.01,
            ),
            TranscriptSegment(
                id=1,
                start=1.0,
                end=2.5,
                text="This is a fake transcription.",
                words=[],
                avg_logprob=-0.3,
                no_speech_prob=0.02,
            ),
        ]
        return TranscriptionResult(
            language=language or "en",
            duration=2.5,
            segments=segments,
            text="Hello world This is a fake transcription.",
            backend=self.name,
            model=self.model_size,
        )


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def app(fake_backend: FakeBackend) -> FastAPI:
    """Minimal FastAPI app with the fake backend wired in."""
    app = FastAPI()
    app.state.backend = fake_backend
    app.state.settings = get_settings()
    app.include_router(health_router)
    app.include_router(transcription_router)
    return stamp_test_tenant(app)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture
def transcription_settings() -> TranscriptionSettings:
    return TranscriptionSettings()
