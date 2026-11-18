"""Route-level tests for /api/transcribe and /api/models."""

from __future__ import annotations

import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.conftest import stamp_test_tenant


def test_transcribe_happy_path(client: TestClient) -> None:
    payload = {"file": ("a.wav", io.BytesIO(b"RIFF....WAVEfmt "), "audio/wav")}
    data = {"language": "en", "word_timestamps": "true", "return_srt": "true"}
    resp = client.post("/api/transcribe", files=payload, data=data)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["backend"] == "fake"
    assert body["language"] == "en"
    assert body["duration"] == pytest.approx(2.5)
    assert body["text"].startswith("Hello world")
    assert len(body["segments"]) == 2
    assert body["segments"][0]["words"], "word_timestamps=True should populate words"
    assert "Hello world" in body["srt"]


def test_transcribe_rejects_empty(client: TestClient) -> None:
    payload = {"file": ("empty.wav", io.BytesIO(b""), "audio/wav")}
    resp = client.post("/api/transcribe", files=payload)
    assert resp.status_code == 400
    assert "Empty" in resp.json()["detail"]


def test_transcribe_rejects_oversize(client: TestClient, app: FastAPI) -> None:
    app.state.settings.transcription.max_audio_bytes = 8
    payload = {"file": ("big.wav", io.BytesIO(b"this is too big"), "audio/wav")}
    resp = client.post("/api/transcribe", files=payload)
    assert resp.status_code == 413


def test_transcribe_accepts_payload_at_limit(client: TestClient, app: FastAPI) -> None:
    # Boundary: a payload exactly at the limit must NOT be rejected by the
    # bounded (max_bytes + 1) read — guards the off-by-one in the size check.
    body = b"RIFF....WAVEfmt "
    app.state.settings.transcription.max_audio_bytes = len(body)
    payload = {"file": ("a.wav", io.BytesIO(body), "audio/wav")}
    resp = client.post("/api/transcribe", files=payload)
    assert resp.status_code == 200, resp.text


def test_transcribe_missing_backend_returns_503() -> None:
    from src.api.routes import router as transcription_router

    app = FastAPI()
    app.state.settings = type(
        "S", (), {"transcription": type("T", (), {"max_audio_bytes": 10_000})()}
    )()
    app.include_router(transcription_router)
    stamp_test_tenant(app)
    client = TestClient(app)
    payload = {"file": ("a.wav", io.BytesIO(b"RIFF"), "audio/wav")}
    resp = client.post("/api/transcribe", files=payload)
    assert resp.status_code == 503


def test_models_endpoint(client: TestClient) -> None:
    resp = client.get("/api/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["current"]["backend"] == "fake"
    assert body["current"]["model"] == "tiny"
    assert "base" in body["available_sizes"]
    assert "large-v3" in body["available_sizes"]


def test_transcribe_word_timestamps_off(client: TestClient) -> None:
    payload = {"file": ("a.wav", io.BytesIO(b"RIFF"), "audio/wav")}
    data = {"word_timestamps": "false"}
    resp = client.post("/api/transcribe", files=payload, data=data)
    assert resp.status_code == 200
    body = resp.json()
    # FakeBackend emits [] for words when word_timestamps=False
    assert body["segments"][0]["words"] == []
