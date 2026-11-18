"""ASGI app construction + lifespan wiring tests."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.config import clear_settings_cache


def _install_fake_faster_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def transcribe(self, *args: Any, **kwargs: Any) -> Any:
            return iter([]), types.SimpleNamespace(language="en", duration=0.0)

    mod = types.ModuleType("faster_whisper")
    setattr(mod, "WhisperModel", _FakeModel)  # noqa: B010
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)


def test_app_startup_and_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("TRANSCRIPTION_BACKEND", "faster")
    monkeypatch.setenv("TRANSCRIPTION_DEVICE", "cpu")
    monkeypatch.setenv("TRANSCRIPTION_MODEL_SIZE", "tiny")
    monkeypatch.setenv("TRANSCRIPTION_LOAD_ON_STARTUP", "true")
    _install_fake_faster_whisper(monkeypatch)
    clear_settings_cache()

    # Fresh import so lifespan uses the patched env + fake faster_whisper.
    for name in list(sys.modules):
        if name == "src.asgi" or name.startswith("src.asgi."):
            del sys.modules[name]
    from src.asgi import app

    with TestClient(app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "running"
        assert "transcribe" in body["endpoints"]

        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        hbody = health_resp.json()
        assert hbody["backend"] == "faster"
        assert hbody["model"] == "tiny"
        assert hbody["loaded"] is True


def test_app_startup_preload_failure_is_non_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("TRANSCRIPTION_BACKEND", "faster")
    monkeypatch.setenv("TRANSCRIPTION_DEVICE", "cpu")
    monkeypatch.setenv("TRANSCRIPTION_MODEL_SIZE", "tiny")
    monkeypatch.setenv("TRANSCRIPTION_LOAD_ON_STARTUP", "true")
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # type: ignore[arg-type]
    clear_settings_cache()

    for name in list(sys.modules):
        if name == "src.asgi" or name.startswith("src.asgi."):
            del sys.modules[name]
    from src.asgi import app

    with TestClient(app) as client:
        # Lifespan completes even when preload raises.
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["loaded"] is False


def test_main_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """src.asgi.main should build a uvicorn config from settings and invoke run."""
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("API_PORT", "6318")
    clear_settings_cache()

    called: dict[str, Any] = {}

    def fake_run(app: str, **kwargs: Any) -> None:
        called["app"] = app
        called["kwargs"] = kwargs

    fake_uvicorn = types.ModuleType("uvicorn")
    setattr(fake_uvicorn, "run", fake_run)  # noqa: B010
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    for name in list(sys.modules):
        if name == "src.asgi":
            del sys.modules[name]
    import src.asgi as asgi_mod

    asgi_mod.main()
    assert called["app"] == "src.asgi:app"
    assert called["kwargs"]["port"] == 6318


def test_lifespan_shutdown_closes_the_admission_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lifespan exit must shut the gate down: no new work admitted, executor stopped."""
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("TRANSCRIPTION_BACKEND", "faster")
    monkeypatch.setenv("TRANSCRIPTION_DEVICE", "cpu")
    monkeypatch.setenv("TRANSCRIPTION_MODEL_SIZE", "tiny")
    _install_fake_faster_whisper(monkeypatch)
    clear_settings_cache()
    for name in list(sys.modules):
        if name == "src.asgi" or name.startswith("src.asgi."):
            del sys.modules[name]
    from src.asgi import app
    from src.services.admission import TRANSCRIBE_LANE

    with TestClient(app):
        gate = app.state.gpu_gate
        assert gate.lane(TRANSCRIBE_LANE).closed is False
    assert gate.lane(TRANSCRIBE_LANE).closed is True, "lifespan did not shut the gate down"
