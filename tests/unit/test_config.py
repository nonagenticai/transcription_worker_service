"""Settings defaults and env overrides."""

from __future__ import annotations

import pytest

from src.config import Settings, TranscriptionSettings, clear_settings_cache, get_settings


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("API_PORT", "TRANSCRIPTION_BACKEND", "TRANSCRIPTION_MODEL_SIZE"):
        monkeypatch.delenv(key, raising=False)
    clear_settings_cache()
    settings = Settings()
    assert settings.api.port == 6318
    assert settings.transcription.backend == "faster"
    assert settings.transcription.model_size is None
    assert settings.transcription.device == "auto"
    assert settings.transcription.max_audio_bytes == 200 * 1024 * 1024


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_PORT", "9999")
    monkeypatch.setenv("TRANSCRIPTION_BACKEND", "openai")
    monkeypatch.setenv("TRANSCRIPTION_MODEL_SIZE", "small")
    monkeypatch.setenv("TRANSCRIPTION_DEVICE", "cpu")
    clear_settings_cache()
    settings = Settings()
    assert settings.api.port == 9999
    assert settings.transcription.backend == "openai"
    assert settings.transcription.model_size == "small"
    assert settings.transcription.device == "cpu"


def test_invalid_backend_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRANSCRIPTION_BACKEND", "mojo")
    with pytest.raises(ValueError):
        TranscriptionSettings()


def test_settings_cache_singleton() -> None:
    clear_settings_cache()
    a = get_settings()
    b = get_settings()
    assert a is b
