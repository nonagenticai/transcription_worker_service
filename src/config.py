"""Configuration for transcription_worker_service (port 6318)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from w2v_common.config import APISettings as _BaseAPISettings
from w2v_common.config import AuthSettings, LoggingSettings

__all__ = [
    "APISettings",
    "AuthSettings",
    "LoggingSettings",
    "MeteringSettings",
    "Settings",
    "TranscriptionSettings",
    "clear_settings_cache",
    "get_settings",
]


class APISettings(_BaseAPISettings):
    """Transcription worker API settings."""

    port: int = Field(default=6318, description="API server port")


class TranscriptionSettings(BaseSettings):
    """Transcription backend configuration."""

    backend: Literal["faster", "openai"] = Field(
        default="faster",
        description="Whisper backend selection (faster=faster-whisper, openai=openai-whisper)",
    )
    model_size: str | None = Field(
        default=None,
        description="Whisper model size (tiny/base/small/medium/large-v3). Auto if None.",
    )
    language: str | None = Field(
        default=None,
        description="ISO-639-1 language code (None = auto-detect).",
    )
    device: Literal["auto", "cpu", "cuda"] = Field(
        default="auto",
        description="Compute device (auto-detects CUDA when 'auto').",
    )
    compute_type: str | None = Field(
        default=None,
        description="faster-whisper compute type (int8/float16/float32). Auto if None.",
    )
    word_timestamps: bool = Field(
        default=True, description="Default word_timestamps value when client omits it"
    )
    max_audio_bytes: int = Field(
        default=200 * 1024 * 1024,
        description="Max accepted upload size in bytes (200 MiB default)",
    )
    load_on_startup: bool = Field(
        default=False,
        description="Preload the model at service startup (slower boot, faster first request)",
    )
    model_config = SettingsConfigDict(
        env_prefix="TRANSCRIPTION_",
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )


class MeteringSettings(BaseSettings):
    """Cost-event metering configuration.

    Emits per-transcription cost events to a Redis stream consumed by
    metering_service. ``redis_url`` is SEPARATE from any other Redis the
    service uses. Leave empty to disable emission silently (events drop).

    The cost-bus stream name is resolved centrally by ``w2v_common.metering``
    (``METERING_STREAM`` env, else the canonical ``improvement-loop:costs``);
    there is intentionally no per-service ``stream`` field to drift out of sync.
    """

    redis_url: str = Field(default="", description="Metering Redis URL")

    model_config = SettingsConfigDict(
        env_prefix="METERING_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class Settings(BaseSettings):
    """Main application settings."""

    app_name: str = Field(default="Transcription Worker", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    environment: str = Field(default="development", description="Deployment environment")

    api: APISettings = Field(default_factory=APISettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    transcription: TranscriptionSettings = Field(default_factory=TranscriptionSettings)
    metering: MeteringSettings = Field(default_factory=MeteringSettings)

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return cached singleton settings."""
    return Settings()


def clear_settings_cache() -> None:
    """Drop the cached settings (tests)."""
    get_settings.cache_clear()
