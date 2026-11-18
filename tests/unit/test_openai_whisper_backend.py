"""openai-whisper backend wrapper tests (mocked whisper module)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from src.services.backends.base import BackendUnavailableError
from src.services.backends.openai_whisper_backend import OpenAIWhisperBackend


class _FakePcm:
    """Stand-in for the decoded PCM array — only its length is read."""

    def __init__(self, samples: int) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return self._samples


def _install_fake_whisper(
    monkeypatch: pytest.MonkeyPatch,
    result_payload: dict[str, Any],
    media_seconds: float | None = None,
) -> None:
    """Stub the whisper module.

    ``media_seconds=None`` omits ``load_audio`` entirely, exercising the
    degraded last-segment-end fallback used when the media cannot be probed.
    """

    class _FakeModel:
        def transcribe(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return result_payload

    fake_mod = types.ModuleType("whisper")
    setattr(fake_mod, "load_model", lambda size, device=None: _FakeModel())  # noqa: B010
    if media_seconds is not None:
        audio_mod = types.ModuleType("whisper.audio")
        setattr(audio_mod, "SAMPLE_RATE", 16000)  # noqa: B010
        setattr(fake_mod, "audio", audio_mod)  # noqa: B010
        setattr(  # noqa: B010
            fake_mod, "load_audio", lambda path: _FakePcm(int(media_seconds * 16000))
        )
    monkeypatch.setitem(sys.modules, "whisper", fake_mod)


def test_transcribe_with_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "language": "en",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.2,
                "text": " Hello world",
                "avg_logprob": -0.25,
                "no_speech_prob": 0.03,
                "words": [
                    {"start": 0.0, "end": 0.6, "word": "Hello", "probability": 0.95},
                    {"start": 0.6, "end": 1.2, "word": "world", "probability": 0.9},
                ],
            },
            {
                "id": 1,
                "start": 1.2,
                "end": 2.5,
                "text": " Second line",
                "avg_logprob": -0.3,
                "no_speech_prob": 0.05,
                "words": [],
            },
        ],
    }
    _install_fake_whisper(monkeypatch, payload)

    backend = OpenAIWhisperBackend(model_size="tiny", device="cpu", compute_type=None)
    assert not backend.is_loaded
    result = backend.transcribe(Path("/tmp/fake.wav"), language="en", word_timestamps=True)
    assert backend.is_loaded
    assert result.backend == "openai"
    assert result.language == "en"
    assert result.duration == pytest.approx(2.5)
    assert len(result.segments) == 2
    assert result.segments[0].words[0].word == "Hello"
    assert result.text.startswith("Hello world")


def test_transcribe_without_word_timestamps(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "language": "en",
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.0, "text": " Hi", "words": []},
        ],
    }
    _install_fake_whisper(monkeypatch, payload)
    backend = OpenAIWhisperBackend(model_size="tiny", device="cpu", compute_type=None)
    result = backend.transcribe(Path("/tmp/fake.wav"), language=None, word_timestamps=False)
    assert result.segments[0].words == []


def test_load_without_library_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "whisper", None)
    backend = OpenAIWhisperBackend(model_size="tiny", device="cpu", compute_type=None)
    with pytest.raises(BackendUnavailableError):
        backend.load()


def test_transcribe_empty_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_whisper(monkeypatch, {"language": "en", "segments": []})
    backend = OpenAIWhisperBackend(model_size="tiny", device="cpu", compute_type=None)
    result = backend.transcribe(Path("/tmp/fake.wav"), language="en", word_timestamps=False)
    assert result.segments == []
    assert result.duration == 0.0
    assert result.text == ""


def test_duration_is_media_length_not_last_speech_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """BILLING: a file whose speech stops early still bills the whole media."""
    payload = {
        "language": "en",
        "segments": [{"id": 0, "start": 0.0, "end": 2.5, "text": " Hello", "words": []}],
    }
    _install_fake_whisper(monkeypatch, payload, media_seconds=10.0)
    backend = OpenAIWhisperBackend(model_size="tiny", device="cpu", compute_type=None)
    result = backend.transcribe(Path("/tmp/fake.wav"), language="en", word_timestamps=False)
    assert result.duration == pytest.approx(10.0)


def test_silent_media_bills_full_media_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    """A silent file produces no segments but still consumed the whole GPU run."""
    _install_fake_whisper(monkeypatch, {"language": "en", "segments": []}, media_seconds=30.0)
    backend = OpenAIWhisperBackend(model_size="tiny", device="cpu", compute_type=None)
    result = backend.transcribe(Path("/tmp/fake.wav"), language="en", word_timestamps=False)
    assert result.duration == pytest.approx(30.0)


def test_duration_falls_back_when_sample_rate_is_bogus(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "language": "en",
        "segments": [{"id": 0, "start": 0.0, "end": 2.5, "text": " Hello", "words": []}],
    }
    _install_fake_whisper(monkeypatch, payload, media_seconds=10.0)
    import whisper

    monkeypatch.setattr(whisper.audio, "SAMPLE_RATE", 0)
    backend = OpenAIWhisperBackend(model_size="tiny", device="cpu", compute_type=None)
    result = backend.transcribe(Path("/tmp/fake.wav"), language="en", word_timestamps=False)
    assert result.duration == pytest.approx(2.5)


def test_duration_falls_back_to_last_segment_when_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "language": "en",
        "segments": [{"id": 0, "start": 0.0, "end": 2.5, "text": " Hello", "words": []}],
    }
    _install_fake_whisper(monkeypatch, payload, media_seconds=10.0)
    import whisper  # the fake installed above

    def _boom(path: str) -> list[float]:
        raise RuntimeError("ffmpeg missing")

    monkeypatch.setattr(whisper, "load_audio", _boom)
    backend = OpenAIWhisperBackend(model_size="tiny", device="cpu", compute_type=None)
    result = backend.transcribe(Path("/tmp/fake.wav"), language="en", word_timestamps=False)
    assert result.duration == pytest.approx(2.5)
