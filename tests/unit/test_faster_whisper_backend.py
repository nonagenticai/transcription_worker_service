"""faster-whisper backend wrapper tests (mocked WhisperModel)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from src.services.backends.base import BackendUnavailableError
from src.services.backends.faster_whisper_backend import FasterWhisperBackend


def _install_fake_faster_whisper(monkeypatch: pytest.MonkeyPatch, segments: list[Any]) -> None:
    """Stub out faster_whisper with a scripted WhisperModel."""

    class _FakeInfo:
        language = "en"
        duration = 3.1

    class _FakeModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        def transcribe(self, *args: Any, **kwargs: Any) -> Any:
            return iter(segments), _FakeInfo()

    fake_mod = types.ModuleType("faster_whisper")
    setattr(fake_mod, "WhisperModel", _FakeModel)  # noqa: B010
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_mod)


def _make_word(start: float, end: float, word: str, probability: float | None = 0.9) -> Any:
    return types.SimpleNamespace(start=start, end=end, word=word, probability=probability)


def _make_segment(
    start: float,
    end: float,
    text: str,
    words: list[Any] | None = None,
) -> Any:
    return types.SimpleNamespace(
        start=start,
        end=end,
        text=text,
        words=words or [],
        avg_logprob=-0.25,
        no_speech_prob=0.05,
    )


def test_transcribe_roundtrip_with_words(monkeypatch: pytest.MonkeyPatch) -> None:
    segments = [
        _make_segment(
            0.0,
            1.0,
            " Hello world",
            words=[_make_word(0.0, 0.5, "Hello"), _make_word(0.5, 1.0, "world")],
        ),
        _make_segment(1.0, 2.5, " Second line", words=[_make_word(1.0, 2.5, "Second")]),
    ]
    _install_fake_faster_whisper(monkeypatch, segments)

    backend = FasterWhisperBackend(model_size="tiny", device="cpu", compute_type=None)
    assert not backend.is_loaded
    backend.load()
    assert backend.is_loaded
    assert backend.compute_type == "int8"

    result = backend.transcribe(Path("/tmp/fake.wav"), language="en", word_timestamps=True)
    assert result.backend == "faster"
    assert result.model == "tiny"
    assert result.language == "en"
    assert result.duration == pytest.approx(3.1)
    assert len(result.segments) == 2
    assert result.segments[0].words[0].word == "Hello"
    assert result.text == "Hello world Second line"


def test_transcribe_without_word_timestamps(monkeypatch: pytest.MonkeyPatch) -> None:
    segments = [_make_segment(0.0, 1.0, " hi there", words=[_make_word(0.0, 1.0, "hi")])]
    _install_fake_faster_whisper(monkeypatch, segments)
    backend = FasterWhisperBackend(model_size="tiny", device="cpu", compute_type="int8")
    result = backend.transcribe(Path("/tmp/fake.wav"), language=None, word_timestamps=False)
    assert result.segments[0].words == []


def test_load_without_library_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    backend = FasterWhisperBackend(model_size="tiny", device="cpu", compute_type=None)
    with pytest.raises(BackendUnavailableError):
        backend.load()


def test_gpu_compute_type_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_faster_whisper(monkeypatch, [])
    backend = FasterWhisperBackend(model_size="large-v3", device="cuda", compute_type=None)
    backend.load()
    assert backend.compute_type == "float16"


def test_load_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_faster_whisper(monkeypatch, [])
    backend = FasterWhisperBackend(model_size="tiny", device="cpu", compute_type="int8")
    backend.load()
    first = backend._model
    backend.load()
    assert backend._model is first
