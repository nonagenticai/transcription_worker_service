"""Billing parity: every backend must report the SAME `duration` quantity.

`duration` is billed (`input_tokens`, unit `audio_seconds`). It must be the
MEDIA duration, so that swapping `TRANSCRIPTION_BACKEND` never changes the
invoice for the same file. Regression: openai-whisper reported the end of the
last speech segment, so a 600 s file with speech ending at 300 s billed half.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from src.services.backends.faster_whisper_backend import FasterWhisperBackend
from src.services.backends.openai_whisper_backend import OpenAIWhisperBackend

MEDIA_SECONDS = 600.0
LAST_SPEECH_END = 300.0
SAMPLE_RATE = 16000


def _install_fake_faster_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    segment = types.SimpleNamespace(
        start=0.0,
        end=LAST_SPEECH_END,
        text=" Hello",
        words=[],
        avg_logprob=-0.2,
        no_speech_prob=0.01,
    )
    info = types.SimpleNamespace(language="en", duration=MEDIA_SECONDS)

    class _FakeModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def transcribe(self, *args: Any, **kwargs: Any) -> Any:
            return iter([segment]), info

    mod = types.ModuleType("faster_whisper")
    setattr(mod, "WhisperModel", _FakeModel)  # noqa: B010
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)


class _FakePcm:
    """Stand-in for the decoded PCM array — only its length is read."""

    def __init__(self, samples: int) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return self._samples


def _install_fake_openai_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "language": "en",
        "segments": [
            {"id": 0, "start": 0.0, "end": LAST_SPEECH_END, "text": " Hello", "words": []}
        ],
    }

    class _FakeModel:
        def transcribe(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return payload

    mod = types.ModuleType("whisper")
    audio_mod = types.ModuleType("whisper.audio")
    setattr(audio_mod, "SAMPLE_RATE", SAMPLE_RATE)  # noqa: B010
    setattr(mod, "audio", audio_mod)  # noqa: B010
    setattr(mod, "load_model", lambda size, device=None: _FakeModel())  # noqa: B010
    setattr(  # noqa: B010
        mod, "load_audio", lambda path: _FakePcm(int(MEDIA_SECONDS * SAMPLE_RATE))
    )
    monkeypatch.setitem(sys.modules, "whisper", mod)


def test_both_backends_bill_the_same_media_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_faster_whisper(monkeypatch)
    _install_fake_openai_whisper(monkeypatch)
    audio = Path("/tmp/parity.wav")

    faster = FasterWhisperBackend(model_size="tiny", device="cpu", compute_type="int8").transcribe(
        audio, language="en", word_timestamps=False
    )
    openai = OpenAIWhisperBackend(model_size="tiny", device="cpu", compute_type=None).transcribe(
        audio, language="en", word_timestamps=False
    )

    assert faster.duration == pytest.approx(MEDIA_SECONDS)
    assert openai.duration == pytest.approx(MEDIA_SECONDS)
    assert faster.duration == pytest.approx(openai.duration), (
        "backend choice must not change the billed quantity"
    )
    assert openai.duration != pytest.approx(LAST_SPEECH_END)
