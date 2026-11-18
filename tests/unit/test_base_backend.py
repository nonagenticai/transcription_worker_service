"""Sanity tests for the WhisperBackend abstract base."""

from __future__ import annotations

from pathlib import Path

from src.services.backends.base import (
    TranscriptionResult,
    TranscriptSegment,
    TranscriptWord,
    WhisperBackend,
)


class _MinimalBackend(WhisperBackend):
    """Concrete subclass that doesn't override ``is_loaded``."""

    name = "minimal"

    def load(self) -> None:
        pass

    def transcribe(
        self,
        audio_path: Path,
        language: str | None,
        word_timestamps: bool,
    ) -> TranscriptionResult:
        return TranscriptionResult(
            language="en",
            duration=0.0,
            segments=[],
            text="",
            backend=self.name,
            model=self.model_size,
        )


def test_default_is_loaded_returns_false() -> None:
    backend = _MinimalBackend(model_size="tiny", device="cpu", compute_type="int8")
    assert backend.is_loaded is False


def test_transcript_word_dataclass() -> None:
    word = TranscriptWord(start=0.0, end=0.5, word="hi", probability=0.9)
    assert word.word == "hi"
    assert word.probability == 0.9


def test_transcript_segment_defaults() -> None:
    seg = TranscriptSegment(id=0, start=0.0, end=1.0, text="hi")
    assert seg.words == []
    assert seg.avg_logprob is None
