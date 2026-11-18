"""openai-whisper backend (reference implementation, slower than faster-whisper)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

from src.services.backends.base import (
    BackendUnavailableError,
    TranscriptionResult,
    TranscriptSegment,
    TranscriptWord,
    WhisperBackend,
)

logger = logging.getLogger(__name__)


def _last_segment_end(segments: list[TranscriptSegment]) -> float:
    """Degraded duration fallback: end of the last transcribed segment."""
    return segments[-1].end if segments else 0.0


class OpenAIWhisperBackend(WhisperBackend):
    """Transcription backend using the ``openai-whisper`` library."""

    name = "openai"

    def __init__(self, model_size: str, device: str, compute_type: str | None) -> None:
        super().__init__(model_size=model_size, device=device, compute_type=compute_type)
        self._model: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _import_native(self) -> None:
        """Import ``whisper`` (pulls in the torch C extension)."""
        import whisper  # noqa: F401  (imported for its side effect only)

    def load(self) -> None:
        if self._model is not None:
            return
        # Double-checked under the shared load lock: two concurrent first
        # requests must build ONE model, not two multi-GB copies.
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import whisper
            except ImportError as exc:
                raise BackendUnavailableError(
                    "openai-whisper is not installed; install it or set "
                    "TRANSCRIPTION_BACKEND=faster"
                ) from exc

            logger.info("Loading openai-whisper model=%s device=%s", self.model_size, self.device)
            self._model = whisper.load_model(self.model_size, device=self.device)
            logger.info("openai-whisper model loaded")

    @staticmethod
    def _media_duration(audio_path: Path) -> float | None:
        """Return the MEDIA duration in seconds, or None if it cannot be read.

        openai-whisper's result payload carries no duration field, so it is
        derived from the decoded PCM using whisper's own ffmpeg-backed loader
        (``whisper.load_audio`` returns mono float32 at ``SAMPLE_RATE``). That
        keeps this backend's ``duration`` the same quantity faster-whisper
        reports via ``info.duration`` — which is what billing charges for.

        Costs one extra ffmpeg decode of an already-local file, inside the
        concurrency-bounded worker thread. Cheap next to the Whisper forward
        pass, and the alternative is mis-billing every request.
        """
        try:
            import whisper

            audio = whisper.load_audio(str(audio_path))
            audio_module = getattr(whisper, "audio", None)
            sample_rate = float(getattr(audio_module, "SAMPLE_RATE", 16000))
            if sample_rate <= 0:
                return None
            return len(audio) / sample_rate
        except ImportError, AttributeError, OSError, RuntimeError, ValueError, TypeError:
            logger.warning(
                "Could not read media duration for %s; falling back to last segment end",
                audio_path,
                exc_info=True,
            )
            return None

    def transcribe(
        self,
        audio_path: Path,
        language: str | None,
        word_timestamps: bool,
    ) -> TranscriptionResult:
        if self._model is None:
            self.load()
        if self._model is None:
            raise BackendUnavailableError("openai-whisper model failed to load")

        result = cast(
            "dict[str, Any]",
            self._model.transcribe(
                str(audio_path),
                language=language,
                word_timestamps=word_timestamps,
            ),
        )

        segments: list[TranscriptSegment] = []
        for raw in result.get("segments", []):
            raw_seg = cast("dict[str, Any]", raw)
            words: list[TranscriptWord] = []
            if word_timestamps:
                for w in raw_seg.get("words", []) or []:
                    raw_w = cast("dict[str, Any]", w)
                    words.append(
                        TranscriptWord(
                            start=float(raw_w.get("start", 0.0)),
                            end=float(raw_w.get("end", 0.0)),
                            word=str(raw_w.get("word", "")),
                            probability=raw_w.get("probability"),
                        )
                    )
            segments.append(
                TranscriptSegment(
                    id=int(raw_seg.get("id", len(segments))),
                    start=float(raw_seg.get("start", 0.0)),
                    end=float(raw_seg.get("end", 0.0)),
                    text=str(raw_seg.get("text", "")).strip(),
                    words=words,
                    avg_logprob=raw_seg.get("avg_logprob"),
                    no_speech_prob=raw_seg.get("no_speech_prob"),
                )
            )

        # BILLED QUANTITY — must match faster-whisper's ``info.duration``
        # (media length), not the last speech end. A 600 s file whose speech
        # stops at 300 s previously billed 300 here and 600 there, purely on an
        # env var. Last-segment end survives only as a degraded fallback when
        # the media cannot be probed.
        media_duration = self._media_duration(audio_path)
        duration = media_duration if media_duration is not None else _last_segment_end(segments)
        full_text = " ".join(s.text for s in segments).strip()

        return TranscriptionResult(
            language=result.get("language"),
            duration=duration,
            segments=segments,
            text=full_text,
            backend=self.name,
            model=self.model_size,
        )
