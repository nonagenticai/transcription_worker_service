"""faster-whisper (CTranslate2) backend.

compute_type defaults
---------------------
When ``compute_type`` is ``None``, the backend selects:

- ``float16`` on CUDA devices (fast, supported by all modern NVIDIA GPUs).
- ``int8`` on CPU (best CPU throughput; faster-whisper supports int8/int8_float32).

Override via the ``TRANSCRIPTION_COMPUTE_TYPE`` env var (e.g. ``int8_float16``,
``float32``). See faster-whisper docs for the full matrix.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.services.backends.base import (
    BackendUnavailableError,
    TranscriptionResult,
    TranscriptSegment,
    TranscriptWord,
    WhisperBackend,
)

logger = logging.getLogger(__name__)


class FasterWhisperBackend(WhisperBackend):
    """Transcription backend using the ``faster-whisper`` library."""

    name = "faster"

    def __init__(self, model_size: str, device: str, compute_type: str | None) -> None:
        super().__init__(model_size=model_size, device=device, compute_type=compute_type)
        self._model: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _import_native(self) -> None:
        """Import ``faster_whisper`` (pulls in the ctranslate2 C extension)."""
        import faster_whisper  # noqa: F401  (imported for its side effect only)

    def load(self) -> None:
        if self._model is not None:
            return
        # Double-checked under the shared load lock: two concurrent first
        # requests must build ONE model, not two multi-GB copies.
        with self._load_lock:
            if self._model is not None:
                return
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise BackendUnavailableError(
                    "faster-whisper is not installed; install it or set "
                    "TRANSCRIPTION_BACKEND=openai"
                ) from exc

            compute_type = self.compute_type or ("float16" if self.device == "cuda" else "int8")
            logger.info(
                "Loading faster-whisper model=%s device=%s compute_type=%s",
                self.model_size,
                self.device,
                compute_type,
            )
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=compute_type,
            )
            self.compute_type = compute_type
            logger.info("faster-whisper model loaded")

    def transcribe(
        self,
        audio_path: Path,
        language: str | None,
        word_timestamps: bool,
    ) -> TranscriptionResult:
        if self._model is None:
            self.load()
        if self._model is None:
            raise BackendUnavailableError("faster-whisper model failed to load")

        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=word_timestamps,
            vad_filter=False,
        )

        segments: list[TranscriptSegment] = []
        for idx, seg in enumerate(segments_iter):
            words: list[TranscriptWord] = []
            if word_timestamps and getattr(seg, "words", None):
                for w in seg.words:
                    words.append(
                        TranscriptWord(
                            start=float(w.start),
                            end=float(w.end),
                            word=w.word,
                            probability=float(w.probability)
                            if getattr(w, "probability", None) is not None
                            else None,
                        )
                    )
            segments.append(
                TranscriptSegment(
                    id=idx,
                    start=float(seg.start),
                    end=float(seg.end),
                    text=seg.text.strip(),
                    words=words,
                    avg_logprob=float(seg.avg_logprob)
                    if getattr(seg, "avg_logprob", None) is not None
                    else None,
                    no_speech_prob=float(seg.no_speech_prob)
                    if getattr(seg, "no_speech_prob", None) is not None
                    else None,
                )
            )

        full_text = " ".join(s.text for s in segments).strip()

        # ``info.duration`` is the MEDIA duration — the billed quantity
        # (see TranscriptionResult.duration). Never substitute the last
        # segment end here: that would under-bill files with trailing silence
        # and bill 0 for a silent file the GPU still processed in full.
        return TranscriptionResult(
            language=info.language if info is not None else language,
            duration=float(info.duration) if info is not None else 0.0,
            segments=segments,
            text=full_text,
            backend=self.name,
            model=self.model_size,
        )
