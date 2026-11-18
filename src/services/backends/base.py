"""Abstract Whisper backend contract.

Both ``faster-whisper`` and ``openai-whisper`` concrete backends implement
this interface so the rest of the service is agnostic to which engine is
loaded at runtime.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


class BackendUnavailableError(RuntimeError):
    """Raised when a backend cannot be initialised (missing deps, bad device)."""


@dataclass(slots=True)
class TranscriptWord:
    """A single word with timing information."""

    start: float
    end: float
    word: str
    probability: float | None = None


@dataclass(slots=True)
class TranscriptSegment:
    """A transcribed segment with timestamps and optional word alignments."""

    id: int
    start: float
    end: float
    text: str
    words: list[TranscriptWord] = field(default_factory=list)
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


@dataclass(slots=True)
class TranscriptionResult:
    """Result of a full transcription run.

    ``duration`` is the **media duration in seconds** — the length of the
    submitted audio, NOT the end of the last speech segment. It is the BILLED
    quantity (``input_tokens``, ``unit=audio_seconds``), so every backend must
    report the same thing: it is what the tenant submitted and what the model
    actually processed, silence included. Backends that cannot get it from the
    engine derive it from the decoded PCM.
    """

    language: str | None
    duration: float
    segments: list[TranscriptSegment]
    text: str
    backend: str
    model: str


class WhisperBackend(ABC):
    """Abstract base class for Whisper transcription backends."""

    name: str = "base"

    def __init__(self, model_size: str, device: str, compute_type: str | None) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        # ``load()`` runs on worker threads (the route calls ``transcribe`` via
        # ``asyncio.to_thread``, and lazy loading happens inside it), so the
        # check-then-set on ``_model`` needs a THREADING lock, not an asyncio
        # one — an asyncio primitive cannot be held across a thread boundary.
        # Without it two simultaneous first requests each build a full
        # ``large-v3`` model, doubling peak RSS on a launchd Mac.
        self._load_lock = threading.Lock()

    def preimport(self) -> None:
        """Import this backend's native extension module on the CALLING thread.

        ``load()`` is reached from ``asyncio.to_thread`` (``src/api/routes.py``
        → ``backend.transcribe`` → lazy ``load()``), so without this the FIRST
        import of ``faster_whisper``/ctranslate2 or ``whisper``/torch executes
        on an arbitrary worker thread of the default executor. Importing a
        native extension off the main thread is not something CPython
        guarantees: module init may install signal handlers, spin up runtime
        thread pools, or cache thread-local state, and the failure modes are
        crashes rather than exceptions.

        Calling this from the ASGI lifespan (main thread) moves ONLY the module
        import forward. The expensive weight load stays lazy, so startup cost
        and resident memory are unchanged unless ``load_on_startup`` is set.

        Degrades to a no-op on ANY failure: a backend whose native module is
        missing or broken must still start and surface that as a 503 from
        ``load()``, exactly as before — this is a placement fix, not a new
        startup gate.
        """
        try:
            self._import_native()
        # BLE001 suppressed deliberately. A curated except-tuple here would encode
        # "the failures I happened to think of" while the docstring promises this
        # NEVER changes startup outcome. Native module init fails in shapes no
        # tuple predicts (ImportError, OSError from a bad dylib, RuntimeError from
        # a runtime probe, library-specific subclasses). Anything narrower would
        # convert a previously-503-at-first-request into a dead launchd unit.
        except Exception:  # noqa: BLE001
            logger.warning(
                "Pre-importing the native module for backend=%s failed; it will be "
                "imported lazily on a worker thread instead",
                self.name,
                exc_info=True,
            )

    def _import_native(self) -> None:
        """Import the native extension module. Overridden by each backend.

        Deliberately concrete-and-empty rather than ``@abstractmethod``: a
        backend with no native dependency (a stub or a pure-Python engine) is
        correct to do nothing here, and forcing it to declare an empty override
        would be noise.
        """
        logger.debug("backend=%s declares no native module to pre-import", self.name)

    @abstractmethod
    def load(self) -> None:
        """Load the underlying model. Can be called lazily or at startup.

        Implementations MUST be idempotent and safe to call from several
        threads at once (double-checked under ``self._load_lock``).
        """

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        language: str | None,
        word_timestamps: bool,
    ) -> TranscriptionResult:
        """Transcribe a local audio file and return structured segments."""

    @property
    def is_loaded(self) -> bool:
        """Return True once the backend has loaded its model weights."""
        return False
