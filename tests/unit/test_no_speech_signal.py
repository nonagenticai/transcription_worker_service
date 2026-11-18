"""``{"text": "", "segments": []}`` must not be indistinguishable from silence.

Downstream (``video_analyzer_service``) turns an empty transcript into the
factual claim "this video has no speech". A degenerate decode that produced
nothing must therefore be a DIFFERENT, typed value on the response — not the
same empty body a genuinely silent clip returns.
"""

from __future__ import annotations

import io
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import router as transcription_router
from src.api.schemas import NoSpeechReason, TranscriptionResponse
from src.config import Settings, clear_settings_cache
from src.services.backends.base import (
    TranscriptionResult,
    TranscriptSegment,
    WhisperBackend,
)
from tests.conftest import stamp_test_tenant


class _ScriptedBackend(WhisperBackend):
    """Returns a caller-supplied ``TranscriptionResult`` verbatim."""

    name = "scripted"

    def __init__(self, result: TranscriptionResult) -> None:
        super().__init__(model_size="tiny", device="cpu", compute_type="int8")
        self._result = result

    @property
    def is_loaded(self) -> bool:
        return True

    def load(self) -> None:
        return None

    def transcribe(
        self,
        audio_path: Path,
        language: str | None,
        word_timestamps: bool,
    ) -> TranscriptionResult:
        return self._result


def _client(result: TranscriptionResult) -> TestClient:
    clear_settings_cache()
    app = FastAPI()
    app.state.backend = _ScriptedBackend(result)
    app.state.settings = Settings()
    app.include_router(transcription_router)
    stamp_test_tenant(app)
    return TestClient(app)


def _files() -> dict[str, tuple[str, io.BytesIO, str]]:
    return {"file": ("a.wav", io.BytesIO(b"RIFF....WAVEfmt "), "audio/wav")}


def _result(
    *, duration: float, segments: list[TranscriptSegment], text: str
) -> TranscriptionResult:
    return TranscriptionResult(
        language="en",
        duration=duration,
        segments=segments,
        text=text,
        backend="scripted",
        model="tiny",
    )


def test_silent_audio_is_reported_as_genuine_silence() -> None:
    """PRODUCTION CALL SITE: POST /api/transcribe.

    Measurable media duration + zero segments == Whisper decoded the whole clip
    and suppressed everything. That, and only that, is "this clip has no speech".
    """
    client = _client(_result(duration=30.0, segments=[], text=""))
    response = client.post("/api/transcribe", files=_files())
    assert response.status_code == 200
    body = response.json()
    assert body["no_speech"] is True
    assert body["no_speech_reason"] == NoSpeechReason.SILENT_AUDIO.value


def test_empty_decode_is_not_reported_as_silence() -> None:
    """PRODUCTION CALL SITE: POST /api/transcribe.

    Zero segments AND no measurable duration means the decode yielded NOTHING —
    there is no evidence of silence, so downstream must not claim any.
    """
    client = _client(_result(duration=0.0, segments=[], text=""))
    response = client.post("/api/transcribe", files=_files())
    assert response.status_code == 200
    body = response.json()
    assert body["no_speech"] is True
    assert body["no_speech_reason"] == NoSpeechReason.EMPTY_DECODE.value, (
        "a decode that produced nothing was reported as genuine silence"
    )


def test_blank_text_with_segments_is_an_empty_decode() -> None:
    """PRODUCTION CALL SITE: POST /api/transcribe.

    Segments with timings but no text is a degenerate decode, not silence.
    """
    client = _client(
        _result(
            duration=12.0,
            segments=[TranscriptSegment(id=0, start=0.0, end=12.0, text="")],
            text="   ",
        )
    )
    body = client.post("/api/transcribe", files=_files()).json()
    assert body["no_speech"] is True
    assert body["no_speech_reason"] == NoSpeechReason.EMPTY_DECODE.value


def test_real_transcript_is_not_flagged() -> None:
    """PRODUCTION CALL SITE: POST /api/transcribe."""
    client = _client(
        _result(
            duration=2.0,
            segments=[TranscriptSegment(id=0, start=0.0, end=2.0, text="hello")],
            text="hello",
        )
    )
    body = client.post("/api/transcribe", files=_files()).json()
    assert body["no_speech"] is False
    assert body["no_speech_reason"] is None


def test_no_speech_is_a_declared_schema_field_not_an_adhoc_key() -> None:
    """The contract must be typed so a downstream client can branch on it."""
    assert "no_speech" in TranscriptionResponse.model_fields
    assert "no_speech_reason" in TranscriptionResponse.model_fields
    schema = TranscriptionResponse.model_json_schema()
    assert schema["properties"]["no_speech"]["type"] == "boolean"
    assert set(NoSpeechReason) == {NoSpeechReason.SILENT_AUDIO, NoSpeechReason.EMPTY_DECODE}
