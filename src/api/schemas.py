"""Pydantic schemas for transcription_worker_service API."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class NoSpeechReason(StrEnum):
    """WHY a transcript came back empty.

    An empty transcript has two causes that are indistinguishable in the body
    alone, and a consumer must never conflate them: ``video_analyzer_service``
    turns an empty transcript into the factual claim "this video has no speech".

    * ``SILENT_AUDIO`` — the decode SUCCEEDED over a measurable media duration
      and Whisper suppressed every segment. This is positive evidence of
      silence, and the only value a consumer may report as "no speech".
    * ``EMPTY_DECODE`` — the decode produced NOTHING usable (no segments and no
      measurable duration, or segments carrying no text). There is no evidence
      about the audio at all, so no claim may be made from it.
    """

    SILENT_AUDIO = "silent_audio"
    EMPTY_DECODE = "empty_decode"


class TranscriptWordResponse(BaseModel):
    """Word-level alignment."""

    start: float
    end: float
    word: str
    probability: float | None = None


class TranscriptSegmentResponse(BaseModel):
    """Segment-level transcript entry."""

    id: int
    start: float
    end: float
    text: str
    words: list[TranscriptWordResponse] = Field(default_factory=list)
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


class TranscriptionResponse(BaseModel):
    """Full transcription response body."""

    backend: str
    model: str
    language: str | None = None
    duration: float
    text: str
    segments: list[TranscriptSegmentResponse]
    srt: str | None = Field(default=None, description="SRT-formatted subtitles")
    no_speech: bool = Field(
        default=False,
        description=(
            "True when this response carries no transcript text. Consumers MUST "
            "check no_speech_reason before asserting the media is silent."
        ),
    )
    no_speech_reason: NoSpeechReason | None = Field(
        default=None,
        description=(
            "Null when text was transcribed. 'silent_audio' = decoded "
            "successfully over measurable audio with zero segments (genuine "
            "silence). 'empty_decode' = the decode produced nothing usable; "
            "this is NOT evidence of silence."
        ),
    )


class ModelInfo(BaseModel):
    """Backend/model info exposed via /api/models and /health."""

    backend: str
    model: str
    device: str
    compute_type: str | None = None
    loaded: bool


class ModelsResponse(BaseModel):
    """Response for GET /api/models."""

    current: ModelInfo
    available_sizes: list[str]
