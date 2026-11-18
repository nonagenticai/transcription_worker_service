"""Non-finite transcript floats: what actually breaks, and what does not.

What does NOT break (do not re-add tests for it): the response body is never
malformed. pydantic 2.13's ``ser_json_inf_nan`` defaults to ``'null'`` and
``fastapi.responses.JSONResponse.render`` already passes ``allow_nan=False``,
so a degenerate decode on ``origin/main`` returns **HTTP 200** with
syntactically valid JSON — ``{"segments":[{"start":null,...}]}``. No bare
``NaN``/``Infinity``/``-Infinity`` token was ever emitted, and an earlier
version of this module claimed otherwise. Asserting "the body parses" or
"``'NaN' not in raw``" is therefore a test that CANNOT FAIL.

What DOES break, and is what these tests pin:

1. **Schema contract.** ``null`` lands in fields ``src/api/schemas.py``
   declares as REQUIRED ``float``: ``TranscriptionResponse.duration``,
   ``TranscriptSegmentResponse.start``/``end``, ``TranscriptWordResponse.
   start``/``end``. Rust ``serde`` refuses ``null`` for ``f64``; Go
   ``encoding/json`` silently coerces it to ``0.0``. The ``float | None``
   fields (``probability``, ``avg_logprob``, ``no_speech_prob``) are NOT
   violated by ``null`` — for those the fix is a deliberate value CHOICE, not
   a contract repair.
2. **``return_srt=true`` returns 500.** ``format_srt`` does
   ``round(seconds * 1000)``: ``ValueError`` on nan, ``OverflowError`` on inf.
   ``origin/main`` → 500; with the guard → 200. This is the strongest,
   most user-visible part of the change.

Mutation contract for this module: replacing ``_json_safe_float``'s body with
``return value`` MUST turn every test in ``TestNonFiniteFloatsAreSanitized``
red. Tests under ``TestPositiveControls`` are the opposite — they pass both
before and after the fix by design, and exist only to prove good data is not
mangled. They are NOT RED→GREEN evidence for anything.
"""

from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.services.backends.base import (
    TranscriptionResult,
    TranscriptSegment,
    TranscriptWord,
    WhisperBackend,
)
from tests.conftest import stamp_test_tenant

_AUDIO = {"file": ("a.wav", io.BytesIO(b"RIFF....WAVEfmt "), "audio/wav")}


class ScriptedBackend(WhisperBackend):
    """Backend returning a caller-supplied (possibly degenerate) result."""

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
    from src.api.routes import router as transcription_router
    from src.config import get_settings

    app = FastAPI()
    app.state.backend = ScriptedBackend(result)
    app.state.settings = get_settings()
    app.include_router(transcription_router)
    stamp_test_tenant(app)
    return TestClient(app)


def _post(result: TranscriptionResult, *, return_srt: bool = False) -> dict[str, Any]:
    resp = _client(result).post(
        "/api/transcribe",
        files=_AUDIO,
        data={"word_timestamps": "true", "return_srt": str(return_srt).lower()},
    )
    # 200 is a real assertion for the SRT path only: unsanitized non-finite
    # timings raise inside format_srt and the request 500s.
    assert resp.status_code == 200, resp.text
    return json.loads(resp.text)


def _degenerate_result() -> TranscriptionResult:
    return TranscriptionResult(
        language="en",
        duration=3.0,
        text="hello world",
        backend="scripted",
        model="tiny",
        segments=[
            TranscriptSegment(
                id=0,
                start=float("inf"),
                end=1.0,
                text="hello world",
                words=[
                    TranscriptWord(start=0.0, end=0.5, word="hello", probability=float("nan")),
                    TranscriptWord(start=0.5, end=1.0, word="world", probability=0.9),
                ],
                avg_logprob=float("-inf"),
                no_speech_prob=float("nan"),
            )
        ],
    )


def _finite_result() -> TranscriptionResult:
    return TranscriptionResult(
        language="en",
        duration=2.5,
        text="hello world",
        backend="scripted",
        model="tiny",
        segments=[
            TranscriptSegment(
                id=0,
                start=0.25,
                end=1.75,
                text="hello world",
                words=[
                    TranscriptWord(start=0.25, end=0.5, word="hello", probability=0.95),
                ],
                avg_logprob=-0.42,
                no_speech_prob=0.03,
            )
        ],
    )


class TestNonFiniteFloatsAreSanitized:
    """Every test here goes red if ``_json_safe_float`` becomes ``return value``."""

    def test_srt_path_does_not_500_on_nonfinite_timings(self) -> None:
        # The headline defect. Unsanitized, format_srt's round(seconds * 1000)
        # raises OverflowError on inf / ValueError on nan and the whole request
        # 500s -- origin/main returns 500 for exactly this body.
        body = _post(_degenerate_result(), return_srt=True)
        assert "hello world" in body["srt"]
        assert "00:00:00,000" in body["srt"]

    def test_required_float_fields_are_never_null(self) -> None:
        # The contract test: every field schemas.py declares as a REQUIRED
        # ``float`` must serialize as a JSON number, never ``null``. Unsanitized,
        # pydantic renders each of these as ``null`` -- valid JSON, invalid
        # against the very schema this endpoint advertises.
        result = _degenerate_result()
        result.duration = float("nan")
        result.segments[0].start = float("inf")
        result.segments[0].end = float("-inf")
        result.segments[0].words[0].start = float("nan")
        result.segments[0].words[0].end = float("inf")
        body = _post(result)

        seg = body["segments"][0]
        word = seg["words"][0]
        required_floats = {
            "duration": body["duration"],
            "segments[0].start": seg["start"],
            "segments[0].end": seg["end"],
            "segments[0].words[0].start": word["start"],
            "segments[0].words[0].end": word["end"],
        }
        for field, value in required_floats.items():
            assert isinstance(value, float), f"{field} is {value!r}, not a float"
            assert math.isfinite(value), f"{field} is non-finite: {value!r}"

    def test_every_nonfinite_field_gets_its_chosen_substitute(self) -> None:
        body = _post(_degenerate_result())
        seg = body["segments"][0]
        # Required floats -> 0.0 (in-domain "unknown", still sorts). These are
        # the assertions that fail without the fix.
        assert seg["start"] == 0.0
        assert seg["end"] == 1.0
        # Optional confidences -> null. NOTE: null is also what pydantic
        # produces unsanitized, so these lines alone prove nothing; they pin the
        # deliberate CHOICE not to fabricate a number. 0.0 is the MAXIMUM of a
        # log-probability, so -inf must never become 0.0.
        assert seg["avg_logprob"] is None
        assert seg["no_speech_prob"] is None
        assert seg["words"][0]["probability"] is None
        # A finite sibling value in the same list is untouched.
        assert seg["words"][1]["probability"] == 0.9

    def test_nan_nested_in_words_inside_segments_is_reached(self) -> None:
        result = _finite_result()
        result.segments[0].words[0].probability = float("nan")
        result.segments[0].words[0].end = float("inf")
        body = _post(result)
        word = body["segments"][0]["words"][0]
        assert word["end"] == 0.0
        assert word["probability"] is None

    def test_1e309_literal_overflows_to_inf_and_is_sanitized(self) -> None:
        # 1e309 is not representable: float("1e309") IS inf. A backend that
        # computes a runaway timing hits this without ever naming "inf".
        overflowed = float("1e309")
        assert math.isinf(overflowed)
        result = _finite_result()
        result.duration = overflowed
        result.segments[0].end = overflowed
        body = _post(result)
        assert body["duration"] == 0.0
        assert body["segments"][0]["end"] == 0.0

    def test_segment_with_empty_words_list_is_sanitized(self) -> None:
        # The no-words branch still has to sanitize the segment's own timings;
        # the empty list must not short-circuit anything.
        result = _finite_result()
        result.segments[0].words = []
        result.segments[0].start = float("nan")
        result.segments[0].no_speech_prob = float("nan")
        body = _post(result)
        seg = body["segments"][0]
        assert seg["words"] == []
        assert seg["start"] == 0.0
        assert seg["no_speech_prob"] is None

    def test_both_timings_nonfinite_yields_zero_length_srt_cue(self) -> None:
        # Documented, accepted degradation: when start AND end are both
        # unknown the cue collapses to zero length rather than 500ing or
        # inventing a duration. Pinned so the collapse stays deliberate.
        result = _finite_result()
        result.segments[0].start = float("nan")
        result.segments[0].end = float("inf")
        body = _post(result, return_srt=True)
        assert "00:00:00,000 --> 00:00:00,000" in body["srt"]


class TestPositiveControls:
    """Positive controls ONLY -- these pass before AND after the fix.

    They exist to prove good data is not mangled by the sanitizer. They are not
    evidence that anything was broken; do not cite them as such.
    """

    def test_all_finite_values_pass_through_byte_exact(self) -> None:
        body = _post(_finite_result(), return_srt=True)
        assert body["duration"] == 2.5
        seg = body["segments"][0]
        assert seg["start"] == 0.25
        assert seg["end"] == 1.75
        assert seg["avg_logprob"] == -0.42
        assert seg["no_speech_prob"] == 0.03
        assert seg["words"][0] == {
            "start": 0.25,
            "end": 0.5,
            "word": "hello",
            "probability": 0.95,
        }
        assert "00:00:00,250 --> 00:00:01,750" in body["srt"]

    def test_none_confidences_stay_none_not_zero(self) -> None:
        # None already means "backend did not report it" -- the helper must not
        # invent a value for it either.
        result = _finite_result()
        result.segments[0].avg_logprob = None
        result.segments[0].no_speech_prob = None
        result.segments[0].words[0].probability = None
        body = _post(result)
        seg = body["segments"][0]
        assert seg["avg_logprob"] is None
        assert seg["no_speech_prob"] is None
        assert seg["words"][0]["probability"] is None
