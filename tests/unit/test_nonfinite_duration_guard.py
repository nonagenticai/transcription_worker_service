"""Non-finite whisper `duration` must not 500 a completed transcription.

A completed, GPU-consumed transcription can carry a non-finite ``duration``
(``nan``/``inf``). Two blast radii:

1. ``_audio_seconds(duration)`` -> ``math.ceil`` raises ``ValueError`` on
   ``nan`` / ``OverflowError`` on ``inf``. Called before the (unguarded)
   ``_emit_cost_event``, so the request 500s AND no cost event is emitted --
   real GPU work billed as nothing.
2. ``_serialize_result`` -> ``TranscriptionResponse(duration=nan)`` serializes
   to invalid JSON (``NaN`` / ``Infinity``), so even the valid transcript text
   never reaches the client.

The transcript is real work: a non-finite duration must be sanitized (billed 0
audio-seconds -- length unmeasurable; serialized 0.0), never crash the path.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Any, cast

import pytest


class TestAudioSecondsNonFinite:
    def test_nan_returns_zero_not_valueerror(self) -> None:
        from src.api.routes import _audio_seconds

        # Before the fix: math.ceil(nan) raises ValueError.
        assert _audio_seconds(float("nan")) == 0

    def test_inf_returns_zero_not_overflowerror(self) -> None:
        from src.api.routes import _audio_seconds

        # Before the fix: math.ceil(inf) raises OverflowError.
        assert _audio_seconds(float("inf")) == 0
        assert _audio_seconds(float("-inf")) == 0

    def test_finite_positive_still_ceils(self) -> None:
        from src.api.routes import _audio_seconds

        assert _audio_seconds(0.4) == 1
        assert _audio_seconds(125.4) == math.ceil(125.4) == 126


class TestSerializeResultNonFinite:
    def _result(self, duration: float) -> SimpleNamespace:
        return SimpleNamespace(
            backend="faster-whisper",
            model="large-v3",
            language="en",
            duration=duration,
            text="hello world",
            segments=[],
        )

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_nonfinite_duration_serializes_to_valid_json(self, bad: float) -> None:
        from src.api.routes import _serialize_result

        response = _serialize_result(cast(Any, self._result(bad)), include_srt=False)
        # Before the fix: model_dump_json raises (out-of-range float is not JSON
        # compliant) / would emit bare NaN/Infinity that json.loads rejects.
        raw = response.model_dump_json()
        parsed = json.loads(raw)
        assert parsed["duration"] == 0.0
        # The real transcript text must still survive sanitization.
        assert parsed["text"] == "hello world"

    def test_finite_duration_is_untouched(self) -> None:
        from src.api.routes import _serialize_result

        response = _serialize_result(cast(Any, self._result(125.4)), include_srt=False)
        assert json.loads(response.model_dump_json())["duration"] == 125.4


class TestCostEventUsageExtraNonFinite:
    """The emitted cost event's ``usage_extra["duration_s"]`` must be finite.

    Third blast radius (the hole this test closes): even after
    ``_audio_seconds`` and ``_serialize_result`` are guarded, ``_emit_cost_event``
    still put the RAW ``result.duration`` into ``usage_extra["duration_s"]``. The
    cost bus serializes ``usage_extra`` with ``json.dumps(..., allow_nan=True)``,
    so a non-finite duration emits bare ``NaN`` / ``Infinity`` -- invalid JSON
    (RFC 8259) that a strict metering consumer rejects, even though
    ``input_tokens`` was already 0. The audit metadata must stay valid JSON.
    """

    def _request(self, *, metering_redis: Any) -> SimpleNamespace:
        return SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(metering_redis=metering_redis)),
            state=SimpleNamespace(tenant_id="tenant-a"),
            headers={"X-App-Id": "app-1"},
        )

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_nonfinite_duration_s_is_finite_and_json_serializable(
        self, bad: float, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from unittest.mock import AsyncMock

        from src.api import routes

        captured: dict[str, Any] = {}

        def _stub_schedule(_redis: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

        monkeypatch.setattr(routes, "schedule_cost_event", _stub_schedule)

        result = SimpleNamespace(
            backend="faster-whisper",
            model="large-v3",
            duration=bad,
            text="hello world",
        )
        routes._emit_cost_event(
            cast(Any, self._request(metering_redis=AsyncMock())),
            result=cast(Any, result),
            latency_ms=100,
            call_id="0199e5a0-0000-7000-8000-000000000001",
        )

        duration_s = captured["usage_extra"]["duration_s"]
        # Before this fix: duration_s is the raw nan/inf -> not finite.
        assert math.isfinite(duration_s)
        assert duration_s == 0.0
        # And the whole usage_extra must survive strict JSON (no bare NaN/Infinity).
        json.loads(json.dumps(captured["usage_extra"], allow_nan=False))
        # Non-finite length is unmeasurable -> 0 audio-seconds billed.
        assert captured["input_tokens"] == 0

    def test_finite_duration_s_is_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import AsyncMock

        from src.api import routes

        captured: dict[str, Any] = {}

        def _stub_schedule(_redis: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

        monkeypatch.setattr(routes, "schedule_cost_event", _stub_schedule)

        result = SimpleNamespace(
            backend="faster-whisper", model="large-v3", duration=125.4, text="hello world"
        )
        routes._emit_cost_event(
            cast(Any, self._request(metering_redis=AsyncMock())),
            result=cast(Any, result),
            latency_ms=100,
            call_id="0199e5a0-0000-7000-8000-000000000001",
        )

        assert captured["usage_extra"]["duration_s"] == 125.4
        assert captured["input_tokens"] == 126
