"""Unit tests for the transcription cost-metering delegation.

Emission, field marshalling, stream resolution and GC-safe task retention now
live in ``w2v_common.metering`` (and are tested there). These tests cover only
this service's responsibilities: the routing identity it stamps onto every
event (``service_key`` / ``cost_source``), the audio-seconds encoding, the
REQUIRED tenant (no sentinel, ever), and the no-op-when-disabled guard.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

import src.services.metering as metering_module
from src.services.metering import (
    COST_SOURCE,
    SERVICE_KEY,
    close_metering_redis,
    init_metering_redis,
    schedule_cost_event,
)


def _make_request(*, metering_redis: Any, tenant: Any = "__unset__") -> SimpleNamespace:
    state = SimpleNamespace() if tenant == "__unset__" else SimpleNamespace(tenant_id=tenant)
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(metering_redis=metering_redis)),
        state=state,
        headers={"X-App-Id": "app-1"},
    )


_CALL = "0199e5a0-0000-7000-8000-000000000001"

_RESULT = SimpleNamespace(
    backend="faster-whisper",
    model="large-v3",
    duration=125.4,
    text="hello world",
)


class TestRoutingIdentity:
    """The service-owned constants are stable routing slugs, not secrets."""

    def test_service_key_and_cost_source(self) -> None:
        assert SERVICE_KEY == "w2v-transcription"  # pragma: allowlist secret -- routing slug
        assert COST_SOURCE == "w2v-transcription"

    def test_no_tenant_sentinel_is_reachable_from_this_package(self) -> None:
        """Product-owner ruling: a ``tenant_id`` may NEVER hold a sentinel.

        This asserted the OPPOSITE before: ``DEFAULT_TENANT_ID ==
        UNATTRIBUTED_TENANT_ID == "00000000-..."``. The nil UUID is a VALID
        uuid, so it passed every cast and dirty-row check while billing bucketed
        it as ``__unattributed__`` and never charged it. No sentinel may be
        reachable from this package -- not as an attribute, not via ``__all__``,
        and not via a ``DEFAULT_TENANT_ID`` env override.
        """
        assert not hasattr(metering_module, "DEFAULT_TENANT_ID")
        assert not hasattr(metering_module, "UNATTRIBUTED_TENANT_ID")
        assert "DEFAULT_TENANT_ID" not in metering_module.__all__
        assert "UNATTRIBUTED_TENANT_ID" not in metering_module.__all__
        # ``hasattr`` above also proves the name is not IMPORTED from
        # w2v_common: w2v_common 0.67.0 DELETES ``UNATTRIBUTED_TENANT_ID``, so
        # importing it would break this service at import time the moment that
        # lock bump lands on main.

    def test_helper_lifecycle_reexported(self) -> None:
        # The Redis lifecycle is re-exported from w2v_common so importers do not
        # depend on where the implementation lives.
        from w2v_common.metering import close_metering_redis as helper_close
        from w2v_common.metering import init_metering_redis as helper_init
        from w2v_common.metering import schedule_cost_event as helper_schedule

        assert init_metering_redis is helper_init
        assert close_metering_redis is helper_close
        assert schedule_cost_event is helper_schedule


class TestEmitCostEventDelegation:
    """``routes._emit_cost_event`` delegates to the centralized helper with the
    service's routing identity + audio-seconds encoding."""

    def test_schedules_with_service_identity_and_encoding(self, monkeypatch) -> None:
        from src.api import routes

        captured: dict[str, Any] = {}

        def _stub_schedule(_redis: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

        monkeypatch.setattr(routes, "schedule_cost_event", _stub_schedule)

        request = _make_request(metering_redis=AsyncMock(), tenant="tenant-a")
        routes._emit_cost_event(
            cast(Any, request), result=cast(Any, _RESULT), latency_ms=8500, call_id=_CALL
        )

        # Service-owned routing identity.
        assert captured["service_key"] == SERVICE_KEY
        assert captured["cost_source"] == COST_SOURCE
        # Tenant comes off request.state.
        assert captured["tenant_id"] == "tenant-a"
        assert captured["app_id"] == "app-1"
        # Audio-seconds encoding (EMITTER_GUIDE for non-token workloads).
        assert captured["provider"] == "faster-whisper"
        assert captured["model"] == "large-v3"
        # Audio-seconds bill CEIL of the media duration (a started second),
        # never truncated -- 125.4 s -> 126, so a positive clip never zero-bills.
        assert captured["input_tokens"] == math.ceil(_RESULT.duration) == 126
        assert captured["output_tokens"] == len(_RESULT.text)
        assert captured["latency_ms"] == 8500
        assert captured["cost_usd"] == 0.0
        assert captured["usage_extra"] == {
            "unit": "audio_seconds",
            "duration_s": _RESULT.duration,
            "backend": "faster-whisper",
        }

    def test_missing_tenant_raises_401_and_emits_nothing(self, monkeypatch) -> None:
        """No tenant on request.state -> 401, and NO cost event on the bus.

        This test asserted the OPPOSITE (``tenant_id == DEFAULT_TENANT_ID ==
        UNATTRIBUTED_TENANT_ID``). An absent tenant is an ERROR, never a
        substituted default -- and an unattributed event must never reach the
        cost stream, because billing silently drops the nil bucket.
        """
        from src.api import routes

        called = False

        def _stub_schedule(*_args: Any, **_kwargs: Any) -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(routes, "schedule_cost_event", _stub_schedule)

        redis = AsyncMock()
        request = _make_request(metering_redis=redis)  # no tenant_id attr

        with pytest.raises(HTTPException) as exc:
            routes._emit_cost_event(
                cast(Any, request), result=cast(Any, _RESULT), latency_ms=100, call_id=_CALL
            )

        assert exc.value.status_code == 401
        assert called is False
        redis.xadd.assert_not_awaited()

    @pytest.mark.parametrize("blank", ["", "   ", None])
    def test_blank_tenant_raises_401_and_emits_nothing(self, monkeypatch, blank) -> None:
        """A present-but-blank tenant is just as absent. Same contract."""
        from src.api import routes

        called = False

        def _stub_schedule(*_args: Any, **_kwargs: Any) -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(routes, "schedule_cost_event", _stub_schedule)

        redis = AsyncMock()
        request = _make_request(metering_redis=redis, tenant=blank)

        with pytest.raises(HTTPException) as exc:
            routes._emit_cost_event(
                cast(Any, request), result=cast(Any, _RESULT), latency_ms=100, call_id=_CALL
            )

        assert exc.value.status_code == 401
        assert called is False
        redis.xadd.assert_not_awaited()

    def test_noop_when_metering_disabled(self, monkeypatch) -> None:
        from src.api import routes

        called = False

        def _stub_schedule(*_args: Any, **_kwargs: Any) -> None:
            nonlocal called
            called = True

        monkeypatch.setattr(routes, "schedule_cost_event", _stub_schedule)

        request = _make_request(metering_redis=None, tenant="t")
        routes._emit_cost_event(
            cast(Any, request), result=cast(Any, _RESULT), latency_ms=100, call_id=_CALL
        )

        assert called is False


class TestSubSecondBillingFloor:
    """A real transcription of a sub-second clip must never bill zero seconds.

    Regression: ``input_tokens=int(result.duration)`` truncates, so any clip
    shorter than 1.0 s billed 0 audio-seconds -- real work charged as nothing.
    """

    def test_audio_seconds_ceils_positive_subsecond_to_one(self) -> None:
        from src.api.routes import _audio_seconds

        assert _audio_seconds(0.4) == 1  # would be int(0.4) == 0 before the fix
        assert _audio_seconds(0.01) == 1
        assert _audio_seconds(1.0) == 1
        assert _audio_seconds(1.2) == 2

    def test_audio_seconds_zero_and_negative_stay_zero(self) -> None:
        from src.api.routes import _audio_seconds

        assert _audio_seconds(0.0) == 0
        assert _audio_seconds(-3.0) == 0

    def test_subsecond_clip_bills_at_least_one_second(self, monkeypatch) -> None:
        from src.api import routes

        captured: dict[str, Any] = {}

        def _stub_schedule(_redis: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

        monkeypatch.setattr(routes, "schedule_cost_event", _stub_schedule)

        subsecond = SimpleNamespace(
            backend="faster-whisper", model="large-v3", duration=0.4, text="hi"
        )
        request = _make_request(metering_redis=AsyncMock(), tenant="tenant-a")
        routes._emit_cost_event(
            cast(Any, request), result=cast(Any, subsecond), latency_ms=50, call_id=_CALL
        )

        assert captured["input_tokens"] >= 1
        assert captured["input_tokens"] == 1
        # The true float precision is still carried for auditing.
        assert captured["usage_extra"]["duration_s"] == 0.4


class TestInitMeteringRedis:
    """Re-exported lifecycle: empty URL -> None; connect failure -> None (never raises)."""

    async def test_empty_url_returns_none(self) -> None:
        client = await init_metering_redis("")
        assert client is None

    async def test_connect_failure_returns_none(self) -> None:
        client = await init_metering_redis("redis://127.0.0.1:1/0")
        assert client is None

    async def test_close_none_is_safe(self) -> None:
        await close_metering_redis(None)
