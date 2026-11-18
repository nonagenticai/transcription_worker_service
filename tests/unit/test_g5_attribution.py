"""G5: every transcription cost event carries full attribution, end to end.

These drive the REAL ``w2v_common`` emit (only the redis connection is a
mock), so the asserted dict is exactly what lands on ``billing:costs``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
import w2v_common.metering.cost_bus as cost_bus
from w2v_common.auth import clear_identity_context, set_identity_context

from src.api import routes

TENANT = "3f2a1b4c-5d6e-4f70-8912-a3b4c5d6e7f8"
LIC = "0199e5a0-1111-7000-8000-000000000001"
SUBJ = "0199e5a0-2222-7000-8000-000000000002"
CALL_A = "0199e5a0-aaaa-7000-8000-00000000000a"
CALL_B = "0199e5a0-bbbb-7000-8000-00000000000b"
_RESULT = SimpleNamespace(backend="openai", model="large-v3", duration=12.5, text="hi there")


@pytest.fixture(autouse=True)
def _clean_identity() -> Any:
    clear_identity_context()
    yield
    clear_identity_context()


async def _emit(*, call_id: str, headers: dict[str, str] | None = None) -> dict[str, str]:
    redis = AsyncMock()
    redis.xadd = AsyncMock(return_value=b"1-0")
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(metering_redis=redis)),
        state=SimpleNamespace(tenant_id=TENANT),
        headers={"X-App-Id": "w2v"} if headers is None else headers,
    )
    routes._emit_cost_event(
        cast(Any, request), result=cast(Any, _RESULT), latency_ms=10, call_id=call_id
    )
    if cost_bus._background_tasks:
        await asyncio.gather(*list(cost_bus._background_tasks))
    return redis.xadd.call_args.args[1]


async def test_full_attribution_on_the_wire() -> None:
    set_identity_context(tenant_id=TENANT, license_id=LIC, subject_id=SUBJ)
    fields = await _emit(call_id=CALL_A)
    assert fields["tenant_id"] == TENANT
    assert fields["license_id"] == LIC
    assert fields["subject_id"] == SUBJ
    assert fields["app_id"] == "w2v"
    # provider "openai" is the LOCAL openai-whisper library: funding must be
    # stated, or billing's provider-name rule would call it platform_external.
    assert fields["funding"] == "local_gpu"
    assert fields["event_key"] == cost_bus.cost_event_key("w2v-transcription", "transcribe", CALL_A)


async def test_event_key_is_deterministic_per_call() -> None:
    set_identity_context(tenant_id=TENANT, license_id=LIC, subject_id=SUBJ)
    first = (await _emit(call_id=CALL_A))["event_key"]
    assert (await _emit(call_id=CALL_A))["event_key"] == first
    assert (await _emit(call_id=CALL_B))["event_key"] != first


async def test_strict_full_attribution_emits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    set_identity_context(tenant_id=TENANT, license_id=LIC, subject_id=SUBJ)
    assert (await _emit(call_id=CALL_A))["funding"] == "local_gpu"


async def test_strict_missing_licence_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    set_identity_context(tenant_id=TENANT, license_id=None, subject_id=SUBJ)
    with pytest.raises(cost_bus.MissingAttributionError, match="license_id"):
        await _emit(call_id=CALL_A)


async def test_strict_missing_app_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("METERING_REQUIRE_ATTRIBUTION", "true")
    set_identity_context(tenant_id=TENANT, license_id=LIC, subject_id=SUBJ)
    with pytest.raises(cost_bus.MissingAttributionError, match="app_id"):
        await _emit(call_id=CALL_A, headers={})
