"""The tenant is REQUIRED at the request boundary — no sentinel, no default.

Product-owner ruling: a ``tenant_id`` may NEVER hold a sentinel. All traffic
carries a real registered tenant forwarded from the requestor; an absent tenant
is an ERROR, never a substituted default.

Regression this pins: ``routes.py`` used to do
``tenant_id = request.state.tenant_id or DEFAULT_TENANT_ID`` where
``DEFAULT_TENANT_ID`` was the nil UUID. Because the nil UUID is a VALID uuid it
passed every cast and dirty-row check, and billing bucketed it as
``__unattributed__`` and NEVER CHARGED IT — real GPU transcription, silently
unbilled.

Both halves matter and are asserted here:
  1. a request with no tenant FAILS (401), and
  2. it emits NO cost event (``redis.xadd`` never awaited).

This service has NO queue/stream consumer: HTTP is the only ingress (Redis is
producer-only, via ``xadd`` on the cost bus), so the request boundary is the
only place a tenant can arrive and the only place that needs the gate.
"""

from __future__ import annotations

import io
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import router as transcription_router
from src.config import get_settings
from tests.conftest import TEST_TENANT_ID, FakeBackend, stamp_test_tenant


def _files() -> dict[str, tuple[str, io.BytesIO, str]]:
    return {"file": ("a.wav", io.BytesIO(b"RIFF" * 64), "audio/wav")}


def _build_app(*, tenant: str | None) -> tuple[FastAPI, AsyncMock]:
    """Build the real router app. ``tenant=None`` omits the auth stamp entirely."""
    app = FastAPI()
    app.state.backend = FakeBackend()
    app.state.settings = get_settings()
    metering_redis = AsyncMock()
    app.state.metering_redis = metering_redis
    app.include_router(transcription_router)
    if tenant is not None:
        stamp_test_tenant(app, tenant)
    return app, metering_redis


def test_transcribe_without_tenant_401s_and_emits_no_cost_event() -> None:
    app, metering_redis = _build_app(tenant=None)
    with TestClient(app) as client:
        resp = client.post("/api/transcribe", files=_files())

    assert resp.status_code == 401, resp.text
    assert "tenant" in resp.json()["detail"].lower()
    # Half two: nothing unattributed reached the cost bus.
    metering_redis.xadd.assert_not_awaited()


def test_transcribe_with_blank_tenant_401s_and_emits_no_cost_event() -> None:
    """A present-but-blank tenant is just as absent — no substitution."""
    app, metering_redis = _build_app(tenant="   ")
    with TestClient(app) as client:
        resp = client.post("/api/transcribe", files=_files())

    assert resp.status_code == 401, resp.text
    metering_redis.xadd.assert_not_awaited()


@pytest.mark.parametrize(
    "nil_spelling",
    [
        "00000000-0000-0000-0000-000000000000",
        "00000000000000000000000000000000",
        "{00000000-0000-0000-0000-000000000000}",
        "urn:uuid:00000000-0000-0000-0000-000000000000",
        "  00000000-0000-0000-0000-000000000000  ",
    ],
)
def test_transcribe_with_the_nil_uuid_401s_and_emits_no_cost_event(nil_spelling: str) -> None:
    """The nil uuid is a VALID uuid, so the blank-only guard never sees it.

    It must be rejected BY VALUE in every spelling, or the exact sentinel this
    service just deleted walks straight back in through the header.
    """
    app, metering_redis = _build_app(tenant=nil_spelling)
    with TestClient(app) as client:
        resp = client.post("/api/transcribe", files=_files())

    assert resp.status_code == 401, resp.text
    metering_redis.xadd.assert_not_awaited()


def test_transcribe_rejects_before_any_decode_is_spent() -> None:
    """The gate runs BEFORE the GPU decode, not after — an unattributed request
    must not consume a model slot and then be rejected at emission time."""
    app, _ = _build_app(tenant=None)
    backend: FakeBackend = app.state.backend
    with TestClient(app) as client:
        assert client.post("/api/transcribe", files=_files()).status_code == 401
    assert backend.calls == []


def test_transcribe_with_a_real_tenant_succeeds() -> None:
    """Mutation guard: the 401s above must come from the MISSING tenant, not
    from a route that is broken for everyone."""
    app, _ = _build_app(tenant=TEST_TENANT_ID)
    with TestClient(app) as client:
        resp = client.post("/api/transcribe", files=_files())
    assert resp.status_code == 200, resp.text


def test_no_sentinel_symbol_is_importable_from_routes() -> None:
    """Nobody may re-add the substitution by re-importing the sentinel."""
    from src.api import routes

    assert not hasattr(routes, "DEFAULT_TENANT_ID")
    assert not hasattr(routes, "UNATTRIBUTED_TENANT_ID")
