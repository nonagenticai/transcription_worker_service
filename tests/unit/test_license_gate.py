"""The entitlement gate on the /api router actually refuses an unlicensed tenant.

Until this change transcription_worker_service authenticated the licence token
for tenant IDENTITY and stopped there: ``AuthMiddleware`` answers "who are
you", never "may you". A tenant whose w2v licence was ``expired`` or
``revoked`` — but who held any syntactically valid platform token — still got
``POST /api/transcribe`` and burned Whisper compute on mini1, tailnet-wide.
``LICENSE_GATE`` is mounted router-wide on the single top-level ``/api`` router
(``src/api/routes.py``) to close that.

Every test here drives a REAL HTTP request through the REAL routers, because
the interesting question is not whether ``require_license`` works — w2v_common
owns that — but whether THIS service's routes actually carry it. Static
introspection of ``route.dependant.dependencies`` is NOT an acceptable probe:
FastAPI includes sub-routers lazily (``_IncludedRouter``), so a parent router's
``dependencies=`` are applied at match time, not baked into the route objects
at import time, and introspection reports live-gated routes as ungated — a
false negative. A request is the only honest probe.

The paired 200s (entitled, and flag-off) are what stop this file from passing
vacuously: if the gate were mounted wrongly, or the fixture were broken, those
cases would not be 200s either.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware
from w2v_common.auth import clear_license_status_cache
from w2v_common.auth.licenses_read_client import LicensesClientError

from src.api.health import router as health_router
from src.api.routes import router as transcription_router
from src.config import get_settings
from tests.conftest import TEST_TENANT_ID, FakeBackend

SUBJECT = "user-under-test"

#: The cheapest genuinely tenant-scoped route on the gated router: it only
#: reads ``app.state.backend``, which the fake supplies. So a non-200 here can
#: only be the gate — which is the whole point.
GATED_PATH = "/api/models"

#: A probe. Must answer 200 for a revoked tenant: a 402/503 on a liveness probe
#: flaps the launchd unit, and an unlicensed tenant cannot be billed by killing
#: our own worker.
PROBE_PATH = "/health/live"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Record:
    """A licences wire record. Only ``model_dump`` is on the gate's path."""

    def __init__(self, **fields: Any) -> None:
        self._fields = fields

    def model_dump(self) -> dict[str, Any]:
        return dict(self._fields)


class _Detail(_Record):
    """An ACTIVE licence detail (the 200 from ``GET /licenses/active``)."""

    conflict = False


class _FakeLicenses:
    """Stands in for ``LicensesReader``.

    ``active`` None models the authoritative 404 ("no active licence"), which
    sends the gate to the subject inventory to refine none/expired/revoked.
    ``error`` models an upstream failure, which must 503, never allow.
    """

    def __init__(
        self,
        *,
        active: _Detail | None = None,
        inventory: list[_Record] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._active = active
        self._inventory = inventory or []
        self._error = error

    async def get_active_license(self, *, tenant_id: str, app_key: str) -> _Detail | None:
        if self._error is not None:
            raise self._error
        return self._active

    async def list_user_licenses(self, subject_id: str, *, tenant_id: str) -> list[_Record]:
        if self._error is not None:
            raise self._error
        return list(self._inventory)


def _revoked() -> list[_Record]:
    return [_Record(app={"key": "w2v"}, tenant_id=TEST_TENANT_ID, status="revoked")]


def _expired() -> list[_Record]:
    return [
        _Record(
            app={"key": "w2v"},
            tenant_id=TEST_TENANT_ID,
            status="expired",
            expires_at="2020-01-01T00:00:00+00:00",
        )
    ]


def _entitled() -> _Detail:
    return _Detail(
        app={"key": "w2v"},
        tenant_id=TEST_TENANT_ID,
        status="active",
        expires_at="2099-01-01T00:00:00+00:00",
        plan={"code": "pro"},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_cached_verdicts() -> Any:
    """The gate caches 60s per (tenant, app, subject). Two tests in one process
    would otherwise share a verdict and the second would assert nothing."""
    clear_license_status_cache()
    yield
    clear_license_status_cache()


@pytest.fixture(autouse=True)
def _gate_disarmed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default the flag OFF, exactly as it ships. Each test arms it explicitly."""
    monkeypatch.delenv("W2V_LICENSE_GATE_ENABLED", raising=False)


@pytest.fixture
def gate_app() -> FastAPI:
    """The REAL routers, with identity stamped the way AuthMiddleware stamps it."""
    application = FastAPI()

    class _StampIdentity(BaseHTTPMiddleware):
        async def dispatch(self, request: Any, call_next: Any) -> Any:
            request.state.tenant_id = TEST_TENANT_ID
            request.state.subject_id = SUBJECT
            return await call_next(request)

    application.add_middleware(_StampIdentity)
    application.state.backend = FakeBackend()
    application.state.settings = get_settings()
    application.include_router(health_router)
    application.include_router(transcription_router)
    return application


@pytest.fixture
def gate_client(gate_app: FastAPI) -> TestClient:
    return TestClient(gate_app)


def _licenses(monkeypatch: pytest.MonkeyPatch, fake: _FakeLicenses) -> None:
    """Patch the client the gate resolves through, in the module that calls it."""
    monkeypatch.setattr(
        "w2v_common.auth.entitlement.get_licenses_read_client", lambda *a, **k: fake
    )


# ---------------------------------------------------------------------------
# Armed: an unentitled tenant is refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("inventory", "code"),
    [
        ([], "license_none"),
        (_revoked(), "license_revoked"),
        (_expired(), "license_expired"),
    ],
    ids=["none", "revoked", "expired"],
)
def test_armed_gate_402s_an_unentitled_tenant(
    gate_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    inventory: list[_Record],
    code: str,
) -> None:
    """402 with the state-specific code — the tenant's licence is the problem."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=inventory))

    response = gate_client.get(GATED_PATH)

    assert response.status_code == 402, response.text
    assert response.json()["detail"]["error"] == code


def test_armed_gate_402s_the_transcribe_route_too(
    gate_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The expensive route is the reason the gate exists: a revoked tenant must
    be refused BEFORE any upload is read or any decode slot is taken."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = gate_client.post(
        "/api/transcribe",
        files={"file": ("a.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
    )

    assert response.status_code == 402, response.text
    assert response.json()["detail"]["error"] == "license_revoked"


def test_armed_gate_503s_when_entitlement_cannot_be_established(
    gate_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An upstream failure is NOT an allow, and NOT a 402 either: we do not know."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(error=LicensesClientError("licenses unreachable")))

    response = gate_client.get(GATED_PATH)

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["error"] == "license_unavailable"
    assert response.headers["Retry-After"]


# ---------------------------------------------------------------------------
# Armed: an entitled tenant is served (the non-vacuity pair)
# ---------------------------------------------------------------------------


def test_armed_gate_serves_an_entitled_tenant(
    gate_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=_entitled()))

    response = gate_client.get(GATED_PATH)

    assert response.status_code == 200, response.text
    assert response.json()["current"]["backend"] == "fake"


# ---------------------------------------------------------------------------
# Dormant on merge
# ---------------------------------------------------------------------------


def test_gate_is_dormant_while_the_flag_is_unset(
    gate_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Merging this changes NOTHING until gitops arms the flag. A revoked tenant
    is still served — deliberately — so the rollout is a flag flip, not a deploy."""
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = gate_client.get(GATED_PATH)

    assert response.status_code == 200, response.text


def test_gate_is_dormant_when_the_flag_is_explicitly_false(
    gate_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "false")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = gate_client.get(GATED_PATH)

    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# Probes are never gated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/health", "/health/live", "/health/ready", "/healthz"])
def test_probe_is_served_for_a_revoked_tenant(
    gate_client: TestClient, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    """A gated probe path flaps the launchd unit. Probes must stay 200."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = gate_client.get(path)

    assert response.status_code == 200, response.text


def test_the_services_own_probe_paths_are_exempt_by_path() -> None:
    """Belt and braces: even if a future refactor mounted the gate on the health
    router, w2v_common would still refuse to enforce on these paths."""
    from w2v_common.auth import is_license_exempt_path

    for path in ("/health", "/health/live", "/health/ready", "/healthz", "/readyz"):
        assert is_license_exempt_path(path), path
    assert not is_license_exempt_path(GATED_PATH)
    assert not is_license_exempt_path("/api/transcribe")
