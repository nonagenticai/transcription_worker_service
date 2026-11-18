"""Admission control around the single shared, non-re-entrant Whisper model.

The ``transcribe`` lane of the shared ``w2v_common.admission.AdmissionGate``
(1 active / 8 waiting / 30 s) replaced the hand-rolled semaphore. Covers:

- saturation: 20 parallel uploads -> exactly 1 decoding, 8 queued, 11 immediate
  503 ``gpu_busy`` + Retry-After, and /health still answers fast;
- queued requests run strictly one at a time (no overlap in the model);
- a request queued past ``max_wait_s`` gets 503 ``reason=timeout``;
- a client that disconnected while queued is SKIPPED (499), never decoded;
- a concurrent double ``load()`` builds the model exactly once.
"""

from __future__ import annotations

import asyncio
import io
import sys
import threading
import time
import types
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from prometheus_client import REGISTRY
from w2v_common.admission import install_gpu_busy_handler

from src.api.health import router as health_router
from src.api.routes import router as transcription_router
from src.config import Settings, clear_settings_cache
from src.services.admission import TRANSCRIBE_LANE, TRANSCRIBE_LANE_DEFAULTS, build_gate
from src.services.backends.base import (
    TranscriptionResult,
    TranscriptSegment,
    WhisperBackend,
)
from src.services.backends.faster_whisper_backend import FasterWhisperBackend
from tests.conftest import TEST_TENANT_ID, stamp_test_tenant


class InstrumentedBackend(WhisperBackend):
    """Backend that records overlap and can be parked inside ``transcribe``."""

    name = "instrumented"

    def __init__(self) -> None:
        super().__init__(model_size="tiny", device="cpu", compute_type="int8")
        self._counter_lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()

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
        with self._counter_lock:
            self.calls += 1
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.entered.set()
        try:
            self.release.wait(timeout=10.0)
            threading.Event().wait(0.005)
        finally:
            with self._counter_lock:
                self.in_flight -= 1
        return TranscriptionResult(
            language=language or "en",
            duration=1.0,
            segments=[TranscriptSegment(id=0, start=0.0, end=1.0, text="hi")],
            text="hi",
            backend=self.name,
            model=self.model_size,
        )


def _build_app(
    backend: WhisperBackend,
    environ: dict[str, str] | None = None,
) -> FastAPI:
    clear_settings_cache()
    app = FastAPI()
    app.state.backend = backend
    app.state.settings = Settings()
    if environ is not None:
        from w2v_common.admission import AdmissionGate

        app.state.gpu_gate = AdmissionGate(
            {TRANSCRIBE_LANE: TRANSCRIBE_LANE_DEFAULTS}, environ=environ
        )
    install_gpu_busy_handler(app)
    app.include_router(health_router)
    app.include_router(transcription_router)
    stamp_test_tenant(app)
    return app


def _files() -> dict[str, tuple[str, io.BytesIO, str]]:
    return {"file": ("a.wav", io.BytesIO(b"RIFF....WAVEfmt "), "audio/wav")}


def _rejected(reason: str) -> float:
    value = REGISTRY.get_sample_value(
        "gpu_admission_rejected_total", {"lane": TRANSCRIBE_LANE, "reason": reason}
    )
    return float(value) if value is not None else 0.0


async def _until(predicate: Any, what: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"timed out waiting for: {what}")


def test_transcribe_lane_code_defaults() -> None:
    """Issue #113: lane ``transcribe`` at 1 active / 8 waiting / 30 s."""
    lane = build_gate().lane(TRANSCRIBE_LANE)
    assert (lane.config.max_active, lane.config.max_waiting, lane.config.max_wait_s) == (
        1,
        8,
        30.0,
    ), "max_active MUST stay 1 -- the shared Whisper model is not re-entrant"


def test_transcribe_lane_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GPU_LANE_TRANSCRIBE_MAX_WAITING", "64")
    monkeypatch.setenv("GPU_LANE_TRANSCRIBE_MAX_WAIT_S", "120")
    cfg = build_gate().lane(TRANSCRIBE_LANE).config
    assert (cfg.max_active, cfg.max_waiting, cfg.max_wait_s) == (1, 64, 120.0)


async def test_saturation_admits_1_queues_8_and_sheds_the_rest() -> None:
    """20 parallel uploads against a parked model: 1 decoding, 8 queued, 11 x 503.

    /health must keep answering in < 200 ms while the lane is saturated -- the
    decode runs in the lane executor, never on the event loop.
    """
    backend = InstrumentedBackend()
    backend.release.clear()
    app = _build_app(backend)
    full_before = _rejected("full")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        tasks = [
            asyncio.create_task(client.post("/api/transcribe", files=_files())) for _ in range(20)
        ]
        await _until(lambda: sum(t.done() for t in tasks) == 11, "11 immediate rejections")
        # The admitted request's thread may still be spilling its temp file.
        await _until(backend.entered.is_set, "admitted request reaches the backend")
        lane = app.state.gpu_gate.lane(TRANSCRIBE_LANE)
        assert lane.active == 1
        assert lane.waiting == 8
        assert backend.in_flight == 1

        rejected = [t.result() for t in tasks if t.done()]
        assert [r.status_code for r in rejected] == [503] * 11
        for r in rejected:
            body = r.json()
            assert body["error"] == "gpu_busy"
            assert body["lane"] == TRANSCRIBE_LANE
            assert int(r.headers["Retry-After"]) == body["retry_after"]
            assert 1 <= body["retry_after"] <= 120
        assert _rejected("full") == full_before + 11

        t0 = time.monotonic()
        health = await client.get("/health")
        elapsed = time.monotonic() - t0
        assert health.status_code == 200
        assert elapsed < 0.2, f"/health took {elapsed * 1000:.0f} ms under saturation"

        backend.release.set()
        served = await asyncio.gather(*(t for t in tasks if not t.done()))

    assert [r.status_code for r in served] == [200] * 9
    assert backend.calls == 9
    assert backend.max_in_flight == 1, (
        f"{backend.max_in_flight} overlapping decodes against one shared model"
    )


async def test_queued_past_max_wait_gets_timeout_503() -> None:
    backend = InstrumentedBackend()
    backend.release.clear()
    app = _build_app(backend, environ={"GPU_LANE_TRANSCRIBE_MAX_WAIT_S": "0.05"})
    timeout_before = _rejected("timeout")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = asyncio.create_task(client.post("/api/transcribe", files=_files()))
        await _until(backend.entered.is_set, "first request reaches the backend")

        late = await client.post("/api/transcribe", files=_files())
        assert late.status_code == 503
        assert late.json()["error"] == "gpu_busy"
        assert "Retry-After" in late.headers
        assert _rejected("timeout") == timeout_before + 1

        backend.release.set()
        assert (await first).status_code == 200
    assert backend.calls == 1


async def test_after_gate_shutdown_requests_get_503_not_500() -> None:
    backend = InstrumentedBackend()
    app = _build_app(backend, environ={})
    assert await app.state.gpu_gate.shutdown(timeout=1.0) is True

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/transcribe", files=_files())
    assert resp.status_code == 503
    assert "shutting down" in resp.text
    assert backend.calls == 0


def _multipart() -> tuple[bytes, bytes]:
    boundary = b"testboundary"
    body = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="a.wav"\r\n'
        b"Content-Type: audio/wav\r\n\r\n"
        b"RIFF....WAVEfmt \r\n"
        b"--" + boundary + b"--\r\n"
    )
    return body, b"multipart/form-data; boundary=" + boundary


_LICENSE_TOKEN = "abc.def.ghi"  # pragma: allowlist secret
_AUTH_HEADER = {"X-Tenant-Id": f"Bearer {_LICENSE_TOKEN}"}


def _real_app(monkeypatch: pytest.MonkeyPatch, backend: WhisperBackend) -> FastAPI:
    """The PRODUCTION ``src.asgi.app`` -- its real middleware stack, auth ON.

    CORS + AuthMiddleware + RequestIdMiddleware (both BaseHTTPMiddleware) +
    DisconnectProbeMiddleware, exactly as asgi.py wires them. Only the licence
    introspection call and the Whisper backend are stubbed; the lifespan is not
    run (httpx's ASGITransport never sends lifespan), so state is set by hand.
    """
    import w2v_common.auth.middleware as w2v_auth

    async def _introspect(token: str) -> dict[str, Any]:
        assert token == _LICENSE_TOKEN
        return {
            "active": True,
            "tenant_id": TEST_TENANT_ID,
            "subject_id": "subj-1",
            "license_id": "lic-1",
            "roles": ["user"],
        }

    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setattr(w2v_auth, "_introspect_license", _introspect)
    w2v_auth._get_license_cache().clear()
    clear_settings_cache()
    for name in list(sys.modules):
        if name == "src.asgi" or name.startswith("src.asgi."):
            del sys.modules[name]
    import src.asgi as asgi_mod

    app = asgi_mod.app
    assert any(m.cls is w2v_auth.AuthMiddleware for m in app.user_middleware), (
        "real stack must include the BaseHTTPMiddleware AuthMiddleware layer"
    )
    app.state.backend = backend
    app.state.settings = asgi_mod.settings
    app.state.metering_redis = None
    return app


async def test_client_gone_while_queued_is_skipped_through_the_real_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A queued request whose client disconnected is never decoded (499, no work).

    Driven through the production app, whose BaseHTTPMiddleware layers hide
    ``http.disconnect`` from ``Request.is_disconnected()``. Only the outermost
    ``DisconnectProbeMiddleware`` makes the gate's skip fire; without it this
    request is decoded and answered 200.
    """
    backend = InstrumentedBackend()
    backend.release.clear()
    app = _real_app(monkeypatch, backend)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = asyncio.create_task(
            client.post("/api/transcribe", files=_files(), headers=_AUTH_HEADER)
        )
        await _until(backend.entered.is_set, "first request reaches the backend")

        # Drive the second request at the raw ASGI layer so the client can
        # report ``http.disconnect`` while it is queued.
        body, ctype = _multipart()
        gone = threading.Event()
        body_sent = False
        never = asyncio.Event()
        sent: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            if gone.is_set():
                return {"type": "http.disconnect"}
            await never.wait()
            return {"type": "http.disconnect"}

        async def send(message: MutableMapping[str, Any]) -> None:
            sent.append(dict(message))

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/transcribe",
            "raw_path": b"/api/transcribe",
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"host", b"test"),
                (b"content-type", ctype),
                (b"x-tenant-id", f"Bearer {_LICENSE_TOKEN}".encode()),
            ],
            "client": ("127.0.0.1", 1),
            "server": ("test", 80),
        }
        second = asyncio.create_task(app(scope, receive, send))
        await _until(
            lambda: (
                getattr(app.state, "gpu_gate", None) is not None
                and app.state.gpu_gate.lane(TRANSCRIBE_LANE).waiting == 1
            ),
            "second request queued",
        )

        gone.set()
        backend.release.set()
        await asyncio.wait_for(second, timeout=5.0)
        first_resp = await first
        assert first_resp.status_code == 200, first_resp.text

    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert starts and starts[0]["status"] == 499, starts
    assert backend.calls == 1, "the disconnected client's job must be skipped, not decoded"


def test_concurrent_load_builds_the_model_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two threads hitting a cold backend must build ONE model, not two."""
    built: list[float] = []
    build_lock = threading.Lock()

    class _SlowModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            threading.Event().wait(0.05)  # widen the check-then-set window
            with build_lock:
                built.append(1.0)

    mod = types.ModuleType("faster_whisper")
    setattr(mod, "WhisperModel", _SlowModel)  # noqa: B010
    monkeypatch.setitem(sys.modules, "faster_whisper", mod)

    backend = FasterWhisperBackend(model_size="tiny", device="cpu", compute_type="int8")
    threads = [threading.Thread(target=backend.load) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert len(built) == 1, f"model built {len(built)} times — load() is not guarded"
    assert backend.is_loaded
