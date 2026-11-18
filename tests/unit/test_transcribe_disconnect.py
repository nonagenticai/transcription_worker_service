"""A client disconnect must not leak the decode slot or the temp file.

``asyncio.to_thread`` cancellation only cancels the AWAITING coroutine — the
worker thread keeps running the (non-re-entrant) Whisper decode. If the route
releases the admission slot and unlinks the temp audio file at that
moment, the NEXT request enters the shared model concurrently and the live
decode reads a file that has been deleted underneath it.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from src.api.routes import router as transcription_router
from src.config import Settings, clear_settings_cache
from src.services.backends.base import (
    TranscriptionResult,
    TranscriptSegment,
    WhisperBackend,
)
from tests.conftest import stamp_test_tenant


class _BlockingBackend(WhisperBackend):
    """Parks inside ``transcribe`` until released, recording overlap + file state."""

    name = "blocking"

    def __init__(self) -> None:
        super().__init__(model_size="tiny", device="cpu", compute_type="int8")
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        # Whether the temp audio file still existed when the decode finished.
        self.path_alive_at_exit: list[bool] = []
        self.paths: list[Path] = []

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
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.paths.append(audio_path)
        self.entered.set()
        try:
            self.release.wait(timeout=5.0)
            self.path_alive_at_exit.append(audio_path.exists())
        finally:
            with self._lock:
                self.in_flight -= 1
        return TranscriptionResult(
            language=language or "en",
            duration=1.0,
            segments=[TranscriptSegment(id=0, start=0.0, end=1.0, text="hi")],
            text="hi",
            backend=self.name,
            model=self.model_size,
        )


def _build_app(backend: WhisperBackend, settings: Settings) -> FastAPI:
    app = FastAPI()
    app.state.backend = backend
    app.state.settings = settings
    app.include_router(transcription_router)
    stamp_test_tenant(app)
    return app


def _files() -> dict[str, tuple[str, io.BytesIO, str]]:
    return {"file": ("a.wav", io.BytesIO(b"RIFF....WAVEfmt "), "audio/wav")}


async def _wait_entered(backend: _BlockingBackend) -> None:
    for _ in range(500):
        if backend.entered.is_set():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("request never reached the backend")


async def test_disconnect_does_not_release_the_slot_early() -> None:
    """PRODUCTION CALL SITE: POST /api/transcribe.

    Cancel request #1 while its decode thread is still running, then fire
    request #2. The second decode must NOT overlap the first — the lane slot
    slot belongs to the running thread, not to the (now cancelled) awaiting task.
    """
    clear_settings_cache()
    settings = Settings()
    backend = _BlockingBackend()
    backend.release.clear()
    app = _build_app(backend, settings)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = asyncio.create_task(client.post("/api/transcribe", files=_files()))
        await _wait_entered(backend)

        # Client goes away mid-decode.
        first.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await first
        # Give the event loop a chance to run the route's cancellation path.
        await asyncio.sleep(0.05)

        second = asyncio.create_task(client.post("/api/transcribe", files=_files()))
        # Long enough for a wrongly-released slot to admit the second decode.
        await asyncio.sleep(0.3)

        assert backend.max_in_flight == 1, (
            f"{backend.max_in_flight} overlapping decodes against one shared model: "
            "the disconnect released the lane slot while the decode thread was live"
        )

        backend.release.set()
        response = await second
        assert response.status_code == 200

    # Neither decode may see its temp audio file deleted underneath it.
    assert backend.path_alive_at_exit and all(backend.path_alive_at_exit), (
        f"temp audio file was unlinked while a decode was still running: "
        f"{backend.path_alive_at_exit}"
    )


async def test_disconnect_keeps_temp_file_until_the_decode_finishes() -> None:
    """PRODUCTION CALL SITE: POST /api/transcribe.

    The cancelled request's OWN decode thread must still find its input file.
    """
    clear_settings_cache()
    settings = Settings()
    backend = _BlockingBackend()
    backend.release.clear()
    app = _build_app(backend, settings)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        task = asyncio.create_task(client.post("/api/transcribe", files=_files()))
        await _wait_entered(backend)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.05)
        backend.release.set()
        for _ in range(500):
            if backend.path_alive_at_exit:
                break
            await asyncio.sleep(0.01)

    assert backend.path_alive_at_exit == [True], (
        f"temp audio file was deleted out from under the live decode: {backend.path_alive_at_exit}"
    )
    # ...and once that decode thread ends, the temp file is cleaned up -- the
    # disconnect must not turn into a leaked upload on disk.
    for _ in range(500):
        if not backend.paths[0].exists():
            break
        await asyncio.sleep(0.01)
    assert not backend.paths[0].exists(), f"temp audio file leaked: {backend.paths[0]}"


class _FailingBackend(_BlockingBackend):
    """Records its input path, then fails the decode."""

    def transcribe(
        self,
        audio_path: Path,
        language: str | None,
        word_timestamps: bool,
    ) -> TranscriptionResult:
        self.paths.append(audio_path)
        assert audio_path.exists()
        raise RuntimeError("decoder blew up")


async def test_failed_decode_still_deletes_the_temp_file() -> None:
    clear_settings_cache()
    backend = _FailingBackend()
    app = _build_app(backend, Settings())

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/transcribe", files=_files())

    assert resp.status_code == 500
    assert len(backend.paths) == 1
    assert not backend.paths[0].exists(), f"failed decode leaked {backend.paths[0]}"


async def test_failed_temp_write_does_not_leave_a_partial_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disk full mid-write: the half-written upload is removed, nothing is decoded."""
    import errno
    import tempfile

    import src.api.routes as routes_mod

    created: list[Path] = []
    real_ntf = tempfile.NamedTemporaryFile

    class _FullDisk:
        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self.name = inner.name

        def write(self, data: bytes) -> int:
            self._inner.write(data[:4])
            self._inner.flush()
            raise OSError(errno.ENOSPC, "No space left on device")

        def __enter__(self) -> _FullDisk:
            return self

        def __exit__(self, *exc: object) -> None:
            self._inner.close()

    def _ntf(*args: Any, **kwargs: Any) -> _FullDisk:
        inner = real_ntf(*args, **kwargs)
        created.append(Path(inner.name))
        return _FullDisk(inner)

    monkeypatch.setattr(routes_mod.tempfile, "NamedTemporaryFile", _ntf)
    clear_settings_cache()
    backend = _BlockingBackend()
    backend.release.set()
    app = _build_app(backend, Settings())

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/transcribe", files=_files())

    assert resp.status_code == 500
    assert len(created) == 1
    assert not created[0].exists(), f"partial upload left on disk: {created[0]}"
    assert backend.paths == [], "nothing may be decoded after a failed write"
