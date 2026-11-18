"""The Whisper native extension must be imported on the MAIN thread.

``src/api/routes.py`` reaches the backend through ``asyncio.to_thread``, and
``WhisperBackend.load()`` imports ``faster_whisper`` (→ ctranslate2) or
``whisper`` (→ torch) lazily. With ``load_on_startup`` false — which is the
default AND what the live launchd unit on mini1 runs with — that makes the very
first import of a native extension happen on an arbitrary executor worker
thread, on the first request.

These tests pin the fix: the ASGI lifespan pre-imports the selected backend's
native module while still on the main thread.

Mechanism note: this wraps ``builtins.__import__`` rather than patching
``sys.modules``. Patching ``sys.modules`` evicts already-initialised C
extensions, which is how a sibling repo earned a SIGSEGV. Wrapping
``__import__`` observes the import statement without touching module state, and
lets us short-circuit the real (multi-second) native import so the test stays
hermetic and fast.
"""

from __future__ import annotations

import asyncio
import builtins
import sys
import threading
import types
from collections.abc import Sequence
from typing import Any

import pytest

from src.config import clear_settings_cache


def _reimport_asgi() -> Any:
    """Re-execute ``src.asgi`` so it picks up the patched environment."""
    for name in list(sys.modules):
        if name == "src.asgi" or name.startswith("src.asgi."):
            del sys.modules[name]
    import src.asgi as asgi_mod

    return asgi_mod


def _record_import_threads(monkeypatch: pytest.MonkeyPatch, target: str) -> Sequence[str]:
    """Record the thread name of every top-level ``import <target>``.

    The target import is short-circuited to a throwaway module so no real
    native extension is loaded; every other import is delegated untouched.
    """
    seen: list[str] = []
    real_import = builtins.__import__

    def recording_import(
        name: str,
        globals: Any = None,  # noqa: A002 - matches builtins.__import__ signature
        locals: Any = None,  # noqa: A002 - matches builtins.__import__ signature
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        if level == 0 and name == target:
            seen.append(threading.current_thread().name)
            return types.ModuleType(target)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", recording_import)
    return seen


@pytest.mark.parametrize(
    ("backend", "native_module"),
    [("faster", "faster_whisper"), ("openai", "whisper")],
)
def test_native_module_is_imported_on_the_main_thread(
    monkeypatch: pytest.MonkeyPatch, backend: str, native_module: str
) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "false")
    monkeypatch.setenv("TRANSCRIPTION_BACKEND", backend)
    # `cpu`, not `auto`: `auto` would import torch in backend_selector and give
    # the torch-backed backend accidental main-thread cover, hiding the defect.
    monkeypatch.setenv("TRANSCRIPTION_DEVICE", "cpu")
    monkeypatch.setenv("TRANSCRIPTION_MODEL_SIZE", "tiny")
    # The production default, and what the live unit runs: NO eager load.
    monkeypatch.setenv("TRANSCRIPTION_LOAD_ON_STARTUP", "false")
    clear_settings_cache()

    asgi_mod = _reimport_asgi()
    assert asgi_mod.settings.transcription.load_on_startup is False

    seen = _record_import_threads(monkeypatch, native_module)

    assert threading.current_thread() is threading.main_thread(), (
        "test harness precondition: this test body must run on the main thread"
    )

    async def _run_lifespan() -> None:
        async with asgi_mod.lifespan(asgi_mod.app):
            pass

    asyncio.run(_run_lifespan())

    assert seen, (
        f"{native_module} was never imported during ASGI startup for "
        f"backend={backend}: the native extension would first be imported "
        f"inside asyncio.to_thread, on an executor worker thread"
    )
    assert set(seen) == {"MainThread"}, (
        f"{native_module} was imported off the main thread for backend={backend}: {seen}"
    )


def test_preimport_degrades_when_the_native_module_is_broken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken native module must not turn startup into a hard failure."""
    from src.services.backends.faster_whisper_backend import FasterWhisperBackend

    backend = FasterWhisperBackend(model_size="tiny", device="cpu", compute_type=None)

    def boom(self: FasterWhisperBackend) -> None:
        raise OSError("simulated broken ctranslate2 dylib")

    monkeypatch.setattr(FasterWhisperBackend, "_import_native", boom)

    backend.preimport()  # must not raise
    assert backend.is_loaded is False
