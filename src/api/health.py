"""Health endpoints for transcription_worker_service."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, Response

router = APIRouter(tags=["health"])


def _readiness(request: Request) -> tuple[bool, str]:
    """Decide readiness and return ``(ready, reason)``.

    Readiness must be FAIL-CLOSED on the load path: the startup preload
    swallows ``BackendUnavailableError`` and the transcribe route records the
    same failure, so a service whose model can never load used to advertise
    ``ready: true`` forever while 500ing/503ing every transcription — traffic
    kept arriving and no supervisor ever restarted it.

    It is deliberately NOT gated on ``is_loaded`` in the default lazy mode
    (``load_on_startup=false``): the model only loads on the first request, so
    requiring ``loaded`` would make the service never ready and never receive
    that first request. With ``load_on_startup=true`` an unloaded model *is*
    a failure, so it counts there.
    """
    backend = getattr(request.app.state, "backend", None)
    if backend is None:
        return False, "backend not initialised"

    load_error: str | None = getattr(request.app.state, "backend_load_error", None)
    if load_error:
        return False, f"backend failed to load: {load_error}"

    settings = getattr(request.app.state, "settings", None)
    eager = bool(getattr(getattr(settings, "transcription", None), "load_on_startup", False))
    if eager and not backend.is_loaded:
        return False, "backend preload configured but model is not loaded"

    return True, "ok"


def _backend_info(request: Request) -> dict[str, Any]:
    backend = getattr(request.app.state, "backend", None)
    if backend is None:
        return {"backend": "unknown", "model": "unknown", "device": "unknown", "loaded": False}
    return {
        "backend": backend.name,
        "model": backend.model_size,
        "device": backend.device,
        "compute_type": backend.compute_type,
        "loaded": bool(backend.is_loaded),
    }


@router.get(
    "/health",
    operation_id="health_check",
    summary="Combined health probe with backend/model info",
)
async def health(request: Request) -> dict[str, Any]:
    """Combined health check exposing backend/model info."""
    return {"status": "healthy", **_backend_info(request)}


@router.get(
    "/health/live",
    operation_id="health_live",
    summary="Kubernetes liveness probe",
)
async def health_live() -> dict[str, str]:
    """Kubernetes liveness probe."""
    return {"status": "ok"}


@router.get(
    "/health/ready",
    operation_id="health_ready",
    summary="Kubernetes readiness probe",
)
async def health_ready(request: Request, response: Response) -> dict[str, Any]:
    """Readiness probe — 503 when the backend cannot serve transcriptions.

    Readiness ONLY. ``/health`` and ``/health/live`` stay unconditionally 200
    so a load failure sheds traffic instead of putting the process into a
    restart loop.
    """
    ready, reason = _readiness(request)
    if not ready:
        response.status_code = 503
    info = _backend_info(request)
    return {
        "status": "healthy" if ready else "degraded",
        "ready": ready,
        "reason": reason,
        **info,
    }


@router.get(
    "/healthz",
    operation_id="health.live",
    summary="Modern K8s liveness probe path (alias of /health/live)",
)
async def healthz() -> dict[str, str]:
    """Modern K8s liveness probe path (alias of /health/live)."""
    return await health_live()


@router.get(
    "/readyz",
    operation_id="health.ready",
    summary="Modern K8s readiness probe path (alias of /health/ready)",
)
async def readyz(request: Request, response: Response) -> dict[str, Any]:
    """Modern K8s readiness probe path (alias of /health/ready)."""
    return await health_ready(request, response)
