"""ASGI entry point for transcription_worker_service."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from w2v_common.admission import DisconnectProbeMiddleware, install_gpu_busy_handler
from w2v_common.asgi import register_error_envelope_handlers

from src.api.health import router as health_router
from src.api.routes import router as transcription_router
from src.config import Settings, get_settings
from src.services.admission import build_gate
from src.services.backend_selector import build_backend
from src.services.metering import close_metering_redis, init_metering_redis

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


_setup_logging()


settings: Settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise the Whisper backend (lazy load unless configured eager)."""
    logger.info("Starting transcription_worker_service on port %d", settings.api.port)
    app.state.settings = settings

    backend = build_backend(settings.transcription)
    app.state.backend = backend

    # Import the native extension (ctranslate2 / torch) HERE, on the main
    # thread. Everything downstream of this reaches the backend through
    # ``asyncio.to_thread`` (routes.py), so the first import would otherwise
    # land on an executor worker thread. ``preimport`` never raises and never
    # loads weights — see WhisperBackend.preimport.
    backend.preimport()

    logger.info(
        "Selected backend=%s model=%s device=%s",
        backend.name,
        backend.model_size,
        backend.device,
    )

    # GPU admission gate (lane "transcribe", see src/services/admission.py):
    # built here so limits + metrics exist before the first request.
    app.state.gpu_gate = build_gate()

    # Readiness is fail-closed on this: a swallowed preload failure used to
    # leave the service advertising ready:true while every transcribe 503'd.
    app.state.backend_load_error = None
    if settings.transcription.load_on_startup:
        try:
            backend.load()
        except (OSError, RuntimeError, ImportError) as exc:
            logger.error("Failed to preload backend: %s", exc)
            app.state.backend_load_error = str(exc)

    app.state.metering_redis = await init_metering_redis(settings.metering.redis_url)

    yield

    logger.info("Shutting down transcription_worker_service")
    # Stop admitting and give the in-flight decode a bounded chance to finish.
    await app.state.gpu_gate.shutdown(timeout=30.0)
    await close_metering_redis(app.state.metering_redis)


# ============================================================================
# Public-endpoint policy (kept in sync with w2v_common.auth.middleware)
# ============================================================================
# Used only by custom_openapi() below to decide which paths advertise the
# Bearer security requirement in the schema. The actual auth enforcement is
# done by w2v_common.auth.middleware.AuthMiddleware (see asgi.py wiring).
_PUBLIC_PATHS: frozenset[str] = frozenset(
    {
        "/",
        "/health",
        "/health/live",
        "/health/ready",
        "/healthz",
        "/readyz",
        "/ready",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/favicon.ico",
        "/metrics",
    }
)
_PUBLIC_PREFIXES: tuple[str, ...] = ("/docs", "/redoc", "/health/")


# ============================================================================
# Create App
# ============================================================================

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Stateless Whisper transcription microservice",
    lifespan=lifespan,
    redirect_slashes=False,
    swagger_ui_parameters={"persistAuthorization": True},
    # Fallback unique-id generator. Every route still sets `operation_id=`
    # explicitly per OPERATION_ID_PLAN.md; this exists so a future route
    # that forgets the kwarg degrades to `<tag>_<func>` instead of FastAPI's
    # default `<func>_<path>_<method>` (garbage in generated clients).
    generate_unique_id_function=lambda route: (
        f"{route.tags[0] if route.tags else 'default'}_{route.name}"
    ),
    openapi_tags=[
        {"name": "health", "description": "Liveness, readiness, dependency probes."},
        {
            "name": "transcription",
            "description": "Whisper transcription endpoints and backend model metadata.",
        },
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=settings.api.cors_allow_credentials,
    allow_methods=settings.api.cors_allow_methods,
    allow_headers=settings.api.cors_allow_headers,
)

if settings.auth.enabled:
    from w2v_common.auth.middleware import AuthMiddleware

    app.add_middleware(AuthMiddleware)
else:
    logger.warning("transcription_worker_service running without auth (development mode)")


# Request ID middleware — registered last so it is the OUTERMOST layer and runs first
# inbound, binding the request_id ContextVar before auth/handlers execute.
from w2v_common.observability.request_context import RequestIdLogFilter  # noqa: E402

from src.middleware.request_id import RequestIdMiddleware  # noqa: E402

app.add_middleware(RequestIdMiddleware)

# Disconnect probe — MUST stay the LAST add_middleware call (outermost, pure ASGI).
# The BaseHTTPMiddleware layers above (RequestId, Auth) hide http.disconnect from
# Request.is_disconnected(); without this the gate's skip-if-client-gone never fires.
app.add_middleware(DisconnectProbeMiddleware)
logging.getLogger().addFilter(RequestIdLogFilter())


app.include_router(health_router)
app.include_router(transcription_router)


@app.get(
    "/metrics",
    operation_id="prometheus_metrics",
    tags=["health"],
    summary="Prometheus metrics exposition",
)
async def metrics() -> Response:
    """Expose Prometheus metrics in the text exposition format."""
    from src.services.metrics import render_metrics

    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)


@app.get(
    "/",
    operation_id="service_root",
    tags=["health"],
    summary="Service banner + endpoint index",
)
async def root() -> dict[str, Any]:
    """Return service metadata and endpoint map."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "endpoints": {
            "transcribe": "/api/transcribe",
            "models": "/api/models",
            "health": "/health",
        },
    }


# ============================================================================
# OpenAPI customisation — BearerAuth scheme + TenantIdHeader parameter
# ============================================================================
#
# Mirrors the analysis_service / kag / worker_management / nlp_service pattern:
# every protected operation advertises ``security: [{BearerAuth: []}]`` and a
# reusable ``TenantIdHeader`` parameter so Swagger UI exposes the license-token
# transport correctly. The actual enforcement lives in
# ``w2v_common.auth.middleware.AuthMiddleware``.
def custom_openapi() -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=app.openapi_tags,
    )

    components = schema.setdefault("components", {})
    components.setdefault("securitySchemes", {})["BearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "License",
        "description": (
            "License token. Transport: `X-Tenant-Id: Bearer <token>` "
            "(not the Authorization header — that is reserved for caller-side auth)."
        ),
    }
    components.setdefault("parameters", {})["TenantIdHeader"] = {
        "name": "X-Tenant-Id",
        "in": "header",
        "required": False,
        "description": "License token transport: `Bearer <token>`.",
        "schema": {"type": "string"},
    }

    for path, item in schema.get("paths", {}).items():
        clean = path.rstrip("/") or "/"
        is_public = clean in _PUBLIC_PATHS or any(clean.startswith(p) for p in _PUBLIC_PREFIXES)
        for operation in item.values():
            if not isinstance(operation, dict):
                continue
            if is_public:
                operation.pop("security", None)
            else:
                operation.setdefault("security", [{"BearerAuth": []}])
            params = operation.setdefault("parameters", [])
            if not any(
                p.get("name", "").lower() == "x-tenant-id" and p.get("in") == "header"
                for p in params
            ):
                params.append({"$ref": "#/components/parameters/TenantIdHeader"})

    app.openapi_schema = schema
    return app.openapi_schema


object.__setattr__(app, "openapi", custom_openapi)
# ============================================================================
# Error Handlers — canonical fleet error envelope (shape #2), from w2v_common
# ============================================================================
register_error_envelope_handlers(app, debug=settings.api.debug)
# GpuBusy -> 503 + Retry-After {"error":"gpu_busy",...}; ClientGone -> bodyless 499.
install_gpu_busy_handler(app)


def main() -> None:
    """Run the ASGI app under uvicorn."""
    import uvicorn

    uvicorn.run(
        "src.asgi:app",
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.logging.level.lower(),
        reload=settings.api.reload,
    )


if __name__ == "__main__":
    main()
