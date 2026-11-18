# Transcription Worker Service

Stateless Whisper microservice that turns uploaded audio or video into text, timestamped segments, and word-level timings. Port 6318.

## What Transcription Worker Service Does

When `video_analyzer_service` reaches the audio step of a CPU-side video analysis, it POSTs the extracted audio to this service over the mesh. The service accepts a multipart upload of up to 200 MiB, writes it to a temporary file, and hands it to a Whisper backend for transcription. Nothing about the request is persisted — the temp file is deleted in a `finally` block and the response is the only output.

Transcription runs through one of two swappable backends selected by an environment switch: `faster-whisper` (the default, a CTranslate2 reimplementation that is roughly 4x faster on CPU) or the reference `openai-whisper`. Device selection auto-detects CUDA when `TRANSCRIPTION_DEVICE=auto` and falls back to CPU when no GPU is present, at which point the default model size drops from `large-v3` (GPU) to `base` (CPU) and the compute type moves from `float16` to `int8`. The heavy `backend.transcribe()` call is dispatched to a worker thread via `asyncio.to_thread` so the event loop stays responsive under concurrent requests.

The response carries the detected language, audio duration, full transcript text, and a list of segments; each segment includes per-word start/end timestamps and confidence probabilities, plus optional SRT subtitle text when the caller sets `return_srt=true`. The caller (`video_analyzer_service`) consumes these segments to build captions and drive downstream vision-plus-language analysis.

The service is stateless — there is no database, no asyncpg, and no SQLite anywhere in the tree; CI runs with `has_postgres: false`. The single optional side effect is a fire-and-forget cost event emitted to a metering Redis stream on each transcription (encoding audio-seconds as input tokens and transcript characters as output tokens); this no-ops silently when the metering Redis URL is unset.

Unlike most of the w2v monorepo, this service does not run as a Kubernetes pod. It runs as a macOS **launchd** unit on an Apple Silicon host, where `uvicorn` runs directly against the on-device virtualenv. There is no production container image. The config default listen port is 6318 (see Known Issues).

## Features

- **Dual Whisper backends** — `faster-whisper` and `openai-whisper` behind one `TRANSCRIPTION_BACKEND` switch so operators trade raw speed (CTranslate2, ~4x on CPU) against reference fidelity without any code change
- **GPU-with-CPU-fallback** — `TRANSCRIPTION_DEVICE=auto` probes CUDA and gracefully drops to CPU (recording a `cuda_fallback_total` metric) so the same build runs on GPU hosts and on the Apple-silicon launchd pool
- **Adaptive model defaults** — model size auto-selects `large-v3` on GPU and `base` on CPU when unset, so each device gets a sensible accuracy/latency balance out of the box
- **Word-level timestamps** — every segment carries per-word start/end and probability so callers can build precise captions and align text to audio
- **Optional SRT output** — a single `return_srt` form flag emits ready-to-use subtitle text so the caller does not have to reformat segments itself
- **Non-blocking transcription** — the CPU/GPU-bound decode runs in a worker thread via `asyncio.to_thread` so concurrent uploads do not stall the event loop
- **Upload guardrails** — empty payloads are rejected (400) and uploads above `max_audio_bytes` (200 MiB) are rejected (413) so a bad request cannot exhaust host memory
- **Metering cost events** — per-transcription usage is streamed to a metering Redis stream (audio-seconds encoding) so billing sees ASR work, degrading silently when metering is disabled
- **License-token auth** — `w2v_common` `AuthMiddleware` enforces `X-Tenant-Id: Bearer <token>` so every transcription is tenant-attributed
- **Prometheus metrics** — a public `/metrics` endpoint exposes counters (including CUDA fallbacks) so the launchd unit is observable without a sidecar

## Authentication

Inbound auth uses the w2v license-token pattern. Callers present the token as `X-Tenant-Id: Bearer <license_token>` — **not** the `Authorization` header, which is reserved for Keycloak JWTs from the SPA. Enforcement is `w2v_common.auth.middleware.AuthMiddleware`, mounted only when `auth.enabled` is true (the default); in development it can be disabled and the app logs a warning. The middleware introspects the token against the licenses service and caches the result for 5 minutes (`cache_ttl=300`).

| Priority | Method | Details |
|----------|--------|---------|
| 1 | License token | `X-Tenant-Id: Bearer <token>` introspected against the licenses service, cached 300s |
| 2 | Trusted network | Synthetic in-cluster identity for trusted CIDRs — **off by default** (`trust_incluster_identity=false`, fail-closed) |

Public paths are exempt from auth: `/`, `/health`, `/health/live`, `/health/ready`, `/healthz`, `/readyz`, `/ready`, `/metrics`, `/docs`, `/redoc`, `/openapi.json`, `/favicon.ico` (plus the `/docs`, `/redoc`, and `/health/` prefixes).

## Quick Start

```bash
uv sync
uv run uvicorn src.asgi:app --host 0.0.0.0 --port 6318
uv run pytest tests/ -v --cov=src
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
```

## API Endpoints

### Transcription `/api/transcribe`, `/api/models`

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/transcribe` | Transcribe a multipart audio upload; returns text, segments, word timings, optional SRT |
| GET | `/api/models` | List available Whisper model sizes and the currently loaded model info |

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service banner and endpoint index |
| GET | `/health` | Combined health probe with backend/model/device info |
| GET | `/health/live` | Liveness probe |
| GET | `/health/ready` | Readiness probe (backend info) |
| GET | `/healthz` | Liveness alias |
| GET | `/readyz` | Readiness alias |
| GET | `/metrics` | Prometheus metrics exposition |

## Configuration

All configuration via environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Listen host |
| `API_PORT` | `6318` | Listen port (base settings reject `8000`) |
| `AUTH_ENABLED` | `true` | Toggle `AuthMiddleware` (disabled only in dev) |
| `AUTH_CACHE_TTL` | `300` | License introspection cache TTL (seconds) |
| `AUTH_TRUST_INCLUSTER_IDENTITY` | `false` | Trusted-CIDR synthetic-identity bypass (fail-closed by default) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `TRANSCRIPTION_BACKEND` | `faster` | Backend: `faster` (faster-whisper) or `openai` (openai-whisper) |
| `TRANSCRIPTION_MODEL_SIZE` | `None` (auto) | `tiny`/`base`/`small`/`medium`/`large-v3`; auto = `base` (CPU) / `large-v3` (GPU) |
| `TRANSCRIPTION_LANGUAGE` | `None` | ISO-639-1 language code; `None` = auto-detect |
| `TRANSCRIPTION_DEVICE` | `auto` | `auto` / `cpu` / `cuda` (auto probes CUDA) |
| `TRANSCRIPTION_COMPUTE_TYPE` | `None` (auto) | faster-whisper precision: `int8` (CPU) / `float16` (GPU) |
| `TRANSCRIPTION_WORD_TIMESTAMPS` | `true` | Default word-timestamp behavior when client omits the flag |
| `TRANSCRIPTION_MAX_AUDIO_BYTES` | `209715200` | Max accepted upload size (200 MiB) |
| `TRANSCRIPTION_LOAD_ON_STARTUP` | `false` | Preload the model at boot (slower start, faster first request) |
| `METERING_REDIS_URL` | `""` | Metering Redis stream URL; empty disables cost-event emission |

## Architecture

```
src/
├── asgi.py                              # FastAPI app, lifespan, CORS + auth + request-id middleware, /, /metrics, OpenAPI
├── config.py                            # Pydantic settings (API/Auth/Logging/Transcription/Metering), port 6318
├── api/
│   ├── health.py                        # /health, /health/live, /health/ready, /healthz, /readyz
│   ├── routes.py                        # /api/transcribe + /api/models, cost-event emission
│   └── schemas.py                       # Transcription request/response DTOs (segments, words, SRT)
├── middleware/
│   └── request_id.py                    # Binds request_id ContextVar (outermost middleware)
└── services/
    ├── backend_selector.py              # Device detection + backend instantiation (no model load)
    ├── metering.py                      # Fire-and-forget cost events to metering Redis stream
    ├── metrics.py                       # Prometheus registry + counters (e.g. cuda_fallback_total)
    ├── srt_formatter.py                 # Segment list → SRT subtitle text
    └── backends/
        ├── base.py                      # WhisperBackend protocol, TranscriptionResult, BackendUnavailableError
        ├── faster_whisper_backend.py    # faster-whisper (CTranslate2) backend
        └── openai_whisper_backend.py    # openai-whisper reference backend
```

## Testing

```bash
uv run pytest tests/ -v --cov=src          # All tests with coverage
uv run pytest tests/unit/ -v               # Unit tests (backends, routes, health, config, metering, metrics)
uv run ruff check src/ tests/              # Lint
uv run ruff format --check src/ tests/     # Format check
```

Tests live in `tests/` (auth-middleware and OpenAPI tests) and `tests/unit/` (backends, routes, health, config, metering, metrics, SRT, request-id), with shared fixtures under `tests/conftest.py` and `tests/fixtures/`. Coverage threshold: 95% (`fail_under = 95` in pyproject.toml). Run `./scripts/ci-local.sh` before pushing — it mirrors the CI Lint+Unit job step-for-step.

## CI/CD

`.github/workflows/ci-public.yml` is self-contained and runs on GitHub-hosted runners: a betterleaks secret scan, the `no-private-refs` guard, then `ruff check`, `ruff format --check` and `pytest`. No image is published from this repository.


There is **no Docker image build/push and no Flux rollout**. This is a launchd service, so lint/type/test pass gates the merge but updates only reach the live unit on the next on-device redeploy. The `torch` CVE-2025-3000 has no upstream fix and is owner-ignored via the workflow's `pip_audit_ignores: "--ignore-vuln CVE-2025-3000"`.

## Docker

**Local / on-device development only — not the deployment path.** `Dockerfile` exists so the image can be built and exercised locally, but production runs `uvicorn` against the on-device venv under launchd, not a container.

```bash
docker build -t transcription-worker-service .
docker run -p 6318:6318 transcription-worker-service
docker run -p 6318:6318 -e LOG_LEVEL=DEBUG -e TRANSCRIPTION_BACKEND=openai transcription-worker-service
```

## Dependencies

- **video_analyzer_service** (port 6317) — the caller; POSTs extracted audio to `asr.test:9201` during CPU-side video analysis
- **licenses service** — validates `X-Tenant-Id` license tokens via introspection (through `w2v_common` auth)
- **metering Redis** — optional cost-event stream; disabled silently when `METERING_REDIS_URL` is empty
- **faster-whisper** — default CTranslate2 Whisper backend
- **openai-whisper** — reference Whisper backend
- **torch** — GPU/CPU tensor runtime and CUDA availability probe
- **ffmpeg** — audio/video decoding (on-device dependency exercised by the launchd pool)
- **FastAPI** — web framework
- **Uvicorn** — ASGI server
- **Pydantic v2 / pydantic-settings** — request/response validation and settings
- **python-multipart** — multipart upload parsing
- **prometheus-client** — metrics exposition
- **redis** — metering stream client
- **w2v_common** — shared auth middleware, settings base classes, error envelope, request-context

## Known Issues

- **Config port 6318 vs live mesh port 9201** — `APISettings.port` defaults to 6318, but the live launchd unit is reached in the mesh at `asr.test:9201`. The gap is config default vs the launchd unit's actual listen address; it is intentional, not a bug, but it trips anyone assuming the config port is the mesh port.
- **Health handlers labelled "Kubernetes probe" though the unit is launchd** — `src/api/health.py` names and docstrings say "Kubernetes liveness/readiness probe", inherited from the fleet template. This service does not run in Kubernetes — there is no pod, HPA, PDB, or NetworkPolicy — so the labels are generic and slightly misleading. The endpoints work identically regardless.
- **No production container** — `Dockerfile` is for local/on-device builds only; the deployment path is the macOS launchd unit and CI never builds or pushes an image.

## License

Apache 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Licensing

This service contains no licence enforcement of its own and runs no licence
server. It runs with no licence switch set; unset means off.
