# Multi-stage Dockerfile for transcription_worker_service
#
# Single-context build: w2v-common is resolved from pypiserver.test
# via the extra-index-url in pyproject.toml.
#
# Build with: docker buildx build -f Dockerfile --add-host pypiserver.test:192.168.1.241 .

ARG BASE_REGISTRY=registry.test:5000
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /bin/

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM base AS builder

ENV UV_HTTP_TIMEOUT=600

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    ca-certificates \
    libsndfile-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock README.md ./

# Resolve and install deps (torch is heavy — caches well in this layer)
RUN uv sync --frozen --no-dev --no-install-project

# Now copy source and install the project itself
COPY src/ ./src/

RUN uv sync --frozen --no-dev

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM ${BASE_REGISTRY}/library/python:3.14-slim AS production

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 transcriber && \
    mkdir -p /app /app/logs && \
    chown -R transcriber:transcriber /app

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=transcriber:transcriber /app/.venv /app/.venv

# Copy application code
COPY --chown=transcriber:transcriber src/ ./src/

ENV PATH="/app/.venv/bin:$PATH"

USER 1000:1000

# Port 6318 (Transcription Worker Service)
EXPOSE 6318

# Kubernetes livenessProbe / readinessProbe in clusters/strato/services/w2v/
# are the source of truth for health. A Dockerfile HEALTHCHECK is dead
# weight under k8s and only causes drift if the probe paths change.

CMD ["python", "-m", "src.asgi"]
