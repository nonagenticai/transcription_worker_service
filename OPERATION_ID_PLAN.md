# Operation ID Plan: transcription-worker-service

Canonical registry of `operation_id` values for every route in
`transcription-worker-service`. These IDs feed the org-wide n8n custom-node
generator and the cross-service catalogue, so they must be:

- **stable** — never rename without coordinating downstream,
- **`verb_noun` snake_case** — matches the org standard (analysis_service,
  nlp_service, worker_management, kag),
- **globally unique** — disambiguated against the rest of the fleet.

This service exposes a small surface: a single transcription POST, a model
metadata GET, plus the standard root/metrics/health probes. Generic verbs
that would collide cross-service (`list_models`, `create_transcription`) are
domain-prefixed where needed.

## Files to edit

- `src/asgi.py` — root + metrics endpoints
- `src/api/health.py` — health probes
- `src/api/routes.py` — transcription router

## Mapping Tables

### `src/asgi.py`

| Method | Function   | operation_id          | Rationale                                                                   |
|--------|------------|-----------------------|-----------------------------------------------------------------------------|
| GET    | `root`     | `service_root`        | Standard fleet name for the service banner endpoint.                        |
| GET    | `metrics`  | `prometheus_metrics`  | Matches analysis_service / kag fleet convention for the `/metrics` scrape.  |

### `src/api/health.py` (tag: `health`)

| Method | Function       | operation_id    | Rationale                                                                    |
|--------|----------------|-----------------|------------------------------------------------------------------------------|
| GET    | `health`       | `health_check`  | Combined health probe — `health_check` matches the fleet template.           |
| GET    | `health_live`  | `health_live`   | k8s liveness probe; one-to-one with route name to keep grep-ability obvious. |
| GET    | `health_ready` | `health_ready`  | k8s readiness probe; same convention.                                        |

### `src/api/routes.py` (tag: `transcription`, prefix `/api`)

| Method | Function              | operation_id                  | Rationale                                                                                                |
|--------|-----------------------|-------------------------------|----------------------------------------------------------------------------------------------------------|
| POST   | `transcribe_endpoint` | `create_transcription`        | `verb_noun` form of "transcribe audio"; the implementation name leaks an `_endpoint` suffix we hide.     |
| GET    | `list_models`         | `list_transcription_models`   | `list_models` would collide with `worker_management.list_workers`-style verbs and the inference-router. |

## Verification

```bash
cd transcription_worker_service
uv run pytest tests/test_openapi.py -x
```

`tests/test_openapi.py` loads `app.openapi()` and asserts that the set of
`operationId` values equals the union of the right-hand column above.
Drift (rename, missed decorator, fallback) fails the test loudly.

## Future routes

When adding a route in this service, **always** set `operation_id="…"`
explicitly. The `generate_unique_id_function` configured in `asgi.py` is a
safety net for forgotten kwargs, never the primary identifier. New IDs go
in the table above as part of the same PR (no out-of-band updates).
