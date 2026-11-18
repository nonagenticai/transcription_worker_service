#!/usr/bin/env bash
# Production entrypoint for transcription_worker_service when run as a host
# process under launchd on mini1 (com.nonagentic.transcription-worker), invoked
# by gitops-production/scripts/host-configs/mini1/plists/
# com.nonagentic.transcription-worker.plist.
#
# WHY THIS FILE EXISTS
# --------------------
# Until this script landed, the plist exec'd `uv run uvicorn` DIRECTLY and never
# sourced an env file. ~/.config/transcription-worker/env existed and was kept
# current by the 5-minute host-config-sync agent, but nothing ever read it: the
# rendered ConfigMap + Secret values reached the disk and stopped there.
# `ps eww <pid> | grep -E '^(METERING|TRANSCRIPTION_|OTEL_)'` returned ZERO
# matches, and the service logged, on every startup:
#
#     w2v_common.metering.cost_bus - WARNING -
#     METERING_REDIS_URL not set -- cost events will not be emitted
#
# So the service ran wholly on code defaults and emitted no cost events. The
# emitters are fail-open by design, so this was silent: billing was simply
# quiet, with no error anywhere but that one WARNING line.
#
# Shape mirrors lean4_prover/start.sh (the sibling this plist's own comment
# named) rather than tts/image-gen's: no `lsof -ti :PORT | xargs kill -9`
# preamble. Under launchd KeepAlive that self-kill races the supervisor, and
# launchd already guarantees a single instance per label.
#
# Runtime config source of truth is the cluster, not this file:
#   clusters/strato/external-services/transcription-worker.yaml  (ConfigMap)
#   clusters/strato/external-services/launchd-metering-secrets.yaml (Secret,
#     transcription-worker-launchd-secret -- carries METERING_REDIS_URL, which
#     is deliberately ABSENT from the ConfigMap because it embeds the `metering`
#     ACL user's password)
# host-config-sync.sh merges the Secret OVER the ConfigMap into
# ~/.config/transcription-worker/env, then kickstarts this unit. Change a value
# there, not here.
#
# For local development use uvicorn directly with a .env file; this file is the
# ONLY entrypoint used by launchd in production.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source the rendered env file. `set -a` exports every assignment so pydantic
# BaseSettings (which reads os.environ, not the file) actually sees them.
ENV_FILE="${ENV_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/transcription-worker/env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
elif [ -f /etc/transcription-worker/env ]; then
  set -a
  # shellcheck disable=SC1091
  source /etc/transcription-worker/env
  set +a
fi

# NOTE: sourcing the env file does NOT arm the licence gate. LICENSE_GATE on the
# /api router is dormant unless W2V_LICENSE_GATE_ENABLED=true, and that key is
# deliberately not in the ConfigMap -- arming it requires modality_gateway to
# present a licence token first. Restoring metering and arming the gate are
# separate changes on purpose.

# API_HOST / API_PORT come from the ConfigMap (env_prefix "API_" in
# src/config.py). Defaults match src.config.APISettings so a missing env file
# still starts the service on its registered port rather than uvicorn's 8000.
exec uv run uvicorn src.asgi:app \
  --host "${API_HOST:-0.0.0.0}" \
  --port "${API_PORT:-6318}"
