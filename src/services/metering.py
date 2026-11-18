"""Cost metering — delegates to the centralized ``w2v_common.metering`` helper.

Historically this module carried its own ``emit_cost_event`` / ``DEFAULT_STREAM``
copy plus a hand-rolled fire-and-forget task set. That logic now lives once in
``w2v_common.metering`` (stream resolution, field marshalling, GC-safe task
retention, and the Redis lifecycle). This module keeps only the
transcription-specific routing identity (``COST_SOURCE`` / ``SERVICE_KEY``)
and re-exports the helper entry points so importers do not need to know where
the implementation lives.
"""

from __future__ import annotations

from typing import Final

from w2v_common.metering import (
    close_metering_redis,
    init_metering_redis,
    schedule_cost_event,
)

__all__ = [
    "COST_SOURCE",
    "FUNDING",
    "SERVICE_KEY",
    "close_metering_redis",
    "init_metering_redis",
    "schedule_cost_event",
]

COST_SOURCE = "w2v-transcription"
SERVICE_KEY = "w2v-transcription"  # pragma: allowlist secret -- routing slug, not a credential

#: WHO FUNDS a decode: every backend here (faster-whisper / mlx-whisper) runs on
#: the platform's own mini1 GPU, so every event is ``local_gpu`` (billing
#: FUNDING_VALUES). No external vendor and no tenant key on this path.
FUNDING: Final = "local_gpu"

# NO TENANT SENTINEL LIVES HERE. Product-owner ruling: a ``tenant_id`` may NEVER
# hold a sentinel. Every message carries a REAL registered tenant forwarded from
# the requestor; an absent tenant is an ERROR, never a substituted default.
#
# This module used to export ``DEFAULT_TENANT_ID = UNATTRIBUTED_TENANT_ID`` (the
# nil UUID). That is the most dangerous spelling of the lie: it is a VALID uuid,
# so it passes every uuid cast and dirty-row check, and billing then buckets it
# as ``__unattributed__`` and NEVER CHARGES IT (billing-service
# ``_parsers.py`` + ``cost_flusher.py``) -- real usage, silently unbilled.
#
# Do not re-add a default here or in ``src/api/routes.py``. The tenant is
# required at the request boundary by ``routes._require_tenant_id`` (HTTP 401).
