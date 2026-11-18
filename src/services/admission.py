"""GPU admission control for the shared Whisper model (w2v_common.admission).

One lane, ``transcribe``, fronts the single shared Whisper model. Code defaults
are ``max_active=1 / max_waiting=8 / max_wait_s=30``; each is overridable via
``GPU_LANE_TRANSCRIBE_MAX_ACTIVE|_MAX_WAITING|_MAX_WAIT_S`` (read once, when the
gate is built).

``max_active`` MUST stay 1 unless the backend is proven re-entrant: both
CTranslate2's ``WhisperModel`` and openai-whisper's model object carry mutable
decoder state, so overlapping calls can interleave that state (corrupt output),
and each in-flight call holds a full copy of the audio + activations (OOM on a
launchd Mac). Raising it is a deliberate, benchmarked decision.

Queue full, or waited past ``max_wait_s`` -> 503 ``{"error":"gpu_busy",...}`` +
``Retry-After`` (handler installed by ``install_gpu_busy_handler``); a client
that disconnected before admission is skipped (bodyless 499). Metrics:
``gpu_admission_*{lane="transcribe"}`` on the default registry (``/metrics``).
"""

from __future__ import annotations

from fastapi import Request
from w2v_common.admission import AdmissionGate, LaneConfig

TRANSCRIBE_LANE = "transcribe"
TRANSCRIBE_LANE_DEFAULTS = LaneConfig(max_active=1, max_waiting=8, max_wait_s=30.0)


def build_gate() -> AdmissionGate:
    """Build the service's gate (code defaults + ``GPU_LANE_TRANSCRIBE_*`` env)."""
    return AdmissionGate({TRANSCRIBE_LANE: TRANSCRIBE_LANE_DEFAULTS})


def get_gate(request: Request) -> AdmissionGate:
    """Return the app-wide gate, creating it on first use.

    The lifespan builds it up front; this lazy path exists so any app that wires
    the router in (including tests) is bounded too, not only the lifespan's.
    """
    gate: AdmissionGate | None = getattr(request.app.state, "gpu_gate", None)
    if gate is None:
        gate = build_gate()
        request.app.state.gpu_gate = gate
    return gate


__all__ = ["TRANSCRIBE_LANE", "TRANSCRIBE_LANE_DEFAULTS", "build_gate", "get_gate"]
