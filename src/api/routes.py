"""Core transcription routes."""

from __future__ import annotations

import logging
import math
import tempfile
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any, cast, overload

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from w2v_common.admission import GateClosedError
from w2v_common.auth import LICENSE_GATE
from w2v_common.auth.identity_context import (
    get_attribution_subject_id_from_context,
    get_license_id_from_context,
)
from w2v_common.metering import cost_event_key

from src.api.schemas import (
    ModelInfo,
    ModelsResponse,
    NoSpeechReason,
    TranscriptionResponse,
    TranscriptSegmentResponse,
    TranscriptWordResponse,
)
from src.services.admission import TRANSCRIBE_LANE, get_gate
from src.services.backends.base import (
    BackendUnavailableError,
    TranscriptionResult,
    TranscriptSegment,
    WhisperBackend,
)
from src.services.metering import (
    COST_SOURCE,
    FUNDING,
    SERVICE_KEY,
    schedule_cost_event,
)
from src.services.srt_formatter import format_srt

logger = logging.getLogger(__name__)

# Entitlement gate, mounted on the TOP-LEVEL router so every current route
# (/api/transcribe, /api/models) and any future sub-router included under this
# prefix inherits it.
#
# AuthMiddleware answers "who are you" (identity) and stops there: a tenant
# whose w2v licence is expired, revoked, or never existed still presents a
# syntactically valid platform token, and every /api route served them —
# Whisper compute, tailnet-wide. LICENSE_GATE answers "may you".
#
# Exemptions are w2v_common's (PUBLIC_ENDPOINTS plus /health, /healthz, /ready);
# no hand-rolled list here — a wrong one 402s the probe and flaps the launchd
# unit. The gate is DORMANT unless W2V_LICENSE_GATE_ENABLED=true.
router = APIRouter(prefix="/api", tags=["transcription"], dependencies=list(LICENSE_GATE))

_AVAILABLE_MODEL_SIZES = [
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
]


def _backend_from_request(request: Request) -> WhisperBackend:
    backend = getattr(request.app.state, "backend", None)
    if backend is None:
        raise HTTPException(status_code=503, detail="Transcription backend not initialised")
    return backend


def _unlink_tmp(tmp_path: Path) -> None:
    try:
        tmp_path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Failed to delete temp audio file %s", tmp_path)


def _decode_in_lane(
    backend: WhisperBackend,
    raw: bytes,
    suffix: str,
    language: str | None,
    word_timestamps: bool,
) -> TranscriptionResult:
    """Spill the upload to a temp file, decode it, delete it -- all on the lane thread.

    Runs inside ``gate.run`` (the ``transcribe`` lane's executor), so the temp
    file's whole lifetime is bounded by THIS thread: it does not exist before
    admission (a shed / queue-timed-out / disconnected request never touches the
    disk) and it is unlinked only when the decode really ends. A client
    disconnect cancels only the awaiting coroutine -- the thread keeps decoding
    and keeps both its slot (released by the gate when the thread ends) and its
    input file.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)  # noqa: SIM115 — closed below
    tmp_path = Path(tmp.name)
    try:
        # The write is inside the try: a failed write (disk full) must not leave
        # a half-written upload behind.
        with tmp:
            tmp.write(raw)
        return backend.transcribe(tmp_path, language, word_timestamps)
    finally:
        _unlink_tmp(tmp_path)


def _record_backend_state(request: Request, *, error: str | None) -> None:
    """Publish backend load health so /health/ready can stop advertising a
    service whose model will never load (see src/api/health.py)."""
    request.app.state.backend_load_error = error


@overload
def _json_safe_float(value: float | None, fallback: float) -> float: ...


@overload
def _json_safe_float(value: float | None, fallback: None = None) -> float | None: ...


def _json_safe_float(value: float | None, fallback: float | None = None) -> float | None:
    """Coerce a non-finite float to ``fallback`` so the response honours its schema.

    What this is NOT: it is not a fix for a bare ``NaN``/``Infinity`` token in
    the body. Nothing on this path ever emitted one -- pydantic's
    ``ser_json_inf_nan`` defaults to ``'null'`` and
    ``fastapi.responses.JSONResponse.render`` already passes ``allow_nan=False``
    -- so an unsanitized degenerate result returns HTTP 200 with syntactically
    valid JSON. The body parses fine.

    The two REAL defects:

    1. **Contract violation.** ``null`` lands in fields the response model
       declares as REQUIRED ``float`` (``duration``, segment ``start``/``end``,
       word ``start``/``end``). A strictly-typed client breaks on that: Rust
       ``serde`` refuses ``null`` for ``f64``, and Go ``encoding/json`` silently
       coerces it to ``0.0`` -- a fabricated timing nobody flagged. The fields
       that are ``float | None`` in the schema (``probability``,
       ``avg_logprob``, ``no_speech_prob``) are NOT violated by ``null``.
    2. **``return_srt=true`` hard-500s.** ``format_srt`` does
       ``round(seconds * 1000)``: ``ValueError`` on nan, ``OverflowError`` on
       inf. That one is a real request failure, not a typing nicety.

    ``fallback`` is chosen PER FIELD by what the value means, not uniformly:

    * timings (segment/word ``start``/``end``, ``duration``) -> ``0.0``. The
      response model requires a float there, so ``None`` is not a legal value;
      0.0 is the only in-domain "unknown" that still renders and still sorts.
    * ``probability`` / ``no_speech_prob`` / ``avg_logprob`` -> ``None``. These
      are optional in the model and ``None`` already means "the backend did not
      report it". Substituting a number would FABRICATE a confidence: 0.0
      no_speech_prob asserts "definitely speech", and 0.0 is the MAXIMUM of a
      log-probability (log(1) = certainty), so coercing ``avg_logprob`` to 0.0
      inverts a ``-inf`` (worst possible) into a perfect score and can sail
      through a downstream quality gate.

    ``None`` input stays ``None``; finite values pass through untouched.
    """
    if value is None or not math.isfinite(value):
        return fallback
    return value


def _classify_no_speech(
    *,
    text: str,
    segments: list[TranscriptSegmentResponse],
    duration: float,
) -> tuple[bool, NoSpeechReason | None]:
    """Classify an empty transcript so downstream cannot fabricate a claim.

    Before this existed, a suppressed decode and a genuinely silent clip both
    returned ``{"text": "", "segments": []}`` with HTTP 200, and
    ``video_analyzer_service`` recorded BOTH as the factual assertion "this
    video has no speech" — a fabricated finding whenever the decode simply
    failed to produce anything.

    The discriminator is the MEDIA duration, which is the one piece of positive
    evidence that survives an empty decode: it is measured from the submitted
    audio (faster-whisper's ``info.duration``, openai-whisper's decoded PCM),
    not from the segments. A positive duration with zero segments means the
    engine really did process N seconds and emit nothing => silence. A
    non-positive duration means the audio was never successfully measured at
    all, so nothing about its content is known. Segments that carry no text are
    likewise a degenerate decode, not evidence of silence.
    """
    if text.strip():
        return False, None
    if not segments and duration > 0.0:
        return True, NoSpeechReason.SILENT_AUDIO
    return True, NoSpeechReason.EMPTY_DECODE


def _serialize_result(result: TranscriptionResult, include_srt: bool) -> TranscriptionResponse:
    segments: list[TranscriptSegmentResponse] = []
    srt_segments: list[TranscriptSegment] = []
    for seg in result.segments:
        # Timings are sanitized ONCE and reused for the SRT: ``format_srt``
        # does ``round(seconds * 1000)``, which raises ValueError on nan and
        # OverflowError on inf -- so an unsanitized segment 500s the request
        # whenever ``return_srt=true``. That 500 is the strongest part of this
        # guard: without it the request fails outright, whereas the JSON path
        # merely returns a schema-violating ``null``.
        #
        # Accepted consequence: when BOTH start and end are non-finite the cue
        # collapses to 00:00:00,000 --> 00:00:00,000 (zero length). That is
        # acceptable -- the timing is genuinely unknown, and a zero-length cue
        # is a visible, self-describing artefact that keeps the rest of the
        # subtitle track (and every other cue's timing) intact. Inventing a
        # duration would be worse: it would silently misalign real speech.
        start = _json_safe_float(seg.start, 0.0)
        end = _json_safe_float(seg.end, 0.0)
        segments.append(
            TranscriptSegmentResponse(
                id=seg.id,
                start=start,
                end=end,
                text=seg.text,
                words=[
                    TranscriptWordResponse(
                        start=_json_safe_float(w.start, 0.0),
                        end=_json_safe_float(w.end, 0.0),
                        word=w.word,
                        probability=_json_safe_float(w.probability),
                    )
                    for w in seg.words
                ],
                avg_logprob=_json_safe_float(seg.avg_logprob),
                no_speech_prob=_json_safe_float(seg.no_speech_prob),
            )
        )
        if include_srt:
            srt_segments.append(replace(seg, start=start, end=end))
    srt_text = format_srt(srt_segments) if include_srt else None
    # Sanitize a non-finite duration at the serialization boundary: pydantic
    # renders nan/inf as ``null``, and ``duration`` is a REQUIRED ``float`` in
    # TranscriptionResponse -- so the emitted body contradicts its own schema
    # and breaks a strictly-typed client. The text is real; only the
    # unmeasurable duration metric is coerced to 0.0 (matching
    # ``_audio_seconds``' 0 bill).
    duration = _json_safe_float(result.duration, 0.0)
    no_speech, no_speech_reason = _classify_no_speech(
        text=result.text, segments=segments, duration=duration
    )
    return TranscriptionResponse(
        backend=result.backend,
        model=result.model,
        language=result.language,
        duration=duration,
        text=result.text,
        segments=segments,
        srt=srt_text,
        no_speech=no_speech,
        no_speech_reason=no_speech_reason,
    )


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    operation_id="create_transcription",
    summary="Transcribe an uploaded audio file",
)
async def transcribe_endpoint(
    request: Request,
    file: Annotated[UploadFile, File(..., description="Audio file to transcribe")],
    language: Annotated[str | None, Form()] = None,
    word_timestamps: Annotated[bool, Form()] = True,
    return_srt: Annotated[bool, Form()] = False,
) -> TranscriptionResponse:
    """Transcribe an audio file using the loaded Whisper backend."""
    # Fail loud on an absent tenant BEFORE spending any GPU decode: unattributed
    # work used to be billed to the nil-UUID sentinel and therefore never charged.
    _require_tenant_id(request)
    # Identity of THIS physical decode, minted once per request. The decode is
    # synchronous (no queue, no redelivery), so the request IS the call and the
    # cost event's idempotency key derives from it (see ``_emit_cost_event``).
    call_id = str(uuid.uuid7())
    backend = _backend_from_request(request)
    settings = request.app.state.settings

    # Bound the read to one byte past the limit: reject oversized uploads
    # without ever materialising the whole (potentially multi-GB) payload in
    # memory. An unbounded ``file.read()`` runs the size check only after the
    # full body is buffered, so the cap did not actually bound peak memory.
    max_bytes = settings.transcription.max_audio_bytes
    raw = await file.read(max_bytes + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio payload")
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail="Audio payload exceeds size limit")

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    t0 = time.monotonic()
    # Admission: the shared ``transcribe`` lane (see src/services/admission.py).
    # GpuBusy (queue full / waited too long) -> 503 + Retry-After, ClientGone ->
    # 499 -- both mapped by install_gpu_busy_handler. The slot is held until the
    # decode THREAD ends, even if this coroutine is cancelled.
    try:
        result = await get_gate(request).run(
            TRANSCRIBE_LANE,
            _decode_in_lane,
            backend,
            raw,
            suffix,
            language,
            word_timestamps,
            request=request,
        )
    except GateClosedError as exc:
        raise HTTPException(status_code=503, detail="Service is shutting down") from exc
    except BackendUnavailableError as exc:
        logger.error("Backend unavailable: %s", exc)
        _record_backend_state(request, error=str(exc))
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
    _record_backend_state(request, error=None)

    _emit_cost_event(
        request,
        result=result,
        latency_ms=int((time.monotonic() - t0) * 1000),
        call_id=call_id,
    )
    return _serialize_result(result, include_srt=return_srt)


def _audio_seconds(duration: float) -> int:
    """Billable audio-seconds quantity for a clip of ``duration`` seconds.

    ``input_tokens`` is an integer count, so the float media duration must be
    rounded to a whole second. We round UP (``math.ceil``): any real
    transcription of positive length bills a started second — the conventional
    audio-billing rounding — so a sub-second clip (e.g. 0.4 s) bills 1, never 0.
    A plain ``int()`` truncates and zero-bills every clip under one second,
    charging real work as nothing. A non-positive duration is no work -> 0.

    A NON-FINITE duration (``nan`` / ``inf`` from a degenerate backend result)
    is guarded FIRST: ``math.ceil(nan)`` raises ``ValueError`` and
    ``math.ceil(inf)`` raises ``OverflowError`` -- unguarded that 500s a
    completed, GPU-consumed transcription and (because emission runs after this
    call) drops the cost event entirely, billing real work as nothing. A
    non-finite duration means the clip length is UNMEASURABLE, so we bill 0
    audio-seconds (bill nothing measurable) rather than fabricate a quantity.
    """
    if not math.isfinite(duration):
        return 0
    if duration <= 0.0:
        return 0
    return math.ceil(duration)


def _require_tenant_id(request: Request) -> str:
    """Return the caller's REAL forwarded tenant, or 401 — never a sentinel.

    Product-owner ruling: a ``tenant_id`` may NEVER hold a sentinel. All traffic
    carries a real registered tenant forwarded from the requestor; an absent one
    is an ERROR, not a substituted default.

    This replaces ``tenant_id = request.state.tenant_id or DEFAULT_TENANT_ID``,
    where ``DEFAULT_TENANT_ID`` was the nil UUID. The nil UUID is a VALID uuid,
    so it passed every cast and dirty-row check, and billing then bucketed it as
    ``__unattributed__`` and NEVER CHARGED IT — real GPU transcription work,
    silently unbilled.

    401 (not 400) is the correct shape here, matching ``slide_service``: this
    service takes no tenant from the body or path, so the tenant can only come
    from the authenticated identity. ``request.state.tenant_id`` is populated by
    ``w2v_common.auth.middleware.AuthMiddleware`` from the JWT ``tenant_id``
    claim / licenses-service lookup. Its absence means the request carries no
    usable identity — an authentication failure, not a malformed payload.
    (``video_compiler_service`` maps its ``MissingTenantError`` to 400 because
    there the tenant is a caller-supplied field the caller can correct.)

    Enforced at the TOP of ``transcribe`` so an unattributed request is rejected
    BEFORE any GPU decode is spent, and re-checked in ``_emit_cost_event`` as a
    tripwire so no code path can reach the cost bus without a real tenant.
    """
    tenant_id = (getattr(request.state, "tenant_id", None) or "").strip()
    # Reject the nil UUID BY VALUE, in every spelling (dashed, undashed,
    # braced, urn:, any case). A blank/None-only guard cannot see it: the nil
    # uuid is a VALID uuid, which is exactly why it survived as a sentinel and
    # why billing never charged that bucket.
    if tenant_id:
        try:
            if uuid.UUID(tenant_id).int == 0:
                tenant_id = ""
        except ValueError:
            pass
    if not tenant_id:
        raise HTTPException(
            status_code=401,
            detail=(
                "No tenant identity on the request. transcription_worker_service is "
                "tenant-scoped and metered: authenticate with a license token or JWT "
                "carrying a tenant_id claim. No default tenant is substituted."
            ),
        )
    return tenant_id


def _emit_cost_event(
    request: Request,
    *,
    result: TranscriptionResult,
    latency_ms: int,
    call_id: str,
) -> None:
    """Fire-and-forget metering emission. No-ops when metering Redis is disabled.

    G5 full attribution (every field stated, none left for billing to guess):
    ``app_id`` is the CALLING app from ``X-App-Id`` (this worker serves more than
    one app, so hard-coding one would mislabel the others); ``funding`` is
    ``local_gpu`` (local whisper on mini1); ``event_key`` derives from
    ``call_id``, one per physical decode. Under ``METERING_REQUIRE_ATTRIBUTION``
    a missing licence/subject/app raises instead of emitting a partial event.

    Encoding for non-token workloads, per ``billing-service/docs/EMITTER_GUIDE.md``
    ("use the natural unit ... cast to int. Document the unit in usage_extra"):
        input_tokens  = ceil(audio_duration_seconds)  # >=1 for any real clip
        output_tokens = total_transcript_characters
        usage_extra   = {"unit": "audio_seconds", "duration_s": ..., "backend": ...}

    ``input_tokens`` MUST be an int. The emitter marshals it with ``str()``, and
    the billing consumer parses with ``int(val)`` falling back to 0 on
    ``ValueError`` -- so emitting a float would put "12.5" on the stream and
    SILENTLY zero-bill the event. Exact float precision lives in
    ``usage_extra["duration_s"]``, which is metadata and not used for billing math.
    """
    # Tripwire: fail loud rather than emit an unattributed cost event. The route
    # already gated this before the decode, so this can only fire if a future
    # caller reaches emission without the gate.
    tenant_id = _require_tenant_id(request)
    metering_redis = getattr(request.app.state, "metering_redis", None)
    if metering_redis is None:
        return
    # Sanitize the media duration ONCE so no raw non-finite value reaches billing
    # OR the audit metadata. The cost bus serializes ``usage_extra`` with
    # ``json.dumps(..., allow_nan=True)``, so a nan/inf ``duration_s`` would emit
    # bare ``NaN``/``Infinity`` -- invalid JSON (RFC 8259) that a strict metering
    # consumer rejects, even though ``input_tokens`` is already 0. A non-finite
    # duration means the clip length is UNMEASURABLE, so it bills 0 audio-seconds
    # and records 0.0 metadata (matching ``_audio_seconds`` / ``_serialize_result``).
    billed_duration = result.duration if math.isfinite(result.duration) else 0.0
    # Fire-and-forget; the helper retains a strong ref to the spawned task and
    # resolves the cost-bus stream (METERING_STREAM env, else canonical) itself.
    schedule_cost_event(
        cast(Any, metering_redis),
        service_key=SERVICE_KEY,
        cost_source=COST_SOURCE,
        tenant_id=tenant_id,
        license_id=get_license_id_from_context(),
        subject_id=get_attribution_subject_id_from_context(),
        app_id=request.headers.get("X-App-Id", "").strip(),
        provider=result.backend,
        model=result.model,
        input_tokens=_audio_seconds(billed_duration),
        output_tokens=len(result.text),
        latency_ms=latency_ms,
        cost_usd=0.0,
        usage_extra={
            "unit": "audio_seconds",
            "duration_s": billed_duration,
            "backend": result.backend,
        },
        funding=FUNDING,
        event_key=cost_event_key(SERVICE_KEY, "transcribe", call_id),
    )


@router.get(
    "/models",
    response_model=ModelsResponse,
    operation_id="list_transcription_models",
    summary="List available Whisper models and the currently loaded one",
)
async def list_models(request: Request) -> ModelsResponse:
    """Return the currently loaded model plus available size strings."""
    backend = _backend_from_request(request)
    info = ModelInfo(
        backend=backend.name,
        model=backend.model_size,
        device=backend.device,
        compute_type=backend.compute_type,
        loaded=bool(backend.is_loaded),
    )
    return ModelsResponse(current=info, available_sizes=_AVAILABLE_MODEL_SIZES)
