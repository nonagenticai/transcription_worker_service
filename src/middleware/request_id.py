"""Request ID middleware — binds inbound X-Request-Id to w2v_common ContextVar for trace continuity."""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from w2v_common.observability.request_context import set_request_id


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Bind X-Request-Id to the w2v_common ContextVar for inbound + outbound trace continuity.

    If the incoming request has X-Request-Id, it is preserved; otherwise a new uuid7 is generated.
    The id is bound to ``w2v_common.observability.request_context`` so that:

    * the JSON log formatter picks it up via ``get_request_id()``;
    * outbound httpx S2S calls forward it via ``trace_event_hook`` (registered on every
      ``BaseServiceClient``); peer services see the same trace id.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        rid = request.headers.get("x-request-id") or str(uuid.uuid7())
        request.state.request_id = rid
        set_request_id(rid)
        response = await call_next(request)
        response.headers["X-Request-Id"] = rid
        return response
