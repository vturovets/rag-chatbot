"""FastAPI application entrypoint."""
from __future__ import annotations

import uuid
from typing import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.backend.api.routes import router as api_router
from app.common.logging import configure_logging, set_request_id, reset_request_id

configure_logging()

app = FastAPI(title="RAG Chatbot", version="0.1.0")


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a request ID to each incoming request for traceability."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:  # type: ignore[override]
        supplied = request.headers.get("x-request-id") or request.headers.get("X-Request-ID")
        request_id = supplied or str(uuid.uuid4())
        token = set_request_id(request_id)
        try:
            response = await call_next(request)
        finally:
            reset_request_id(token)
        response.headers.setdefault("X-Request-ID", request_id)
        return response


app.add_middleware(RequestIDMiddleware)
app.include_router(api_router)


__all__ = ["app"]
