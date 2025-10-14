"""FastAPI application entrypoint."""
from __future__ import annotations

import logging
import uuid
from typing import Awaitable, Callable

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from app.backend import exceptions
from app.backend.api.routes import router as api_router
from app.backend.config import get_settings
from app.common.logging import configure_logging, reset_request_id, set_request_id

configure_logging()

app = FastAPI(title="RAG Chatbot", version="0.1.0")


_error_logger = logging.getLogger("app.backend.errors")


def _log_error(
    *,
    event: str,
    request: Request,
    payload: dict[str, object],
    status_code: int,
    exc: Exception | None,
) -> None:
    """Emit a structured log for handled exceptions."""

    level = logging.ERROR if status_code >= 500 else logging.WARNING
    fields = {
        "path": str(request.url.path),
        "method": request.method,
        "status_code": status_code,
        **payload,
    }
    extras = {f"_json_{key}": value for key, value in fields.items()}
    _error_logger.log(level, event, exc_info=exc, extra=extras)


def _json_response(status_code: int, payload: dict[str, object]) -> JSONResponse:
    return JSONResponse(status_code=status_code, content=payload)


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


@app.exception_handler(exceptions.RagError)
async def handle_rag_error(request: Request, exc: exceptions.RagError) -> JSONResponse:
    payload = exc.to_payload()
    _log_error(event="error.rag", request=request, payload=payload, status_code=exc.status_code, exc=exc)
    return _json_response(exc.status_code, payload)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    settings = get_settings()
    hint = None
    if settings.debug_mode:
        error_messages = [err.get("msg", "") for err in exc.errors()]
        hint = "; ".join(filter(None, error_messages)) or None
    error = exceptions.invalid_request(hint=hint)
    payload = error.to_payload()
    _log_error(event="error.request.validation", request=request, payload=payload, status_code=error.status_code, exc=exc)
    return _json_response(error.status_code, payload)


@app.exception_handler(StarletteHTTPException)
async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    if isinstance(exc, exceptions.RagError):
        return await handle_rag_error(request, exc)
    status_code = exc.status_code or status.HTTP_500_INTERNAL_SERVER_ERROR
    if status_code == status.HTTP_404_NOT_FOUND:
        error = exceptions.resource_not_found()
    else:
        settings = get_settings()
        hint = None
        if settings.debug_mode and exc.detail:
            hint = str(exc.detail)
        error = exceptions.http_error(
            status_code=status_code,
            message="An HTTP error occurred while processing the request.",
            hint=hint,
        )
    payload = error.to_payload()
    _log_error(event="error.http", request=request, payload=payload, status_code=status_code, exc=exc)
    return _json_response(status_code, payload)


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    error = exceptions.internal_error()
    payload = error.to_payload()
    _log_error(event="error.unexpected", request=request, payload=payload, status_code=error.status_code, exc=exc)
    return _json_response(error.status_code, payload)


__all__ = ["app"]
