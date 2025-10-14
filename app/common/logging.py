"""Structured logging helpers."""
from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import Any, Iterator

_LOG_FORMAT = "%(message)s"
_REQUEST_ID: ContextVar[str | None] = ContextVar("request_id", default=None)


def _timestamp() -> str:
    """Return an ISO-8601 timestamp with millisecond precision."""

    return datetime.now(tz=timezone.utc).isoformat(timespec="milliseconds")


def set_request_id(request_id: str | None) -> Token[str | None]:
    """Bind the request identifier into the logging context."""

    return _REQUEST_ID.set(request_id)


def reset_request_id(token: Token[str | None]) -> None:
    """Reset the request identifier context to a previous token."""

    _REQUEST_ID.reset(token)


def get_request_id() -> str | None:
    """Return the current request identifier, if any."""

    return _REQUEST_ID.get()


def configure_logging(level: int = logging.INFO) -> None:
    """Configure application-wide JSON logging."""

    class JsonLogFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:  # noqa: D401
            payload = {
                "level": record.levelname,
                "name": record.name,
                "message": record.getMessage(),
                "timestamp": getattr(record, "timestamp", _timestamp()),
            }
            request_id = getattr(record, "request_id", None) or get_request_id()
            if request_id:
                payload["request_id"] = request_id
            if record.exc_info:
                payload["exc_info"] = self.formatException(record.exc_info)
            for key, value in record.__dict__.items():
                if key.startswith("_json_"):
                    payload[key[6:]] = value
            return json.dumps(payload, ensure_ascii=False)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonLogFormatter(_LOG_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(handler)


def json_log(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Emit a structured JSON log entry with optional payload fields."""

    extras = {f"_json_{key}": value for key, value in fields.items()}
    extras.setdefault("timestamp", _timestamp())
    request_id = get_request_id()
    if request_id:
        extras.setdefault("request_id", request_id)
    logger.log(level, message, extra=extras)


@contextmanager
def scoped_request_id(request_id: str | None) -> Iterator[None]:
    """Context manager that temporarily binds a request identifier."""

    token = set_request_id(request_id)
    try:
        yield
    finally:
        reset_request_id(token)


__all__ = [
    "configure_logging",
    "get_request_id",
    "json_log",
    "reset_request_id",
    "scoped_request_id",
    "set_request_id",
]
