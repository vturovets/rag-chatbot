"""Structured logging helpers."""
from __future__ import annotations

import json
import logging
import sys
from typing import Any

_LOG_FORMAT = "%(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure application-wide JSON logging."""

    class JsonLogFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:  # noqa: D401
            payload = {
                "level": record.levelname,
                "name": record.name,
                "message": record.getMessage(),
            }
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


__all__ = ["configure_logging"]
