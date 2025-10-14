"""FastAPI application entrypoint."""
from __future__ import annotations

from fastapi import FastAPI

from app.backend.api.routes import router as api_router
from app.common.logging import configure_logging

configure_logging()

app = FastAPI(title="RAG Chatbot", version="0.1.0")
app.include_router(api_router)


__all__ = ["app"]
