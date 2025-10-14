"""Models describing service health and diagnostics."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthLimits(BaseModel):
    """Operational limits that are useful when debugging."""

    chunk_size: int = Field(..., ge=1)
    chunk_overlap: int = Field(..., ge=0)
    top_k: int = Field(..., ge=1)
    max_upload_mb: int = Field(..., ge=1)
    max_pdf_pages: int = Field(..., ge=1)
    max_audio_minutes: int = Field(..., ge=1)


class HealthResponse(BaseModel):
    """Structured response for the ``/health`` endpoint."""

    status: Literal["ok"]
    provider: str
    vector_db: str
    environment: str
    debug_mode: bool
    limits: HealthLimits | None = None


__all__ = ["HealthLimits", "HealthResponse"]
