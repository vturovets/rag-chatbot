"""Models for ingestion endpoints."""
from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class FileMetadata(BaseModel):
    """Metadata describing an uploaded file."""

    file_id: UUID = Field(default_factory=uuid4)
    filename: str
    content_type: str
    size_bytes: int
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    kind: str


class UploadResponse(BaseModel):
    """Response returned after ingesting a file."""

    file_id: UUID
    filename: str
    kind: str
    page_count: int | None = None
    duration_seconds: float | None = None


class ExtractionResult(BaseModel):
    """Result of text extraction from a file."""

    text: str
    pages: int | None = None


class TranscriptionResult(BaseModel):
    """Result of an audio transcription."""

    transcript: str
    duration_seconds: float


__all__ = [
    "FileMetadata",
    "UploadResponse",
    "ExtractionResult",
    "TranscriptionResult",
]
