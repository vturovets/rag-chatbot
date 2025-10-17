"""Models for ingestion endpoints and persistence."""
from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class FileKind(str, Enum):
    """Supported upload categories."""

    PDF = "pdf"
    AUDIO = "audio"


class FileMetadata(BaseModel):
    """Metadata describing an uploaded file."""

    file_id: UUID = Field(default_factory=uuid4)
    filename: str
    content_type: str
    size_bytes: int
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(default_factory=lambda: datetime.utcnow() + timedelta(hours=24))
    kind: FileKind
    source: str | None = None

    def has_expired(self, reference: datetime | None = None) -> bool:
        """Return True if the metadata has expired relative to ``reference``."""

        reference = reference or datetime.utcnow()
        return reference >= self.expires_at


class FileRecord(BaseModel):
    """Materialized record that combines metadata with filesystem location."""

    metadata: FileMetadata
    path: Path

    def has_expired(self, reference: datetime | None = None) -> bool:
        return self.metadata.has_expired(reference)


class UploadResponse(BaseModel):
    """Response returned after ingesting a file."""

    file_id: UUID
    filename: str
    kind: FileKind
    source: str
    page_count: int | None = None
    duration_seconds: float | None = None
    expires_at: datetime
    session_id: UUID | None = None


class ExtractionResult(BaseModel):
    """Result of text extraction from a file."""

    text: str
    pages: int | None = None


class TranscriptionResult(BaseModel):
    """Result of an audio transcription."""

    transcript: str
    duration_seconds: float


__all__ = [
    "FileKind",
    "FileMetadata",
    "FileRecord",
    "UploadResponse",
    "ExtractionResult",
    "TranscriptionResult",
]
