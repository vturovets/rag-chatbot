"""Models related to chat operations."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

from app.backend.models.ingestion import FileKind


class Chunk(BaseModel):
    """A normalized chunk of source content."""

    chunk_id: UUID = Field(default_factory=uuid4)
    text: str
    source_file_id: UUID
    order: int
    source: FileKind


class EmbedVector(BaseModel):
    """Represents a vector embedding for a chunk."""

    chunk_id: UUID
    values: List[float]


class RetrievalHit(BaseModel):
    """A retrieval match for a query."""

    chunk_id: UUID
    score: float
    text: str
    source_file_id: UUID
    source: FileKind


class ChatRequest(BaseModel):
    """Request payload for the chat endpoint."""

    query: str
    top_k: int | None = Field(default=None, ge=1, le=8)
    session_id: UUID | None = None


class ChatResponse(BaseModel):
    """Response payload for the chat endpoint."""

    answer: str
    latency_ms: int
    session_id: UUID


class PipelineStageDiagnostics(BaseModel):
    """Diagnostics for debug pipeline responses."""

    stage: str
    input_payload: Dict[str, Any]
    output_payload: Dict[str, Any]
    completed_at: datetime = Field(default_factory=datetime.utcnow)


class DebugChunkPayload(BaseModel):
    """Represents a chunk supplied directly to the debug pipeline."""

    chunk_id: UUID
    order: int
    text: str | None = None
    preview: str | None = None

    @model_validator(mode="after")
    def _ensure_text(self) -> "DebugChunkPayload":
        content = (self.text or self.preview or "").strip()
        if not content:
            raise ValueError("Chunk payload must include text or preview content")
        return self

    def resolved_text(self) -> str:
        """Return the best available textual representation for the chunk."""

        return (self.text or self.preview or "").strip()


class DebugPipelineRequest(BaseModel):
    """Request for the debug pipeline endpoint."""

    file_id: UUID | None = None
    text: str | None = None
    query: str | None = None
    chunk_size: int | None = Field(default=None, ge=1)
    chunk_overlap: int | None = Field(default=None, ge=0)
    top_k: int | None = Field(default=None, ge=1, le=8)
    chunks: List[DebugChunkPayload] | None = None

    @model_validator(mode="after")
    def _validate_payload(self) -> "DebugPipelineRequest":
        if self.file_id is None:
            text = (self.text or "").strip()
            query = (self.query or "").strip()
            has_chunks = bool(self.chunks)
            if not text and not has_chunks and not query:
                raise ValueError(
                    "text, chunks, or query is required when file_id is not provided"
                )
        if self.chunk_size is not None and self.chunk_overlap is not None:
            if self.chunk_overlap >= self.chunk_size:
                raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class DebugPipelineResponse(BaseModel):
    """Response for debug pipeline invocation."""

    stages: List[PipelineStageDiagnostics]


class SessionContext(BaseModel):
    """Represents ephemeral chat session state."""

    session_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_interaction_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(default_factory=lambda: datetime.utcnow() + timedelta(hours=24))
    active_file_ids: List[UUID] = Field(default_factory=list)

    @classmethod
    def new(
        cls,
        *,
        session_id: UUID | None = None,
        ttl_hours: int = 24,
        active_file_ids: List[UUID] | None = None,
    ) -> "SessionContext":
        created = datetime.utcnow()
        return cls(
            session_id=session_id or uuid4(),
            created_at=created,
            last_interaction_at=created,
            expires_at=created + timedelta(hours=ttl_hours),
            active_file_ids=list(active_file_ids or []),
        )

    def touch(self) -> None:
        """Refresh the last interaction timestamp."""

        self.last_interaction_at = datetime.utcnow()

    def has_expired(self, reference: datetime | None = None) -> bool:
        reference = reference or datetime.utcnow()
        return reference >= self.expires_at


__all__ = [
    "Chunk",
    "EmbedVector",
    "RetrievalHit",
    "ChatRequest",
    "ChatResponse",
    "PipelineStageDiagnostics",
    "DebugChunkPayload",
    "DebugPipelineRequest",
    "DebugPipelineResponse",
    "SessionContext",
]
