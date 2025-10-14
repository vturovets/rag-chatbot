"""Models related to chat operations."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A normalized chunk of source content."""

    chunk_id: UUID = Field(default_factory=uuid4)
    text: str
    source_file_id: UUID
    order: int


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


class DebugPipelineRequest(BaseModel):
    """Request for the debug pipeline endpoint."""

    file_id: UUID


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
    "DebugPipelineRequest",
    "DebugPipelineResponse",
    "SessionContext",
]
