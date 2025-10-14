"""Models related to chat operations."""
from __future__ import annotations

from datetime import datetime
from typing import List
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
    input_payload: dict
    output_payload: dict
    completed_at: datetime = Field(default_factory=datetime.utcnow)


class DebugPipelineRequest(BaseModel):
    """Request for the debug pipeline endpoint."""

    file_id: UUID
    break_at: str = "generate"
    raw: bool = False


class DebugPipelineResponse(BaseModel):
    """Response for debug pipeline invocation."""

    stages: List[PipelineStageDiagnostics]


__all__ = [
    "Chunk",
    "EmbedVector",
    "RetrievalHit",
    "ChatRequest",
    "ChatResponse",
    "PipelineStageDiagnostics",
    "DebugPipelineRequest",
    "DebugPipelineResponse",
]
