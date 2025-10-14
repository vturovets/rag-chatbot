"""Pipeline orchestration for ingestion and chat."""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import List
from uuid import UUID

from app.backend import exceptions
from app.backend.config import get_settings
from app.backend.models.chat import (
    ChatRequest,
    ChatResponse,
    Chunk,
    DebugPipelineResponse,
    PipelineStageDiagnostics,
    RetrievalHit,
)
from app.backend.models.ingestion import ExtractionResult, FileKind, TranscriptionResult
from app.backend.services.embeddings import EmbeddingService
from app.backend.services.extraction import ExtractionService
from app.backend.services.session_store import SessionStore
from app.backend.services.storage import FileStorage
from app.backend.services.vector_store import VectorStore
from app.common.chunking import ChunkConfig, NormalizedChunk, chunk_text


class PipelineService:
    """Coordinate ingestion, retrieval, and generation."""

    def __init__(
        self,
        storage: FileStorage | None = None,
        extraction: ExtractionService | None = None,
        embeddings: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
        session_store: SessionStore | None = None,
    ) -> None:
        self._settings = get_settings()
        self._storage = storage or FileStorage()
        self._extraction = extraction or ExtractionService(self._storage)
        self._embeddings = embeddings or EmbeddingService()
        self._vector_store = vector_store or VectorStore(self._embeddings)
        self._sessions = session_store or SessionStore()

    async def handle_pdf_upload(self, file_id: UUID) -> ExtractionResult:
        extraction = await self._extraction.extract_pdf(file_id)
        self._index_extracted_text(extraction, file_id)
        return extraction

    async def handle_audio_upload(self, file_id: UUID) -> TranscriptionResult:
        transcription = await self._extraction.transcribe_audio(file_id)
        self._index_transcript(transcription, file_id)
        return transcription

    def _index_extracted_text(self, extraction: ExtractionResult, file_id: UUID) -> None:
        chunks, _ = self._chunk_text(extraction.text, file_id)
        self._vector_store.upsert(chunks)

    def _index_transcript(self, transcription: TranscriptionResult, file_id: UUID) -> None:
        chunks, _ = self._chunk_text(transcription.transcript, file_id)
        self._vector_store.upsert(chunks)

    def _chunk_text(self, text: str, file_id: UUID) -> tuple[List[Chunk], List[NormalizedChunk]]:
        config = ChunkConfig(chunk_size=self._settings.chunk_size, chunk_overlap=self._settings.chunk_overlap)
        normalized = chunk_text(text, config)
        chunks = [
            Chunk(text=piece.text, source_file_id=file_id, order=order)
            for order, piece in enumerate(normalized)
        ]
        return chunks, normalized

    async def chat(self, request: ChatRequest) -> ChatResponse:
        if not request.query.strip():
            raise exceptions.missing_query()
        context = self._sessions.get_or_create(request.session_id)
        session_id = context.session_id
        top_k = request.top_k or self._settings.top_k
        start = time.perf_counter()
        hits = self._vector_store.similarity_search(
            request.query,
            top_k,
            allowed_source_ids=context.active_file_ids,
        )
        if not hits:
            hits = self._vector_store.similarity_search(request.query, top_k)
        self._sessions.associate_files(session_id, [hit.source_file_id for hit in hits])
        answer = self._generate_answer(hits)
        latency_ms = int((time.perf_counter() - start) * 1000)
        return ChatResponse(answer=answer, latency_ms=latency_ms, session_id=session_id)

    def _generate_answer(self, hits: List[RetrievalHit]) -> str:
        contexts = [hit.text for hit in hits]
        if not contexts:
            return "I could not find relevant information for that question yet."
        prompt = " ".join(contexts)
        return f"Based on the uploaded materials, here is what I found: {prompt[:500]}"

    async def debug_pipeline(self, file_id: UUID, break_at: str, raw: bool) -> DebugPipelineResponse:
        stages: List[PipelineStageDiagnostics] = []
        metadata = self._storage.get_metadata(file_id)

        if metadata.kind == FileKind.PDF:
            extraction = await self._extraction.extract_pdf(file_id)
            extraction_text = extraction.text
        else:
            extraction = await self._extraction.transcribe_audio(file_id)
            extraction_text = extraction.transcript
        stages.append(
            PipelineStageDiagnostics(
                stage="extract",
                input_payload={"file_id": str(file_id), "file_type": metadata.kind.value},
                output_payload=extraction.model_dump(),
            )
        )
        if break_at == "extract":
            return DebugPipelineResponse(stages=stages)

        chunks, normalized_chunks = self._chunk_text(extraction_text, file_id)
        chunk_stats = {
            "count": len(normalized_chunks),
            "total_tokens": sum(item.token_count for item in normalized_chunks),
        }
        if chunk_stats["count"]:
            chunk_stats["avg_tokens"] = chunk_stats["total_tokens"] / chunk_stats["count"]
        stages.append(
            PipelineStageDiagnostics(
                stage="chunk",
                input_payload={"chunk_size": self._settings.chunk_size, "overlap": self._settings.chunk_overlap},
                output_payload=
                chunk_stats
                if not raw
                else {
                    "chunks": [chunk.model_dump() for chunk in chunks],
                    "token_windows": [asdict(item) for item in normalized_chunks],
                },
            )
        )
        if break_at == "chunk":
            return DebugPipelineResponse(stages=stages)

        vectors = self._embeddings.embed_chunks(chunks)
        stages.append(
            PipelineStageDiagnostics(
                stage="embed",
                input_payload={"fingerprint": self._embeddings.index_fingerprint()},
                output_payload={"count": len(vectors), "dimension": len(vectors[0].values) if vectors else 0},
            )
        )
        if break_at == "embed":
            return DebugPipelineResponse(stages=stages)

        self._vector_store.upsert(chunks)
        hits = self._vector_store.similarity_search("debug", self._settings.top_k)
        stages.append(
            PipelineStageDiagnostics(
                stage="retrieve",
                input_payload={"query": "debug", "top_k": self._settings.top_k},
                output_payload={"count": len(hits)} if not raw else {"hits": [hit.model_dump() for hit in hits]},
            )
        )
        if break_at == "retrieve":
            return DebugPipelineResponse(stages=stages)

        answer = self._generate_answer(hits)
        stages.append(
            PipelineStageDiagnostics(
                stage="generate",
                input_payload={"query": "debug"},
                output_payload={"answer": answer},
            )
        )
        return DebugPipelineResponse(stages=stages)


__all__ = ["PipelineService"]
