"""Pipeline orchestration for ingestion and chat."""
from __future__ import annotations

import asyncio
import math
import time
from dataclasses import asdict
from typing import List, Sequence
from uuid import UUID

from app.backend import exceptions
from app.backend.config import get_settings
from app.backend.models.chat import (
    ChatRequest,
    ChatResponse,
    Chunk,
    DebugPipelineResponse,
    EmbedVector,
    PipelineStageDiagnostics,
    RetrievalHit,
)
from app.backend.models.ingestion import ExtractionResult, FileKind, TranscriptionResult
from app.backend.services.embeddings import EmbeddingService
from app.backend.services.extraction import ExtractionService
from app.backend.services.generation import (
    GenerationError,
    GenerationService,
    GenerationTimeoutError,
    ProviderUnavailableError,
    RateLimitedError,
)
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
        generation: GenerationService | None = None,
    ) -> None:
        self._settings = get_settings()
        self._storage = storage or FileStorage()
        self._extraction = extraction or ExtractionService(self._storage)
        self._embeddings = embeddings or EmbeddingService()
        self._vector_store = vector_store or VectorStore(self._embeddings)
        self._sessions = session_store or SessionStore()
        self._generation = generation or GenerationService()

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
        requested_top_k = request.top_k or self._settings.top_k
        top_k = max(1, min(requested_top_k, 8))
        start = time.perf_counter()

        hits = await self._retrieve_with_timeout(
            request.query,
            top_k,
            allowed_source_ids=context.active_file_ids,
        )

        self._sessions.associate_files(session_id, [hit.source_file_id for hit in hits])

        prompt = await self._build_prompt_with_timeout(request.query, hits)

        answer = await self._generate_with_guard(
            prompt=prompt,
            query=request.query,
            context=[hit.text for hit in hits],
        )

        latency_ms = int((time.perf_counter() - start) * 1000)
        return ChatResponse(answer=answer, latency_ms=latency_ms, session_id=session_id)

    async def _retrieve_with_timeout(
        self,
        query: str,
        top_k: int,
        *,
        allowed_source_ids: Sequence[UUID],
    ) -> List[RetrievalHit]:
        def _search() -> List[RetrievalHit]:
            hits = self._vector_store.similarity_search(
                query,
                top_k,
                allowed_source_ids=allowed_source_ids,
            )
            if not hits:
                hits = self._vector_store.similarity_search(query, top_k)
            return hits

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_search), timeout=self._settings.retrieval_timeout_s
            )
        except asyncio.TimeoutError as exc:
            raise exceptions.timeout_stage("Processing timed out while retrieving context.") from exc
        except Exception as exc:
            raise exceptions.vector_db_unavailable() from exc

    async def _build_prompt_with_timeout(self, query: str, hits: List[RetrievalHit]) -> str:
        def _build() -> str:
            snippets = []
            for hit in hits:
                text = " ".join(hit.text.strip().split()) if hit.text else ""
                if text:
                    snippets.append(text)
            context_block = "\n".join(f"- {snippet}" for snippet in snippets[:8])
            if not context_block:
                return (
                    "You are assisting with retrieval-augmented questions but no supporting context was found.\n"
                    f"Question: {query.strip()}\n"
                    "Answer concisely and acknowledge the lack of context."
                )
            instructions = (
                "You are a retrieval-augmented assistant. Use only the provided context to answer "
                "the question. Respond with a concise summary (max three sentences) and do not "
                "include citations or references."
            )
            return f"{instructions}\n\nContext:\n{context_block}\n\nQuestion: {query.strip()}\nAnswer:"

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_build), timeout=self._settings.prompt_timeout_s
            )
        except asyncio.TimeoutError as exc:
            raise exceptions.timeout_stage("Processing timed out while preparing the prompt.") from exc

    async def _generate_with_guard(self, *, prompt: str, query: str, context: Sequence[str]) -> str:
        try:
            return await self._generation.generate(
                prompt=prompt,
                query=query,
                context=context,
                timeout=self._settings.generation_timeout_s,
            )
        except GenerationTimeoutError as exc:
            raise exceptions.generation_timeout() from exc
        except ProviderUnavailableError as exc:
            raise exceptions.llm_provider_down() from exc
        except RateLimitedError as exc:
            raise exceptions.rate_limit_exceeded() from exc
        except GenerationError as exc:
            raise exceptions.internal_error() from exc

    @staticmethod
    def _preview_text(text: str, limit: int = 240) -> str:
        cleaned = " ".join(text.strip().split())
        if len(cleaned) <= limit:
            return cleaned
        return f"{cleaned[:limit].rstrip()}..."

    def _build_probe_query(self, chunks: List[NormalizedChunk], fallback_text: str) -> str:
        if chunks:
            probe = chunks[0].text.strip()
            if probe:
                return probe[:200]
        cleaned = " ".join(fallback_text.strip().split())
        return cleaned[:120] or "debug pipeline probe"

    def _rank_hits(
        self,
        query_vector: Sequence[float],
        vectors: Sequence[EmbedVector],
        chunks: Sequence[Chunk],
        *,
        top_k: int,
    ) -> List[RetrievalHit]:
        if not query_vector or not vectors:
            return []
        chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
        scored: List[tuple[float, Chunk]] = []
        for vector in vectors:
            chunk = chunk_lookup.get(vector.chunk_id)
            if chunk is None:
                continue
            score = self._cosine_similarity(query_vector, vector.values)
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        hits: List[RetrievalHit] = []
        for score, chunk in scored[: max(1, top_k)]:
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    score=score,
                    text=chunk.text,
                    source_file_id=chunk.source_file_id,
                )
            )
        return hits

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        if not left or not right:
            return 0.0
        dot = sum(l * r for l, r in zip(left, right))
        left_norm = math.sqrt(sum(l * l for l in left)) or 1.0
        right_norm = math.sqrt(sum(r * r for r in right)) or 1.0
        return dot / (left_norm * right_norm)

    async def debug_pipeline(self, file_id: UUID, break_at: str, raw: bool) -> DebugPipelineResponse:
        allowed_stages = ["extract", "chunk", "embed", "retrieve", "generate"]
        target_stage = break_at.strip().lower()
        if target_stage not in allowed_stages:
            raise exceptions.invalid_debug_stage()

        stages: List[PipelineStageDiagnostics] = []
        metadata = self._storage.get_metadata(file_id)

        if metadata.kind == FileKind.PDF:
            extraction = await self._extraction.extract_pdf(file_id)
            extracted_text = extraction.text
            extract_output = extraction.model_dump()
            if not raw:
                extract_output = {
                    "pages": extraction.pages,
                    "text": self._preview_text(extraction.text),
                }
            extract_input: dict[str, object] = {
                "file_id": str(file_id),
                "file_type": metadata.kind.value,
            }
        else:
            transcription = await self._extraction.transcribe_audio(file_id)
            extracted_text = transcription.transcript
            extract_output = transcription.model_dump()
            if not raw:
                extract_output = {
                    "duration_seconds": transcription.duration_seconds,
                    "transcript": self._preview_text(transcription.transcript),
                }
            extract_input = {
                "file_id": str(file_id),
                "file_type": metadata.kind.value,
                "language": "en",
            }
        if raw:
            extract_input["filename"] = metadata.filename
        stages.append(
            PipelineStageDiagnostics(stage="extract", input_payload=extract_input, output_payload=extract_output)
        )
        if target_stage == "extract":
            return DebugPipelineResponse(stages=stages)

        chunks, normalized_chunks = self._chunk_text(extracted_text, file_id)
        total_tokens = sum(item.token_count for item in normalized_chunks)
        chunk_total = len(normalized_chunks)
        chunk_counts: dict[str, float | int] = {
            "chunks": chunk_total,
            "total_tokens": total_tokens,
        }
        if chunk_total:
            chunk_counts["avg_tokens"] = total_tokens / chunk_total
        chunk_sample = chunks if raw else chunks[: min(5, len(chunks))]
        chunk_items: List[dict[str, object]] = []
        for chunk in chunk_sample:
            item: dict[str, object] = {
                "chunk_id": str(chunk.chunk_id),
                "order": chunk.order,
            }
            item["text" if raw else "preview"] = chunk.text if raw else self._preview_text(chunk.text)
            chunk_items.append(item)
        chunk_output: dict[str, object] = {"counts": chunk_counts, "chunks": chunk_items}
        if raw:
            chunk_output["token_windows"] = [asdict(item) for item in normalized_chunks]
        chunk_input: dict[str, object] = {
            "chunk_size": self._settings.chunk_size,
            "overlap": self._settings.chunk_overlap,
        }
        if raw:
            chunk_input["text"] = extracted_text
        else:
            chunk_input["text_preview"] = self._preview_text(extracted_text)
        stages.append(
            PipelineStageDiagnostics(stage="chunk", input_payload=chunk_input, output_payload=chunk_output)
        )
        if target_stage == "chunk":
            return DebugPipelineResponse(stages=stages)

        vectors = self._embeddings.embed_chunks(chunks)
        fingerprint = self._embeddings.index_fingerprint()
        embed_input = {
            "chunks": [
                {
                    "chunk_id": str(chunk.chunk_id),
                    "order": chunk.order,
                }
                for chunk in chunk_sample
            ]
        }
        embed_output: dict[str, object] = {
            "vectors": {
                "count": len(vectors),
                "dim": len(vectors[0].values) if vectors else 0,
            },
            "index_fingerprint": fingerprint,
        }
        if raw:
            embed_output["vectors"] = {
                "count": len(vectors),
                "dim": len(vectors[0].values) if vectors else 0,
                "items": [
                    {"chunk_id": str(vector.chunk_id), "values": vector.values}
                    for vector in vectors
                ],
            }
            embed_output["index_fingerprint"] = fingerprint
        stages.append(
            PipelineStageDiagnostics(stage="embed", input_payload=embed_input, output_payload=embed_output)
        )
        if target_stage == "embed":
            return DebugPipelineResponse(stages=stages)

        probe_query = self._build_probe_query(normalized_chunks, extracted_text)
        top_k = max(1, min(self._settings.top_k, 8))
        query_vector = self._embeddings.embed_query(probe_query)
        hits = self._rank_hits(query_vector, vectors, chunks, top_k=top_k)
        retrieve_input = {"query": probe_query, "top_k": top_k}
        retrieve_hits: List[dict[str, object]] = []
        for hit in hits:
            entry: dict[str, object] = {
                "chunk_id": str(hit.chunk_id),
                "score": round(hit.score, 6),
                "source_file_id": str(hit.source_file_id),
            }
            entry["text" if raw else "preview"] = hit.text if raw else self._preview_text(hit.text)
            retrieve_hits.append(entry)
        retrieve_output = {"hits": retrieve_hits, "count": len(retrieve_hits)}
        stages.append(
            PipelineStageDiagnostics(stage="retrieve", input_payload=retrieve_input, output_payload=retrieve_output)
        )
        if target_stage == "retrieve":
            return DebugPipelineResponse(stages=stages)

        prompt = await self._build_prompt_with_timeout(probe_query, hits)
        context = [hit.text for hit in hits]
        answer = await self._generate_with_guard(prompt=prompt, query=probe_query, context=context)
        generate_input = {
            "query": probe_query,
            "context_ids": [str(hit.chunk_id) for hit in hits],
        }
        generate_output: dict[str, object] = {
            "prompt": prompt if raw else self._preview_text(prompt, limit=320),
            "answer": answer,
        }
        if raw:
            generate_output["context"] = context
        stages.append(
            PipelineStageDiagnostics(
                stage="generate", input_payload=generate_input, output_payload=generate_output
            )
        )
        return DebugPipelineResponse(stages=stages)


__all__ = ["PipelineService"]
