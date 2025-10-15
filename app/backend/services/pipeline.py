"""Pipeline orchestration for ingestion and chat."""
from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import asdict
from typing import List, Sequence
from uuid import UUID, uuid4

from app.backend import exceptions
from app.backend.config import get_settings
from app.backend.models.chat import (
    ChatRequest,
    ChatResponse,
    Chunk,
    DebugChunkPayload,
    DebugPipelineRequest,
    DebugPipelineResponse,
    EmbedVector,
    PipelineStageDiagnostics,
    RetrievalHit,
)
from app.backend.models.ingestion import ExtractionResult, FileKind, TranscriptionResult
from app.backend.services.embeddings import (
    EmbeddingOperationError,
    EmbeddingRateLimitError,
    EmbeddingService,
)
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
from app.common.logging import json_log


logger = logging.getLogger(__name__)


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
        overall_start = time.perf_counter()
        extraction_start = time.perf_counter()
        extraction = await self._extraction.extract_pdf(file_id)
        extraction_ms = int((time.perf_counter() - extraction_start) * 1000)
        indexing = self._index_extracted_text(extraction, file_id)
        total_ms = int((time.perf_counter() - overall_start) * 1000)
        json_log(
            logger,
            logging.INFO,
            "ingestion.complete",
            ingestion_type="pdf",
            file_id=str(file_id),
            counts={
                "chunks": indexing["chunk_count"],
                "embeddings": indexing["embedding_count"],
                "tokens_total": indexing["token_total"],
                "tokens_avg": round(indexing["token_avg"], 2),
            },
            timings={
                "extraction_ms": extraction_ms,
                "chunk_ms": indexing["chunk_ms"],
                "embedding_ms": indexing["embedding_ms"],
                "vector_store_ms": indexing["vector_store_ms"],
                "total_ms": total_ms,
            },
            file_stats={"pages": extraction.pages},
        )
        return extraction

    async def handle_audio_upload(self, file_id: UUID) -> TranscriptionResult:
        overall_start = time.perf_counter()
        transcription_start = time.perf_counter()
        transcription = await self._extraction.transcribe_audio(file_id)
        transcription_ms = int((time.perf_counter() - transcription_start) * 1000)
        indexing = self._index_transcript(transcription, file_id)
        total_ms = int((time.perf_counter() - overall_start) * 1000)
        json_log(
            logger,
            logging.INFO,
            "ingestion.complete",
            ingestion_type="audio",
            file_id=str(file_id),
            counts={
                "chunks": indexing["chunk_count"],
                "embeddings": indexing["embedding_count"],
                "tokens_total": indexing["token_total"],
                "tokens_avg": round(indexing["token_avg"], 2),
            },
            timings={
                "transcription_ms": transcription_ms,
                "chunk_ms": indexing["chunk_ms"],
                "embedding_ms": indexing["embedding_ms"],
                "vector_store_ms": indexing["vector_store_ms"],
                "total_ms": total_ms,
            },
            file_stats={"duration_seconds": round(transcription.duration_seconds, 2)},
        )
        return transcription

    def _index_extracted_text(self, extraction: ExtractionResult, file_id: UUID) -> dict[str, float | int]:
        return self._index_text(extraction.text, file_id)

    def _index_transcript(self, transcription: TranscriptionResult, file_id: UUID) -> dict[str, float | int]:
        return self._index_text(transcription.transcript, file_id)

    def _chunk_text(self, text: str, file_id: UUID) -> tuple[List[Chunk], List[NormalizedChunk]]:
        config = ChunkConfig(chunk_size=self._settings.chunk_size, chunk_overlap=self._settings.chunk_overlap)
        normalized = chunk_text(text, config)
        chunks = [
            Chunk(text=piece.text, source_file_id=file_id, order=order)
            for order, piece in enumerate(normalized)
        ]
        return chunks, normalized

    def _index_text(self, text: str, file_id: UUID) -> dict[str, float | int]:
        chunk_start = time.perf_counter()
        chunks, normalized = self._chunk_text(text, file_id)
        chunk_ms = int((time.perf_counter() - chunk_start) * 1000)

        try:
            embed_start = time.perf_counter()
            vectors = self._embeddings.embed_chunks(chunks)
            embedding_ms = int((time.perf_counter() - embed_start) * 1000)
        except EmbeddingRateLimitError as exc:
            raise exceptions.rate_limit_exceeded() from exc
        except EmbeddingOperationError as exc:
            raise exceptions.embedding_error() from exc

        upsert_start = time.perf_counter()
        self._vector_store.upsert_vectors(chunks, vectors)
        vector_store_ms = int((time.perf_counter() - upsert_start) * 1000)

        chunk_count = len(chunks)
        embedding_count = len(vectors)
        token_total = sum(item.token_count for item in normalized)
        avg_tokens = (token_total / chunk_count) if chunk_count else 0.0

        return {
            "chunk_ms": chunk_ms,
            "embedding_ms": embedding_ms,
            "vector_store_ms": vector_store_ms,
            "chunk_count": chunk_count,
            "embedding_count": embedding_count,
            "token_total": token_total,
            "token_avg": avg_tokens,
        }

    async def chat(self, request: ChatRequest) -> ChatResponse:
        if not request.query.strip():
            raise exceptions.missing_query()
        context = self._sessions.get_or_create(request.session_id)
        session_id = context.session_id
        requested_top_k = request.top_k or self._settings.top_k
        top_k = max(1, min(requested_top_k, 8))
        start = time.perf_counter()

        retrieve_start = time.perf_counter()
        hits = await self._retrieve_with_timeout(
            request.query,
            top_k,
            allowed_source_ids=context.active_file_ids,
        )
        retrieval_ms = int((time.perf_counter() - retrieve_start) * 1000)

        self._sessions.associate_files(session_id, [hit.source_file_id for hit in hits])

        prompt_start = time.perf_counter()
        prompt = await self._build_prompt_with_timeout(request.query, hits)
        prompt_ms = int((time.perf_counter() - prompt_start) * 1000)

        generate_start = time.perf_counter()
        answer = await self._generate_with_guard(
            prompt=prompt,
            query=request.query,
            context=[hit.text for hit in hits],
        )
        generation_ms = int((time.perf_counter() - generate_start) * 1000)

        latency_ms = int((time.perf_counter() - start) * 1000)
        json_log(
            logger,
            logging.INFO,
            "chat.completed",
            session_id=str(session_id),
            query_preview=self._preview_text(request.query, limit=120),
            retrieval={
                "requested_top_k": top_k,
                "returned": len(hits),
                "precision_at_k": self._precision_sample(hits, top_k),
            },
            latency_breakdown={
                "total_ms": latency_ms,
                "retrieve_ms": retrieval_ms,
                "prompt_ms": prompt_ms,
                "generate_ms": generation_ms,
            },
        )
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
        except EmbeddingRateLimitError as exc:
            raise exceptions.rate_limit_exceeded() from exc
        except EmbeddingOperationError as exc:
            raise exceptions.embedding_error() from exc
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

    @staticmethod
    def _precision_sample(hits: Sequence[RetrievalHit], top_k: int) -> dict[str, object]:
        if not hits:
            return {"top_k": top_k, "estimated": 0.0, "threshold": 0.6, "scores": [], "sampled": 0}
        threshold = 0.6
        sample = hits[: max(1, min(top_k, len(hits)))]
        relevant = sum(1 for hit in sample if hit.score >= threshold)
        estimated = relevant / len(sample)
        return {
            "top_k": top_k,
            "threshold": threshold,
            "estimated": round(estimated, 3),
            "scores": [round(hit.score, 4) for hit in sample],
            "chunk_ids": [str(hit.chunk_id) for hit in sample],
            "sampled": len(sample),
        }

    def _build_probe_query(self, chunks: List[NormalizedChunk], fallback_text: str) -> str:
        if chunks:
            probe = chunks[0].text.strip()
            if probe:
                return probe[:200]
        cleaned = " ".join(fallback_text.strip().split())
        return cleaned[:120] or "debug pipeline probe"

    def _build_chunks_from_payload(
        self, payloads: Sequence[DebugChunkPayload], source_file_id: UUID
    ) -> tuple[List[Chunk], List[NormalizedChunk]]:
        normalized: List[NormalizedChunk] = []
        chunks: List[Chunk] = []
        start_token = 0
        for payload in sorted(payloads, key=lambda item: item.order):
            text = payload.resolved_text()
            if not text:
                continue
            token_count = len(text.split())
            end_token = start_token + token_count
            normalized.append(
                NormalizedChunk(
                    text=text,
                    token_count=token_count,
                    start_token=start_token,
                    end_token=end_token,
                )
            )
            chunks.append(
                Chunk(
                    chunk_id=payload.chunk_id,
                    text=text,
                    source_file_id=source_file_id,
                    order=payload.order,
                )
            )
            start_token = end_token
        return chunks, normalized

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

    async def debug_pipeline(
        self, request: DebugPipelineRequest, break_at: str, raw: bool
    ) -> DebugPipelineResponse:
        allowed_stages = ["extract", "chunk", "embed", "retrieve", "generate"]
        target_stage = break_at.strip().lower()
        if target_stage not in allowed_stages:
            raise exceptions.invalid_debug_stage()

        chunk_size = request.chunk_size or self._settings.chunk_size
        chunk_overlap = request.chunk_overlap or self._settings.chunk_overlap
        try:
            chunk_config = ChunkConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        except ValueError as exc:
            raise exceptions.invalid_request(hint=str(exc)) from exc

        stages: List[PipelineStageDiagnostics] = []
        provided_chunks = request.chunks or []
        sorted_chunk_payloads = sorted(provided_chunks, key=lambda item: item.order)
        inline_text = (request.text or "").strip()
        inline_query = (request.query or "").strip()

        if (
            request.file_id is None
            and not sorted_chunk_payloads
            and not inline_text
            and target_stage in {"retrieve", "generate"}
        ):
            if not inline_query:
                raise exceptions.invalid_request(hint="Query payload cannot be empty.")
            requested_top_k = request.top_k or self._settings.top_k
            top_k = max(1, min(requested_top_k, 8))
            hits = await self._retrieve_with_timeout(
                inline_query,
                top_k,
                allowed_source_ids=[],
            )
            retrieve_input = {"query": inline_query, "top_k": top_k}
            retrieve_hits: List[dict[str, object]] = []
            for hit in hits:
                entry: dict[str, object] = {
                    "chunk_id": str(hit.chunk_id),
                    "score": round(hit.score, 6),
                    "source_file_id": str(hit.source_file_id),
                }
                entry["text" if raw else "preview"] = (
                    hit.text if raw else self._preview_text(hit.text)
                )
                retrieve_hits.append(entry)
            stages.append(
                PipelineStageDiagnostics(
                    stage="retrieve",
                    input_payload=retrieve_input,
                    output_payload={"hits": retrieve_hits, "count": len(retrieve_hits)},
                )
            )
            if target_stage == "retrieve":
                return DebugPipelineResponse(stages=stages)
            prompt = await self._build_prompt_with_timeout(inline_query, hits)
            context = [hit.text for hit in hits]
            answer = await self._generate_with_guard(
                prompt=prompt,
                query=inline_query,
                context=context,
            )
            generate_input = {
                "query": inline_query,
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
                    stage="generate",
                    input_payload=generate_input,
                    output_payload=generate_output,
                )
            )
            return DebugPipelineResponse(stages=stages)

        if request.file_id is not None:
            metadata = self._storage.get_metadata(request.file_id)

            if metadata.kind == FileKind.PDF:
                extraction = await self._extraction.extract_pdf(request.file_id)
                extracted_text = extraction.text
                extract_output = extraction.model_dump()
                if not raw:
                    extract_output = {
                        "pages": extraction.pages,
                        "text": self._preview_text(extraction.text),
                    }
                extract_input: dict[str, object] = {
                    "file_id": str(request.file_id),
                    "file_type": metadata.kind.value,
                }
            else:
                transcription = await self._extraction.transcribe_audio(request.file_id)
                extracted_text = transcription.transcript
                extract_output = transcription.model_dump()
                if not raw:
                    extract_output = {
                        "duration_seconds": transcription.duration_seconds,
                        "transcript": self._preview_text(transcription.transcript),
                    }
                extract_input = {
                    "file_id": str(request.file_id),
                    "file_type": metadata.kind.value,
                    "language": "en",
                }
            if raw:
                extract_input["filename"] = metadata.filename
            stages.append(
                PipelineStageDiagnostics(
                    stage="extract", input_payload=extract_input, output_payload=extract_output
                )
            )
            if target_stage == "extract":
                return DebugPipelineResponse(stages=stages)
            extracted_source_id = request.file_id
        else:
            if target_stage != "chunk" and not sorted_chunk_payloads:
                raise exceptions.invalid_request(
                    hint="file_id or chunks payload is required to debug stages beyond chunk."
                )
            if not inline_text and not sorted_chunk_payloads:
                raise exceptions.invalid_request(hint="Text payload cannot be empty.")
            if inline_text:
                extracted_text = inline_text
            else:
                extracted_text = "\n\n".join(chunk.resolved_text() for chunk in sorted_chunk_payloads)
            extracted_source_id = uuid4()

        if (
            sorted_chunk_payloads
            and request.file_id is None
            and not inline_text
        ):
            chunks, normalized_chunks = self._build_chunks_from_payload(
                sorted_chunk_payloads, extracted_source_id
            )
        else:
            normalized_chunks = chunk_text(extracted_text, chunk_config)
            chunks = [
                Chunk(text=piece.text, source_file_id=extracted_source_id, order=order)
                for order, piece in enumerate(normalized_chunks)
            ]
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
            "chunk_size": chunk_config.chunk_size,
            "overlap": chunk_config.chunk_overlap,
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

        if request.file_id is None and not sorted_chunk_payloads:
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
