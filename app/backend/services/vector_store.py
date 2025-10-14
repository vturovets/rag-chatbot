"""In-memory vector store compatible with the SRS interface."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List
from uuid import UUID

from app.backend.models.chat import Chunk, RetrievalHit
from app.backend.services.embeddings import EmbeddingService


@dataclass
class _Collection:
    dimension: int
    entries: Dict[UUID, tuple[Chunk, List[float]]]


class VectorStore:
    """A minimal in-memory approximation of ChromaDB collections using cosine metric."""

    def __init__(self, embedding_service: EmbeddingService | None = None) -> None:
        self._embedding_service = embedding_service or EmbeddingService()
        self._store: Dict[str, _Collection] = {}

    def _get_collection(self, fingerprint: str, dimension: int) -> _Collection:
        collection = self._store.get(fingerprint)
        if collection is None:
            collection = _Collection(dimension=dimension, entries={})
            self._store[fingerprint] = collection
            return collection
        if collection.dimension != dimension:
            raise ValueError(
                f"Vector dimension mismatch for collection with fingerprint {fingerprint}"
            )
        return collection

    def upsert(self, chunks: Iterable[Chunk]) -> None:
        chunk_list = list(chunks)
        if not chunk_list:
            return
        vectors = self._embedding_service.embed_chunks(chunk_list)
        if not vectors:
            return
        dimension = len(vectors[0].values)
        if any(len(vector.values) != dimension for vector in vectors):
            raise ValueError("Embedding vectors have inconsistent dimensions")
        fingerprint = self._embedding_service.index_fingerprint()
        collection = self._get_collection(fingerprint, dimension)
        for chunk, vector in zip(chunk_list, vectors):
            collection.entries[chunk.chunk_id] = (chunk, vector.values)

    def similarity_search(self, query: str, top_k: int) -> List[RetrievalHit]:
        fingerprint = self._embedding_service.index_fingerprint()
        collection = self._store.get(fingerprint)
        if not collection:
            return []
        query_vector = self._embedding_service.embed_query(query)
        hits: List[RetrievalHit] = []
        for chunk, vector in collection.entries.values():
            score = self._cosine_similarity(query_vector, vector)
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    score=score,
                    text=chunk.text,
                    source_file_id=chunk.source_file_id,
                )
            )
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

    @staticmethod
    def _cosine_similarity(left: List[float], right: List[float]) -> float:
        dot = sum(l * r for l, r in zip(left, right))
        left_norm = math.sqrt(sum(l * l for l in left)) or 1.0
        right_norm = math.sqrt(sum(r * r for r in right)) or 1.0
        return dot / (left_norm * right_norm)


__all__ = ["VectorStore"]
