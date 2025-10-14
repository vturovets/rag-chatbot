"""In-memory vector store compatible with the SRS interface."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List
from uuid import UUID

from app.backend.models.chat import Chunk, RetrievalHit
from app.backend.services.embeddings import EmbeddingService


class VectorStore:
    """A minimal in-memory approximation of ChromaDB collections."""

    def __init__(self, embedding_service: EmbeddingService | None = None) -> None:
        self._embedding_service = embedding_service or EmbeddingService()
        self._store: Dict[str, Dict[UUID, tuple[Chunk, List[float]]]] = defaultdict(dict)

    def _collection(self) -> Dict[UUID, tuple[Chunk, List[float]]]:
        fingerprint = self._embedding_service.index_fingerprint()
        return self._store[fingerprint]

    def upsert(self, chunks: Iterable[Chunk]) -> None:
        chunk_list = list(chunks)
        vectors = self._embedding_service.embed_chunks(chunk_list)
        collection = self._collection()
        for chunk, vector in zip(chunk_list, vectors):
            collection[chunk.chunk_id] = (chunk, vector.values)

    def similarity_search(self, query: str, top_k: int) -> List[RetrievalHit]:
        query_vector = self._embedding_service.embed_query(query)
        collection = self._collection()
        hits: List[RetrievalHit] = []
        for chunk, vector in collection.values():
            score = self._cosine_similarity(query_vector, vector)
            hits.append(RetrievalHit(chunk_id=chunk.chunk_id, score=score, text=chunk.text))
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

    @staticmethod
    def _cosine_similarity(left: List[float], right: List[float]) -> float:
        dot = sum(l * r for l, r in zip(left, right))
        left_norm = math.sqrt(sum(l * l for l in left)) or 1.0
        right_norm = math.sqrt(sum(r * r for r in right)) or 1.0
        return dot / (left_norm * right_norm)


__all__ = ["VectorStore"]
