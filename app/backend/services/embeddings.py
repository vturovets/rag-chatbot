"""Embedding generation services."""
from __future__ import annotations

import hashlib
import math
from typing import Iterable, List

from app.backend.config import get_settings
from app.backend.models.chat import Chunk, EmbedVector


def _hash_to_unit_vector(text: str, dim: int = 1536) -> List[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = []
    for i in range(dim):
        byte = digest[i % len(digest)]
        normalized = (byte / 255.0) * 2 - 1
        values.append(normalized)
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


class EmbeddingService:
    """Generate embeddings for chunks using deterministic hashing."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._dimension = 1536

    def index_fingerprint(self) -> str:
        return f"{self._settings.llm_provider}:{self._settings.embedding_model}:{self._dimension}"

    def embed_chunks(self, chunks: Iterable[Chunk]) -> List[EmbedVector]:
        vectors = []
        for chunk in chunks:
            vector = EmbedVector(chunk_id=chunk.chunk_id, values=_hash_to_unit_vector(chunk.text, self._dimension))
            vectors.append(vector)
        return vectors

    def embed_query(self, query: str) -> List[float]:
        return _hash_to_unit_vector(query, self._dimension)


__all__ = ["EmbeddingService"]
