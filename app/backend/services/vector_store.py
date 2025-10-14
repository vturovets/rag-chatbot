"""ChromaDB-backed vector store adhering to the project requirements."""
from __future__ import annotations

from pathlib import Path
import threading
from typing import Any, Dict, Iterable, List, Optional
from uuid import UUID

try:  # pragma: no cover - import guard for optional dependency
    import chromadb
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection
    from chromadb.config import Settings as ChromaSettings
except ModuleNotFoundError:  # pragma: no cover - executed when chromadb is missing
    chromadb = None
    ClientAPI = Any  # type: ignore[assignment]
    Collection = Any  # type: ignore[assignment]
    ChromaSettings = None  # type: ignore[assignment]

from app.backend.config import get_settings
from app.backend.models.chat import Chunk, RetrievalHit
from app.backend.services.embeddings import EmbeddingService


class VectorStore:
    """Repository abstraction over ChromaDB collections.

    Each collection is keyed by the embedding provider/model fingerprint to satisfy the
    "index fingerprint" requirement in the SRS. All operations are guarded by a lock to
    ensure thread-safety for concurrent ingestion and retrieval across sessions.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        *,
        persist_directory: Path | None = None,
        client: ClientAPI | None = None,
    ) -> None:
        self._embedding_service = embedding_service or EmbeddingService()
        self._settings = get_settings()
        self._lock = threading.RLock()

        self._use_in_memory = chromadb is None and client is None
        self._collections: Dict[str, Collection] = {}
        self._dimensions: Dict[str, int] = {}
        self._memory_entries: Dict[str, Dict[UUID, tuple[Chunk, List[float]]]] = {}

        if self._use_in_memory:
            self._client = None
        elif client is not None:
            self._client = client
        else:
            persist_directory = persist_directory or self._default_persist_directory()
            persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = self._create_client(persist_directory)

    def _default_persist_directory(self) -> Path:
        return self._settings.storage_dir / "chroma"

    @staticmethod
    def _create_client(path: Path) -> ClientAPI:
        if chromadb is None:
            raise RuntimeError(
                "chromadb package is required for vector store operations; install it via 'pip install chromadb'."
            )
        if ChromaSettings is not None:
            return chromadb.PersistentClient(
                path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
            )
        return chromadb.PersistentClient(path=str(path))

    @staticmethod
    def _collection_name(fingerprint: str) -> str:
        # Chroma collection names must be alphanumeric with limited symbols. Replace
        # separators to guarantee compatibility while keeping uniqueness.
        safe = fingerprint.replace(":", "_").replace("/", "-")
        return f"rag_{safe}"

    def _get_collection(self, fingerprint: str, dimension: int) -> Collection:
        with self._lock:
            collection = self._collections.get(fingerprint)
            if collection is None:
                metadata = {"fingerprint": fingerprint, "dimension": str(dimension)}
                collection = self._client.get_or_create_collection(
                    name=self._collection_name(fingerprint),
                    metadata=metadata,
                )
                self._collections[fingerprint] = collection
                self._dimensions[fingerprint] = dimension
                return collection

            expected_dimension = self._dimensions.get(fingerprint)
            if expected_dimension is None:
                stored_dimension = collection.metadata.get("dimension") if collection.metadata else None
                if stored_dimension is not None:
                    try:
                        expected_dimension = int(stored_dimension)
                    except (TypeError, ValueError):
                        expected_dimension = None
            if expected_dimension is not None and expected_dimension != dimension:
                raise ValueError(
                    f"Vector dimension mismatch for collection with fingerprint {fingerprint}"
                )
            self._dimensions[fingerprint] = dimension
            return collection

    def upsert(self, chunks: Iterable[Chunk]) -> None:
        chunk_list = [chunk for chunk in chunks if chunk.text.strip()]
        if not chunk_list:
            return

        vectors = self._embedding_service.embed_chunks(chunk_list)
        if not vectors:
            return

        dimension = len(vectors[0].values)
        if any(len(vector.values) != dimension for vector in vectors):
            raise ValueError("Embedding vectors have inconsistent dimensions")

        fingerprint = self._embedding_service.index_fingerprint()
        if self._use_in_memory:
            self._upsert_in_memory(fingerprint, chunk_list, vectors, dimension)
            return

        collection = self._get_collection(fingerprint, dimension)

        ids = [str(chunk.chunk_id) for chunk in chunk_list]
        documents = [chunk.text for chunk in chunk_list]
        metadatas = [
            {"chunk_id": str(chunk.chunk_id), "source_file_id": str(chunk.source_file_id), "order": chunk.order}
            for chunk in chunk_list
        ]
        embeddings = [vector.values for vector in vectors]

        with self._lock:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)

    def similarity_search(
        self,
        query: str,
        top_k: int,
        *,
        allowed_source_ids: Optional[Iterable[UUID]] = None,
    ) -> List[RetrievalHit]:
        if top_k <= 0:
            return []

        fingerprint = self._embedding_service.index_fingerprint()
        collection: Collection | None = None
        if not self._use_in_memory:
            with self._lock:
                collection = self._collections.get(fingerprint)
                if collection is None:
                    try:
                        collection = self._client.get_collection(name=self._collection_name(fingerprint))
                    except ValueError:
                        return []
                    self._collections[fingerprint] = collection
                    stored_dimension = collection.metadata.get("dimension") if collection.metadata else None
                    if stored_dimension:
                        try:
                            self._dimensions[fingerprint] = int(stored_dimension)
                        except (TypeError, ValueError):
                            pass

        query_vector = self._embedding_service.embed_query(query)
        if not query_vector:
            return []

        if self._use_in_memory:
            return self._similarity_search_in_memory(fingerprint, query_vector, top_k, allowed_source_ids)

        where_clause = None
        if allowed_source_ids:
            identifiers = [str(file_id) for file_id in allowed_source_ids]
            if identifiers:
                where_clause = {"source_file_id": {"$in": identifiers}}

        try:
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where_clause,
                include=["distances", "documents", "metadatas"],
            )
        except ValueError:
            return []

        distances = results.get("distances", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        hits: List[RetrievalHit] = []
        for distance, document, metadata in zip(distances, documents, metadatas):
            if metadata is None:
                metadata = {}
            chunk_id = metadata.get("chunk_id")
            source_file_id = metadata.get("source_file_id")
            if chunk_id is None or source_file_id is None:
                continue
            try:
                chunk_uuid = UUID(chunk_id)
                source_uuid = UUID(source_file_id)
            except (TypeError, ValueError):
                continue
            similarity = 1 - float(distance) if distance is not None else 0.0
            hits.append(
                RetrievalHit(
                    chunk_id=chunk_uuid,
                    score=similarity,
                    text=document or "",
                    source_file_id=source_uuid,
                )
            )

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

    def remove_by_source_ids(self, source_file_ids: Iterable[UUID]) -> None:
        identifiers = [str(file_id) for file_id in source_file_ids]
        if not identifiers:
            return
        if self._use_in_memory:
            self._remove_in_memory(identifiers)
            return
        with self._lock:
            for collection in self._collections.values():
                collection.delete(where={"source_file_id": {"$in": identifiers}})

    def _upsert_in_memory(
        self,
        fingerprint: str,
        chunk_list: List[Chunk],
        vectors: List[Any],
        dimension: int,
    ) -> None:
        with self._lock:
            store = self._memory_entries.setdefault(fingerprint, {})
            self._dimensions[fingerprint] = dimension
            for chunk, vector in zip(chunk_list, vectors):
                store[chunk.chunk_id] = (chunk, list(vector.values))

    def _similarity_search_in_memory(
        self,
        fingerprint: str,
        query_vector: List[float],
        top_k: int,
        allowed_source_ids: Optional[Iterable[UUID]],
    ) -> List[RetrievalHit]:
        store = self._memory_entries.get(fingerprint)
        if not store:
            return []
        allowed = set(allowed_source_ids or [])
        hits: List[RetrievalHit] = []
        for chunk, vector in store.values():
            if allowed and chunk.source_file_id not in allowed:
                continue
            similarity = self._cosine_similarity(query_vector, vector)
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    score=similarity,
                    text=chunk.text,
                    source_file_id=chunk.source_file_id,
                )
            )
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:top_k]

    def _remove_in_memory(self, identifiers: List[str]) -> None:
        with self._lock:
            targets = {UUID(identifier) for identifier in identifiers}
            for store in self._memory_entries.values():
                for chunk_id, (chunk, _) in list(store.items()):
                    if chunk.source_file_id in targets:
                        store.pop(chunk_id, None)

    @staticmethod
    def _cosine_similarity(left: List[float], right: List[float]) -> float:
        dot = sum(l * r for l, r in zip(left, right))
        left_norm = (sum(l * l for l in left) ** 0.5) or 1.0
        right_norm = (sum(r * r for r in right) ** 0.5) or 1.0
        return dot / (left_norm * right_norm)


__all__ = ["VectorStore"]
