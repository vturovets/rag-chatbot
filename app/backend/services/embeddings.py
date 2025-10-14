"""Embedding generation services supporting multiple providers."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, List, Protocol

from app.backend.config import get_settings
from app.backend.models.chat import Chunk, EmbedVector


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    @property
    def provider_name(self) -> str:
        ...

    @property
    def model_name(self) -> str:
        ...

    @property
    def dimension(self) -> int:
        ...

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...


class ProviderConfigurationError(RuntimeError):
    """Raised when a provider cannot be initialised due to configuration issues."""


def _hash_to_unit_vector(text: str, dim: int = 1536) -> List[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = []
    for i in range(dim):
        byte = digest[i % len(digest)]
        normalized = (byte / 255.0) * 2 - 1
        values.append(normalized)
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


@dataclass
class LocalHashProvider:
    """Deterministic fallback provider used when external APIs are unavailable."""

    model_name: str
    dimension: int = 1536
    provider_name: str = "local"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [_hash_to_unit_vector(text, self.dimension) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return _hash_to_unit_vector(text, self.dimension)


class OpenAIEmbeddingProvider:
    """Embedding provider backed by OpenAI's embeddings API."""

    def __init__(self, model_name: str, api_key: str | None, api_base: str | None) -> None:
        if not api_key:
            raise ProviderConfigurationError("OpenAI API key is not configured")
        from openai import OpenAI

        client_kwargs = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base
        self._client = OpenAI(**client_kwargs)
        self._model_name = model_name
        self._dimension: int | None = None

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # Default to the largest expected dimension when unknown.
            return 1536
        return self._dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model_name, input=texts)
        vectors: List[List[float]] = []
        for text, item in zip(texts, response.data):
            vector = list(item.embedding)
            if not vector:
                base_dimension = self._dimension or 1536
                vector = _hash_to_unit_vector(text, base_dimension)
            vectors.append(vector)
        if vectors:
            self._dimension = len(vectors[0])
        return vectors

    def embed_query(self, text: str) -> List[float]:
        vector = self.embed_documents([text])
        return vector[0] if vector else []


class GoogleAIVectorProvider:
    """Embedding provider backed by Google Generative AI embeddings."""

    def __init__(self, model_name: str, api_key: str | None) -> None:
        if not api_key:
            raise ProviderConfigurationError("Google API key is not configured")
        try:
            import google.generativeai as genai
        except ModuleNotFoundError as exc:  # pragma: no cover - import guard
            raise ProviderConfigurationError("google-generativeai package is required") from exc

        genai.configure(api_key=api_key)
        self._model_name = model_name
        self._dimension: int | None = None
        self._client = genai

    @property
    def provider_name(self) -> str:
        return "google"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            return 768
        return self._dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            response = self._client.embed_content(model=self._model_name, content=text)
            vector = response.get("embedding", [])
            if not vector:
                base_dimension = self._dimension or 768
                vector = _hash_to_unit_vector(text, base_dimension)
            embeddings.append(vector)
            if vector and self._dimension is None:
                self._dimension = len(vector)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = self._client.embed_content(model=self._model_name, content=text)
        vector = response.get("embedding", [])
        if not vector:
            base_dimension = self._dimension or 768
            vector = _hash_to_unit_vector(text, base_dimension)
        if vector and self._dimension is None:
            self._dimension = len(vector)
        return vector


class EmbeddingService:
    """Generate embeddings for chunks using configured provider."""

    def __init__(self, provider: EmbeddingProvider | None = None) -> None:
        self._settings = get_settings()
        self._provider = provider or self._resolve_provider()

    def _resolve_provider(self) -> EmbeddingProvider:
        provider_name = self._settings.llm_provider.lower()
        model = self._settings.embedding_model
        try:
            if provider_name == "openai":
                return OpenAIEmbeddingProvider(model, self._settings.openai_api_key, self._settings.openai_api_base)
            if provider_name == "google":
                return GoogleAIVectorProvider(model, self._settings.google_api_key)
        except ProviderConfigurationError:
            if self._settings.environment == "prod":
                raise
        return LocalHashProvider(model_name=model)

    @property
    def vector_dimension(self) -> int:
        return self._provider.dimension

    def index_fingerprint(self) -> str:
        return f"{self._provider.provider_name}:{self._provider.model_name}:{self.vector_dimension}"

    def embed_chunks(self, chunks: Iterable[Chunk]) -> List[EmbedVector]:
        chunk_list = [chunk for chunk in chunks if chunk.text.strip()]
        if not chunk_list:
            return []
        vectors = self._provider.embed_documents([chunk.text for chunk in chunk_list])
        if len(vectors) != len(chunk_list):
            raise RuntimeError("Embedding provider returned unexpected number of vectors")
        return [EmbedVector(chunk_id=chunk.chunk_id, values=vector) for chunk, vector in zip(chunk_list, vectors)]

    def embed_query(self, query: str) -> List[float]:
        return self._provider.embed_query(query)


__all__ = ["EmbeddingService", "ProviderConfigurationError", "EmbeddingProvider"]
