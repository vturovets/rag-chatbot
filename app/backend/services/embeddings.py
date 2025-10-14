"""Embedding generation services supporting multiple providers."""
from __future__ import annotations

import time
from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, List, Protocol

try:  # pragma: no cover - optional dependency import guards
    from openai import (  # type: ignore import for optional dependency
        APIConnectionError,
        APIError,
        APIStatusError,
        APITimeoutError,
        OpenAI,
        OpenAIError,
        RateLimitError,
    )
except Exception:  # pragma: no cover - guard when openai not installed
    APIConnectionError = APIError = APIStatusError = APITimeoutError = OpenAIError = RateLimitError = None  # type: ignore
    OpenAI = None  # type: ignore

try:  # pragma: no cover - optional dependency import guards
    import google.generativeai as genai  # type: ignore
    from google.api_core import exceptions as google_exceptions  # type: ignore
except Exception:  # pragma: no cover - guard when google generative AI not installed
    genai = None  # type: ignore
    google_exceptions = None  # type: ignore

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


class EmbeddingOperationError(RuntimeError):
    """Raised when embeddings cannot be generated."""


class EmbeddingRateLimitError(EmbeddingOperationError):
    """Raised when the embedding provider signals rate limiting."""


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
        if OpenAI is None:
            raise ProviderConfigurationError("openai package is not available")
        if not api_key:
            raise ProviderConfigurationError("OpenAI API key is not configured")
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
        delay = 0.5
        attempts = 4
        last_exception: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
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
            except RateLimitError as exc:  # type: ignore[arg-type]
                last_exception = exc
                if attempt == attempts:
                    raise EmbeddingRateLimitError("OpenAI rate limit exceeded") from exc
                time.sleep(delay)
            except (APIStatusError, APIError, APIConnectionError, APITimeoutError, OpenAIError) as exc:  # type: ignore[arg-type]
                last_exception = exc
                if attempt == attempts:
                    raise EmbeddingOperationError("OpenAI embedding failed") from exc
                time.sleep(delay)
            delay *= 2
        raise EmbeddingOperationError("OpenAI embedding failed") from last_exception

    def embed_query(self, text: str) -> List[float]:
        vector = self.embed_documents([text])
        return vector[0] if vector else []


class GoogleAIVectorProvider:
    """Embedding provider backed by Google Generative AI embeddings."""

    def __init__(self, model_name: str, api_key: str | None) -> None:
        if not api_key:
            raise ProviderConfigurationError("Google API key is not configured")
        if genai is None:
            raise ProviderConfigurationError("google-generativeai package is required")
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

    def _embed_with_retry(self, text: str) -> List[float]:
        attempts = 4
        delay = 0.5
        last_exception: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = self._client.embed_content(model=self._model_name, content=text)
                vector = response.get("embedding", [])
                if not vector:
                    base_dimension = self._dimension or 768
                    vector = _hash_to_unit_vector(text, base_dimension)
                if vector and self._dimension is None:
                    self._dimension = len(vector)
                return vector
            except Exception as exc:  # pragma: no cover - provider specific exceptions
                last_exception = exc
                if google_exceptions and isinstance(exc, google_exceptions.ResourceExhausted):
                    if attempt == attempts:
                        raise EmbeddingRateLimitError("Google Generative AI rate limit exceeded") from exc
                    time.sleep(delay)
                    delay *= 2
                    continue
                if google_exceptions and isinstance(
                    exc,
                    (
                        google_exceptions.ServiceUnavailable,
                        google_exceptions.InternalServerError,
                        google_exceptions.DeadlineExceeded,
                    ),
                ):
                    if attempt == attempts:
                        raise EmbeddingOperationError("Google Generative AI embedding failed") from exc
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise EmbeddingOperationError("Google Generative AI embedding failed") from exc
        raise EmbeddingOperationError("Google Generative AI embedding failed") from last_exception

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            embeddings.append(self._embed_with_retry(text))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._embed_with_retry(text)


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
        try:
            vectors = self._provider.embed_documents([chunk.text for chunk in chunk_list])
        except EmbeddingRateLimitError:
            raise
        except Exception as exc:
            raise EmbeddingOperationError("Failed to embed chunks") from exc
        if len(vectors) != len(chunk_list):
            raise EmbeddingOperationError("Embedding provider returned unexpected number of vectors")
        return [EmbedVector(chunk_id=chunk.chunk_id, values=vector) for chunk, vector in zip(chunk_list, vectors)]

    def embed_query(self, query: str) -> List[float]:
        try:
            return self._provider.embed_query(query)
        except EmbeddingRateLimitError:
            raise
        except Exception as exc:
            raise EmbeddingOperationError("Failed to embed query") from exc


__all__ = [
    "EmbeddingService",
    "EmbeddingProvider",
    "EmbeddingOperationError",
    "EmbeddingRateLimitError",
    "ProviderConfigurationError",
]
