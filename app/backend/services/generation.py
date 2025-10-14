"""LLM response generation services."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

try:  # pragma: no cover - optional dependency import guards
    from openai import (  # type: ignore import for optional dependency
        APIConnectionError,
        APIError,
        APIStatusError,
        APITimeoutError,
        AsyncOpenAI,
        OpenAIError,
        RateLimitError,
    )
except Exception:  # pragma: no cover - guard when openai not installed
    APIConnectionError = APIError = APIStatusError = APITimeoutError = OpenAIError = RateLimitError = None  # type: ignore
    AsyncOpenAI = None  # type: ignore

try:  # pragma: no cover - optional dependency import guards
    import google.generativeai as genai  # type: ignore
    from google.api_core import exceptions as google_exceptions  # type: ignore
except Exception:  # pragma: no cover - guard when google generative AI not installed
    genai = None  # type: ignore
    google_exceptions = None  # type: ignore

from app.backend.config import get_settings


class GenerationError(RuntimeError):
    """Base error for generation failures."""


class GenerationTimeoutError(GenerationError):
    """Raised when the provider exceeds allotted time."""


class ProviderUnavailableError(GenerationError):
    """Raised when the provider cannot be reached."""


class RateLimitedError(GenerationError):
    """Raised when the provider signals rate limiting."""


class ProviderConfigurationError(RuntimeError):
    """Raised when a provider cannot be initialised due to configuration issues."""


class LLMProvider(Protocol):
    """Protocol implemented by provider-specific adapters."""

    @property
    def name(self) -> str:
        ...

    async def generate(
        self,
        *,
        prompt: str,
        query: str,
        context: Sequence[str],
        timeout: float | None,
    ) -> str:
        ...


class GenerationService:
    """Generate answers using the configured LLM provider."""

    def __init__(self, provider: LLMProvider | None = None) -> None:
        self._settings = get_settings()
        self._provider = provider or self._resolve_provider()

    def _resolve_provider(self) -> LLMProvider:
        provider_name = self._settings.llm_provider.lower()
        model_name = self._settings.llm_model

        try:
            if provider_name == "openai":
                return OpenAIChatProvider(model_name=model_name)
            if provider_name == "google":
                return GoogleChatProvider(model_name=model_name)
        except ProviderConfigurationError:
            if self._settings.environment == "prod":
                raise
        return LocalFallbackProvider()

    async def generate(
        self,
        *,
        prompt: str,
        query: str,
        context: Sequence[str],
        timeout: float | None,
    ) -> str:
        return await self._provider.generate(prompt=prompt, query=query, context=context, timeout=timeout)


class OpenAIChatProvider:
    """OpenAI-backed chat generation provider."""

    def __init__(self, *, model_name: str) -> None:
        if AsyncOpenAI is None:
            raise ProviderConfigurationError("openai package is not available")
        api_key = get_settings().openai_api_key
        api_base = get_settings().openai_api_base
        if not api_key:
            raise ProviderConfigurationError("OpenAI API key is not configured")
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base
        self._client = AsyncOpenAI(**client_kwargs)
        self._model_name = model_name

    @property
    def name(self) -> str:
        return "openai"

    async def generate(
        self,
        *,
        prompt: str,
        query: str,
        context: Sequence[str],
        timeout: float | None,
    ) -> str:
        assert AsyncOpenAI is not None  # for type checkers
        attempts = 4
        delay = 0.5
        deadline = time.perf_counter() + timeout if timeout is not None else None
        last_exception: Exception | None = None
        for attempt in range(1, attempts + 1):
            per_attempt_timeout: float | None = timeout
            if deadline is not None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise GenerationTimeoutError("OpenAI response timed out")
                per_attempt_timeout = remaining
            try:
                response = await asyncio.wait_for(
                    self._client.chat.completions.create(  # type: ignore[call-arg]
                        model=self._model_name,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a retrieval-augmented assistant. Provide concise, citation-free "
                                    "answers grounded in the supplied context."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.1,
                        max_tokens=256,
                    ),
                    timeout=per_attempt_timeout,
                )
            except asyncio.TimeoutError as exc:
                raise GenerationTimeoutError("OpenAI response timed out") from exc
            except RateLimitError as exc:  # type: ignore[arg-type]
                last_exception = exc
                if attempt == attempts:
                    raise RateLimitedError("OpenAI rate limit exceeded") from exc
                await asyncio.sleep(delay)
                delay *= 2
                continue
            except (APITimeoutError, APIConnectionError) as exc:  # type: ignore[arg-type]
                last_exception = exc
                if attempt == attempts:
                    raise ProviderUnavailableError("Failed to reach OpenAI service") from exc
                await asyncio.sleep(delay)
                delay *= 2
                continue
            except (APIStatusError, APIError, OpenAIError) as exc:  # type: ignore[arg-type]
                raise GenerationError("OpenAI generation failed") from exc
            if not response or not getattr(response, "choices", None):
                raise GenerationError("OpenAI returned an empty response")
            choice = response.choices[0]
            content = getattr(getattr(choice, "message", None), "content", None)
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                combined = "".join(
                    segment.get("text", "") if isinstance(segment, dict) else str(segment) for segment in content
                ).strip()
                if combined:
                    return combined
            if hasattr(choice, "text") and isinstance(choice.text, str) and choice.text.strip():
                return choice.text.strip()
            raise GenerationError("OpenAI response did not contain text content")
        raise GenerationError("OpenAI response did not contain text content") from last_exception


class GoogleChatProvider:
    """Google Generative AI backed provider."""

    def __init__(self, *, model_name: str) -> None:
        if genai is None:
            raise ProviderConfigurationError("google-generativeai package is not available")
        api_key = get_settings().google_api_key
        if not api_key:
            raise ProviderConfigurationError("Google API key is not configured")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)

    @property
    def name(self) -> str:
        return "google"

    async def generate(
        self,
        *,
        prompt: str,
        query: str,
        context: Sequence[str],
        timeout: float | None,
    ) -> str:
        def _call_model() -> Any:
            return self._model.generate_content(
                prompt,
                generation_config={"temperature": 0.1, "max_output_tokens": 256},
            )

        attempts = 4
        delay = 0.5
        deadline = time.perf_counter() + timeout if timeout is not None else None
        last_exception: Exception | None = None
        for attempt in range(1, attempts + 1):
            per_attempt_timeout: float | None = timeout
            if deadline is not None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise GenerationTimeoutError("Google Generative AI response timed out")
                per_attempt_timeout = remaining
            try:
                response = await asyncio.wait_for(asyncio.to_thread(_call_model), timeout=per_attempt_timeout)
            except asyncio.TimeoutError as exc:
                raise GenerationTimeoutError("Google Generative AI response timed out") from exc
            except Exception as exc:  # pragma: no cover - provider specific exceptions
                last_exception = exc
                if google_exceptions and isinstance(
                    exc,
                    (
                        google_exceptions.ServiceUnavailable,
                        google_exceptions.InternalServerError,
                        google_exceptions.DeadlineExceeded,
                    ),
                ):
                    if attempt == attempts:
                        raise ProviderUnavailableError("Google Generative AI service unavailable") from exc
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                if google_exceptions and isinstance(exc, google_exceptions.ResourceExhausted):
                    if attempt == attempts:
                        raise RateLimitedError("Google Generative AI rate limit exceeded") from exc
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                raise GenerationError("Google Generative AI generation failed") from exc
            if not response:
                raise GenerationError("Google Generative AI returned an empty response")
            text = getattr(response, "text", None)
            if isinstance(text, str) and text.strip():
                return text.strip()

            parts = getattr(response, "candidates", None) or []
            for candidate in parts:
                content = getattr(candidate, "content", None)
                if content and getattr(content, "parts", None):
                    fragments = [
                        getattr(part, "text", "")
                        for part in getattr(content, "parts", [])
                        if getattr(part, "text", "")
                    ]
                    combined = "".join(fragments).strip()
                    if combined:
                        return combined

            raise GenerationError("Google Generative AI response did not contain text content")
        raise GenerationError("Google Generative AI response did not contain text content") from last_exception


@dataclass
class LocalFallbackProvider:
    """Deterministic fallback when remote providers are unavailable."""

    name: str = "local"

    async def generate(
        self,
        *,
        prompt: str,
        query: str,
        context: Sequence[str],
        timeout: float | None,
    ) -> str:
        del prompt  # unused, but maintained for signature compatibility
        snippets = [" ".join(fragment.strip().split()) for fragment in context if fragment.strip()]
        if not snippets:
            return "I do not yet have enough information from the uploaded materials to answer that."
        summary = " ".join(snippets)[:480]
        return f"Based on the uploaded materials, here is what I can share: {summary}"


__all__ = [
    "GenerationService",
    "GenerationError",
    "GenerationTimeoutError",
    "ProviderUnavailableError",
    "RateLimitedError",
    "ProviderConfigurationError",
]

