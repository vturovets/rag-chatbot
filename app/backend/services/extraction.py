"""Content extraction services for PDF and audio."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any
from uuid import UUID

import fitz
import pdfplumber
from mutagen import MutagenError
from mutagen.mp3 import MP3
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAIError,
    RateLimitError,
)

try:  # pragma: no cover - optional dependency in some environments
    from openai import APIStatusError
except ImportError:  # pragma: no cover - compatibility fallback
    APIStatusError = APIError

from app.backend import exceptions
from app.backend.config import get_settings
from app.backend.models.ingestion import ExtractionResult, TranscriptionResult
from app.backend.services.storage import FileStorage


class ExtractionService:
    """Aggregate PDF extraction and audio transcription logic."""

    def __init__(self, storage: FileStorage | None = None, openai_client: AsyncOpenAI | None = None) -> None:
        self._settings = get_settings()
        self._storage = storage or FileStorage()
        self._openai_client = openai_client or self._build_openai_client()

    def _build_openai_client(self) -> AsyncOpenAI | None:
        api_key = self._settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if self._settings.openai_api_base:
            client_kwargs["base_url"] = self._settings.openai_api_base
        try:
            return AsyncOpenAI(**client_kwargs)
        except Exception:  # pragma: no cover - client init errors surfaced on call
            return None

    async def extract_pdf(self, file_id: UUID) -> ExtractionResult:
        path = self._storage.get_file(file_id)

        def _extract() -> ExtractionResult:
            try:
                with fitz.open(path) as document:
                    page_count = document.page_count
                    if page_count > self._settings.max_pdf_pages:
                        raise exceptions.timeout_stage("Processing timed out while extracting text.")
                    preview_text = [document.load_page(i).get_text("text").strip() for i in range(page_count)]
            except Exception as exc:  # pragma: no cover - PyMuPDF specific errors
                raise exceptions.invalid_file_type("Unsupported file format. Please upload a PDF or MP3 file.") from exc

            if not any(preview_text):
                raise exceptions.invalid_pdf_structure()

            try:
                with pdfplumber.open(path) as pdf:
                    extracted_pages = [page.extract_text() or "" for page in pdf.pages[:page_count]]
            except Exception:  # pragma: no cover - pdfplumber fallback
                extracted_pages = preview_text

            combined = "\n".join(segment.strip() for segment in extracted_pages if segment.strip())
            if not combined:
                combined = "\n".join(segment for segment in preview_text if segment)

            if not combined.strip():
                raise exceptions.invalid_pdf_structure()

            return ExtractionResult(text=combined, pages=page_count)

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_extract), timeout=self._settings.extraction_timeout_s
            )
        except asyncio.TimeoutError as exc:  # pragma: no cover - safety guard
            raise exceptions.timeout_stage("Processing timed out while extracting text.") from exc

    async def transcribe_audio(self, file_id: UUID) -> TranscriptionResult:
        path = self._storage.get_file(file_id)

        def _load_mp3(metadata_path: Path) -> MP3:
            return MP3(metadata_path)

        try:
            audio = await asyncio.wait_for(
                asyncio.to_thread(_load_mp3, path), timeout=self._settings.transcription_timeout_s
            )
        except asyncio.TimeoutError as exc:
            raise exceptions.timeout_stage("Processing timed out while transcribing audio.") from exc
        except MutagenError as exc:
            raise exceptions.invalid_file_type("Unsupported file format. Please upload a PDF or MP3 file.") from exc
        except Exception as exc:  # pragma: no cover - library errors
            raise exceptions.invalid_file_type("Unsupported file format. Please upload a PDF or MP3 file.") from exc

        duration_seconds = getattr(getattr(audio, "info", None), "length", 0.0) or 0.0
        if duration_seconds <= 0:
            raise exceptions.transcription_error()
        if (duration_seconds / 60) > self._settings.max_audio_minutes:
            raise exceptions.audio_too_long()

        transcript = await self._transcribe_with_whisper(path)
        return TranscriptionResult(transcript=transcript, duration_seconds=duration_seconds)

    async def _transcribe_with_whisper(self, path: Path) -> str:
        if self._openai_client is None:
            raise exceptions.transcription_error(
                hint="Set RAG_OPENAI_API_KEY or OPENAI_API_KEY to enable transcription."
            )

        delay = self._settings.transcription_retry_backoff_s
        attempts = self._settings.transcription_max_retries
        last_exception: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                with path.open("rb") as stream:
                    response = await asyncio.wait_for(
                        self._openai_client.audio.transcriptions.create(
                            model=self._settings.whisper_model,
                            file=stream,
                            response_format="text",
                        ),
                        timeout=self._settings.transcription_timeout_s,
                    )
                text = self._extract_transcript_text(response)
                if not text:
                    raise exceptions.transcription_error()
                return text
            except RateLimitError as exc:
                last_exception = exc
                if attempt == attempts:
                    raise exceptions.rate_limit_exceeded() from exc
                await asyncio.sleep(delay)
                delay *= 2
            except asyncio.TimeoutError as exc:
                raise exceptions.timeout_stage("Processing timed out while transcribing audio.") from exc
            except (APITimeoutError, APIConnectionError, APIStatusError, APIError, OpenAIError) as exc:
                raise exceptions.transcription_error() from exc

        raise exceptions.transcription_error() from last_exception

    @staticmethod
    def _extract_transcript_text(response: Any) -> str:
        if isinstance(response, str):
            return response.strip()
        text = getattr(response, "text", None)
        if text:
            return str(text).strip()
        if hasattr(response, "model_dump"):
            payload = response.model_dump()
            for key in ("text", "transcript"):
                if payload.get(key):
                    return str(payload[key]).strip()
        if isinstance(response, dict):
            for key in ("text", "transcript"):
                if response.get(key):
                    return str(response[key]).strip()
        return ""


__all__ = ["ExtractionService"]
