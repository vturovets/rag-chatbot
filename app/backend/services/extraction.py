"""Content extraction services for PDF and audio."""
from __future__ import annotations

import asyncio
from uuid import UUID

import pdfplumber
from mutagen.mp3 import MP3

from app.backend import exceptions
from app.backend.config import get_settings
from app.backend.models.ingestion import ExtractionResult, TranscriptionResult
from app.backend.services.storage import FileStorage


class ExtractionService:
    """Aggregate PDF extraction and audio transcription logic."""

    def __init__(self, storage: FileStorage | None = None) -> None:
        self._settings = get_settings()
        self._storage = storage or FileStorage()

    async def extract_pdf(self, file_id: UUID) -> ExtractionResult:
        path = self._storage.get_file(file_id)

        def _extract() -> ExtractionResult:
            with pdfplumber.open(path) as pdf:
                page_count = len(pdf.pages)
                if page_count > self._settings.max_pdf_pages:
                    raise exceptions.timeout_stage("Processing timed out while extracting text.")
                texts = [page.extract_text() or "" for page in pdf.pages]
            combined = "\n".join(filter(None, texts))
            if not combined.strip():
                raise exceptions.invalid_pdf_structure()
            return ExtractionResult(text=combined, pages=page_count)

        try:
            return await asyncio.wait_for(asyncio.to_thread(_extract), timeout=self._settings.extraction_timeout_s)
        except asyncio.TimeoutError as exc:  # pragma: no cover - safety guard
            raise exceptions.timeout_stage("Processing timed out while extracting text.") from exc

    async def transcribe_audio(self, file_id: UUID) -> TranscriptionResult:
        path = self._storage.get_file(file_id)
        try:
            audio = await asyncio.wait_for(asyncio.to_thread(MP3, path), timeout=self._settings.transcription_timeout_s)
        except asyncio.TimeoutError as exc:
            raise exceptions.timeout_stage("Processing timed out while transcribing audio.") from exc
        except Exception as exc:  # pragma: no cover - library errors
            raise exceptions.invalid_file_type("Unsupported audio file.") from exc
        duration_minutes = audio.info.length / 60
        if duration_minutes > self._settings.max_audio_minutes:
            raise exceptions.audio_too_long()
        transcript = "Transcription is unavailable in the offline environment."
        return TranscriptionResult(transcript=transcript, duration_seconds=audio.info.length)


__all__ = ["ExtractionService"]
