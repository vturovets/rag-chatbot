"""Content extraction services for PDF and audio."""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
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

try:  # pragma: no cover - optional local transcription dependency
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover - make dependency optional for environments without it
    WhisperModel = None  # type: ignore[misc]

from app.backend import exceptions
from app.backend.config import get_settings
from app.backend.models.ingestion import ExtractionResult, TranscriptionResult
from app.backend.services.storage import FileStorage


class LocalTranscriber:
    """Lightweight wrapper around faster-whisper for local inference."""

    def __init__(self, model_size: str, device: str, compute_type: str) -> None:
        if WhisperModel is None:  # pragma: no cover - enforced via dependency
            raise ImportError("faster-whisper is not installed")
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model: WhisperModel | None = None

    def _ensure_model(self) -> WhisperModel:
        if self._model is not None:
            return self._model

        try:
            self._model = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
            return self._model
        except Exception as primary_exc:
            fallback_device = "cpu"
            if self._device == fallback_device:
                raise

            fallback_compute_type = self._compute_type
            if fallback_compute_type not in {"auto", "int8", "int8_float16", "int8_float32"}:
                fallback_compute_type = "int8"

            try:
                self._model = WhisperModel(
                    self._model_size,
                    device=fallback_device,
                    compute_type=fallback_compute_type,
                )
                self._device = fallback_device
                self._compute_type = fallback_compute_type
                return self._model
            except Exception as fallback_exc:
                raise RuntimeError(
                    "Failed to initialize faster-whisper on any device. "
                    f"Primary error: {primary_exc}. Fallback error: {fallback_exc}"
                ) from primary_exc

    def transcribe(self, path: Path) -> str:
        model = self._ensure_model()

        segments, _ = model.transcribe(str(path), beam_size=5)
        transcript = " ".join(segment.text.strip() for segment in segments if segment.text).strip()
        if not transcript:
            raise ValueError("Local transcription returned no text")
        return transcript


class ExtractionService:
    """Aggregate PDF extraction and audio transcription logic."""

    def __init__(
        self,
        storage: FileStorage | None = None,
        openai_client: AsyncOpenAI | None = None,
        local_transcriber: LocalTranscriber | None = None,
    ) -> None:
        self._settings = get_settings()
        self._storage = storage or FileStorage()
        self._openai_client = openai_client or self._build_openai_client()
        self._local_transcriber = local_transcriber

    def _get_local_transcriber(self) -> LocalTranscriber:
        if self._local_transcriber is not None:
            return self._local_transcriber

        try:
            model_size = getattr(self._settings, "local_transcription_model", "base")
            device = getattr(self._settings, "local_transcription_device", "auto")
            compute_type = getattr(self._settings, "local_transcription_compute_type", "auto")
            self._local_transcriber = LocalTranscriber(model_size, device, compute_type)
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise exceptions.transcription_error(
                hint="Install the 'faster-whisper' package to enable local transcription."
            ) from exc
        except Exception as exc:  # pragma: no cover - model initialization failure
            raise exceptions.transcription_error(
                hint=f"Failed to initialize local Whisper model: {exc}"
            ) from exc

        return self._local_transcriber

    def _convert_to_wav(self, source: Path) -> Path:
        temp_dir = Path(tempfile.mkdtemp(prefix="rag-transcribe-"))
        target = temp_dir / "audio.wav"
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(target),
        ]

        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as exc:
            raise exceptions.transcription_error(
                hint="ffmpeg is required for local transcription. Install ffmpeg and try again."
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if isinstance(exc.stderr, str) else None
            raise exceptions.transcription_error(
                hint=stderr or "ffmpeg failed while converting the audio file."
            ) from exc

        if not target.exists():
            raise exceptions.transcription_error(
                hint="ffmpeg did not produce the expected WAV output."
            )

        return target

    def _cleanup_temp_audio(self, path: Path) -> None:
        try:
            if path.exists():
                path.unlink()
        except OSError:  # pragma: no cover - best effort cleanup
            pass
        temp_dir = path.parent
        if temp_dir.name.startswith("rag-transcribe-"):
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _transcribe_locally_sync(self, path: Path) -> str:
        converted = self._convert_to_wav(path)
        try:
            transcriber = self._get_local_transcriber()
            try:
                transcript = transcriber.transcribe(converted)
            except Exception as exc:
                hint = str(exc).strip() or "Local transcription failed."
                raise exceptions.transcription_error(hint=hint) from exc
        finally:
            self._cleanup_temp_audio(converted)

        return transcript

    async def _transcribe_locally(self, path: Path) -> str:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._transcribe_locally_sync, path),
                timeout=self._settings.transcription_timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise exceptions.timeout_stage("Processing timed out while transcribing audio.") from exc

    def _build_openai_client(self) -> AsyncOpenAI | None:
        api_key = self._resolve_openai_api_key()
        if not api_key:
            return None

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        base_url = self._resolve_openai_base()
        if base_url:
            client_kwargs["base_url"] = base_url

        try:
            return AsyncOpenAI(**client_kwargs)
        except Exception:  # pragma: no cover - client init errors surfaced on call
            return None

    def _resolve_openai_api_key(self) -> str | None:
        for candidate in (
            getattr(self._settings, "openai_api_key", None),
            os.getenv("RAG_OPENAI_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
            self._load_env_file_api_key(),
        ):
            normalized = self._normalize_api_key(candidate)
            if normalized:
                os.environ.setdefault("OPENAI_API_KEY", normalized)
                return normalized
        return None

    def _resolve_openai_base(self) -> str | None:
        for candidate in (
            getattr(self._settings, "openai_api_base", None),
            os.getenv("RAG_OPENAI_API_BASE"),
            os.getenv("OPENAI_API_BASE"),
        ):
            normalized = self._normalize_api_key(candidate)
            if normalized:
                return normalized
        return None

    def _load_env_file_api_key(self) -> str | None:
        env_file = getattr(self._settings, "env_file", None)
        if not env_file:
            return None
        path = Path(env_file)
        if not path.exists():
            return None
        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                key, _, value = line.partition("=")
                if key.strip() == "RAG_OPENAI_API_KEY":
                    return value.split("#", 1)[0].strip()
        except OSError:  # pragma: no cover - filesystem errors
            return None
        return None

    @staticmethod
    def _normalize_api_key(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        if cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"} and len(cleaned) >= 2:
            cleaned = cleaned[1:-1].strip()
        return cleaned or None

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
        if getattr(self._settings, "local_transcription_only", False):
            return await self._transcribe_locally(path)

        if self._openai_client is None:
            return await self._transcribe_locally(path)

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
            except APIConnectionError as exc:
                raise exceptions.transcription_error(
                    hint="Unable to reach the OpenAI Whisper API. Verify network connectivity or OPENAI_API_BASE."
                ) from exc
            except asyncio.TimeoutError as exc:
                raise exceptions.timeout_stage("Processing timed out while transcribing audio.") from exc
            except APITimeoutError as exc:
                raise exceptions.timeout_stage("Processing timed out while transcribing audio.") from exc
            except APIStatusError as exc:
                status_code = getattr(exc, "status_code", None)
                if status_code == 429:
                    raise exceptions.rate_limit_exceeded() from exc
                if status_code == 401:
                    raise exceptions.transcription_error(
                        hint="OpenAI authentication failed. Set RAG_OPENAI_API_KEY or OPENAI_API_KEY with a valid key."
                    ) from exc
                if status_code == 404:
                    raise exceptions.transcription_error(
                        hint=(
                            "The configured transcription model "
                            f"'{self._settings.whisper_model}' is unavailable. "
                            "Set RAG_WHISPER_MODEL to a supported option such as "
                            "'gpt-4o-mini-transcribe'."
                        )
                    ) from exc
                hint = self._openai_error_hint(exc)
                if self._should_use_local_fallback(hint):
                    return await self._transcribe_locally(path)
                raise exceptions.transcription_error(hint=hint) from exc
            except (APIError, OpenAIError) as exc:
                hint = self._openai_error_hint(exc)
                if self._should_use_local_fallback(hint):
                    return await self._transcribe_locally(path)
                raise exceptions.transcription_error(hint=hint) from exc

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

    @staticmethod
    def _openai_error_hint(exc: Exception) -> str | None:
        """Extract a human-friendly hint from an OpenAI exception."""

        message = getattr(exc, "message", None)
        if isinstance(message, str) and message.strip():
            return message.strip()
        text = str(exc).strip()
        return text or None

    def _should_use_local_fallback(self, hint: str | None) -> bool:
        if not hint:
            return False
        normalized = hint.lower()
        return "unsupported file format" in normalized or "unsupported_value" in normalized


__all__ = ["ExtractionService", "LocalTranscriber"]
