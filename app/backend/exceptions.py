"""Domain-specific exceptions and error payload helpers."""
from __future__ import annotations

from typing import Dict

from fastapi import HTTPException, status


def build_error_payload(error_code: str, message: str, hint: str | None = None) -> Dict[str, str]:
    """Return a standardized error payload."""

    payload: Dict[str, str] = {"error_code": error_code, "message": message}
    if hint:
        payload["hint"] = hint
    return payload


class RagError(HTTPException):
    """Base error with standardized payload."""

    def __init__(self, *, status_code: int, error_code: str, message: str, hint: str | None = None) -> None:
        self.error_code = error_code
        self.message = message
        self.hint = hint
        super().__init__(
            status_code=status_code,
            detail=build_error_payload(error_code=error_code, message=message, hint=hint),
        )

    def to_payload(self) -> Dict[str, str]:
        """Return the serialized payload for the error."""

        return build_error_payload(self.error_code, self.message, self.hint)


def invalid_file_type(message: str = "Unsupported file format. Please upload a PDF or MP3 file.") -> RagError:
    return RagError(status_code=status.HTTP_400_BAD_REQUEST, error_code="INVALID_FILE_TYPE", message=message)


def invalid_pdf_structure(message: str = "This PDF contains no text layer and cannot be processed.") -> RagError:
    return RagError(status_code=status.HTTP_400_BAD_REQUEST, error_code="INVALID_PDF_STRUCTURE", message=message)


def file_too_large(message: str = "File exceeds size limit (200 MB).") -> RagError:
    return RagError(status_code=status.HTTP_400_BAD_REQUEST, error_code="FILE_TOO_LARGE", message=message)


def audio_too_long(message: str = "Audio exceeds 60-minute limit.") -> RagError:
    return RagError(status_code=status.HTTP_400_BAD_REQUEST, error_code="AUDIO_TOO_LONG", message=message)


def missing_query(message: str = "Query text is required.") -> RagError:
    return RagError(status_code=status.HTTP_400_BAD_REQUEST, error_code="MISSING_QUERY", message=message)


def unauthorized_debug(message: str = "Debug endpoints are only available in development mode.") -> RagError:
    return RagError(status_code=status.HTTP_401_UNAUTHORIZED, error_code="UNAUTHORIZED_DEBUG", message=message)


def invalid_debug_stage(message: str = "Unsupported debug stage requested.") -> RagError:
    return RagError(status_code=status.HTTP_400_BAD_REQUEST, error_code="INVALID_DEBUG_STAGE", message=message)


def file_not_found(message: str = "Referenced file not found or expired.") -> RagError:
    return RagError(status_code=status.HTTP_404_NOT_FOUND, error_code="FILE_NOT_FOUND", message=message)


def timeout_stage(message: str) -> RagError:
    return RagError(status_code=status.HTTP_408_REQUEST_TIMEOUT, error_code="TIMEOUT_STAGE", message=message)


def embedding_error(message: str = "Embedding generation failed.") -> RagError:
    return RagError(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, error_code="EMBEDDING_ERROR", message=message)


def transcription_error(
    message: str = "Audio transcription failed.",
    *,
    hint: str | None = None,
) -> RagError:
    return RagError(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        error_code="TRANSCRIPTION_ERROR",
        message=message,
        hint=hint,
    )


def rate_limit_exceeded(message: str = "Service is temporarily busy. Please try again later.") -> RagError:
    return RagError(status_code=status.HTTP_429_TOO_MANY_REQUESTS, error_code="RATE_LIMIT_EXCEEDED", message=message)


def internal_error(message: str = "Unexpected server error. Please retry or contact support.") -> RagError:
    return RagError(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, error_code="INTERNAL_ERROR", message=message)


def invalid_request(
    message: str = "The request payload is invalid.",
    *,
    hint: str | None = None,
) -> RagError:
    return RagError(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        error_code="INVALID_REQUEST",
        message=message,
        hint=hint,
    )


def http_error(
    message: str = "An HTTP error occurred while processing the request.",
    *,
    status_code: int = status.HTTP_400_BAD_REQUEST,
    hint: str | None = None,
) -> RagError:
    return RagError(status_code=status_code, error_code="HTTP_ERROR", message=message, hint=hint)


def resource_not_found(message: str = "The requested resource was not found.") -> RagError:
    return RagError(status_code=status.HTTP_404_NOT_FOUND, error_code="RESOURCE_NOT_FOUND", message=message)


def llm_provider_down(message: str = "Model provider unavailable. Try again shortly.") -> RagError:
    return RagError(status_code=status.HTTP_502_BAD_GATEWAY, error_code="LLM_PROVIDER_DOWN", message=message)


def vector_db_unavailable(message: str = "Vector database temporarily unreachable.") -> RagError:
    return RagError(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, error_code="VECTOR_DB_UNAVAILABLE", message=message)


def generation_timeout(message: str = "Response generation took too long.") -> RagError:
    return RagError(status_code=status.HTTP_504_GATEWAY_TIMEOUT, error_code="GENERATION_TIMEOUT", message=message)


__all__ = [
    "RagError",
    "invalid_file_type",
    "invalid_pdf_structure",
    "file_too_large",
    "audio_too_long",
    "missing_query",
    "unauthorized_debug",
    "invalid_debug_stage",
    "file_not_found",
    "timeout_stage",
    "embedding_error",
    "transcription_error",
    "rate_limit_exceeded",
    "internal_error",
    "invalid_request",
    "http_error",
    "resource_not_found",
    "llm_provider_down",
    "vector_db_unavailable",
    "generation_timeout",
    "build_error_payload",
]
