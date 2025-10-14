"""Application configuration settings."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field, validator

try:  # pragma: no cover - ConfigDict introduced in Pydantic v2
    from pydantic import ConfigDict
except Exception:  # pragma: no cover - fall back when unavailable
    ConfigDict = None  # type: ignore[misc]

try:  # pragma: no cover - import location varies across pydantic versions
    from pydantic import BaseSettings
except Exception:  # pragma: no cover - fallback for pydantic>=2.12 where BaseSettings moved
    from pydantic import BaseModel

    class BaseSettings(BaseModel):
        """Lightweight compatibility shim for projects without pydantic-settings."""

        if ConfigDict is not None:
            model_config = ConfigDict(extra="ignore")  # type: ignore[assignment]
        env_prefix: ClassVar[str] = ""
        env_file: ClassVar[str | None] = None
        env_file_encoding: ClassVar[str] = "utf-8"

        @classmethod
        def _env_key(cls, field_name: str) -> str:
            return f"{cls.env_prefix}{field_name}".upper()

        @classmethod
        def _load_environment(cls) -> dict[str, object]:
            values: dict[str, object] = {}
            field_definitions = getattr(cls, "model_fields", None)
            if field_definitions is None:  # pragma: no cover - compatibility with pydantic<2
                field_definitions = getattr(cls, "__fields__", {})
            for name in field_definitions:
                env_key = cls._env_key(name)
                if env_key in os.environ:
                    values[name] = os.environ[env_key]
            return values

        def __init__(self, **data: object) -> None:
            combined: dict[str, object] = {}
            combined.update(self._load_environment())
            combined.update(data)
            super().__init__(**combined)


if ConfigDict is not None:
    SETTINGS_MODEL_CONFIG = ConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
    )
else:  # pragma: no cover - compatibility with older pydantic versions
    SETTINGS_MODEL_CONFIG = None


class Settings(BaseSettings):
    """Central application configuration."""

    environment: Literal["dev", "prod"] = Field("dev", description="Runtime environment")
    debug_mode: bool = Field(True, description="Enable debug endpoints")
    storage_dir: Path = Field(default=Path("storage"), description="Directory for uploaded files")
    file_retention_hours: int = Field(24, ge=1, description="Hours to retain uploaded files")
    session_retention_hours: int = Field(24, ge=1, description="Hours to retain chat sessions")

    chunk_size: int = Field(500, ge=64, description="Token target per chunk")
    chunk_overlap: int = Field(60, ge=0, description="Token overlap between chunks")
    top_k: int = Field(5, ge=1, le=8, description="Default retrieval depth")

    max_upload_mb: int = Field(200, ge=1, description="Maximum upload size")
    max_pdf_pages: int = Field(200, ge=1, description="Maximum number of PDF pages")
    max_audio_minutes: int = Field(60, ge=1, description="Maximum audio duration in minutes")

    llm_provider: Literal["openai", "google"] = Field("openai", description="LLM provider selection")
    llm_model: str = Field("gpt-4o-mini", description="LLM model identifier")
    embedding_model: str = Field("text-embedding-3-large", description="Embedding model name")

    extraction_timeout_s: float = Field(5.0, gt=0, description="Timeout for extraction stage")
    transcription_timeout_s: float = Field(8.0, gt=0, description="Timeout for transcription stage")
    retrieval_timeout_s: float = Field(2.5, gt=0, description="Timeout for retrieval stage")
    prompt_timeout_s: float = Field(0.5, gt=0, description="Timeout for prompt assembly")
    generation_timeout_s: float = Field(3.0, gt=0, description="Timeout for generation stage")

    transcription_max_retries: int = Field(3, ge=1, description="Retry attempts for transcription API")
    transcription_retry_backoff_s: float = Field(1.0, gt=0, description="Initial backoff for retries")
    whisper_model: str = Field("whisper-large-v3", description="Whisper model identifier")

    openai_api_key: str | None = Field(default=None, description="API key for OpenAI services")
    openai_api_base: str | None = Field(default=None, description="Optional override for OpenAI API base URL")
    google_api_key: str | None = Field(default=None, description="API key for Google Generative AI")

    env_prefix: ClassVar[str] = "RAG_"
    env_file: ClassVar[str | None] = ".env"
    env_file_encoding: ClassVar[str] = "utf-8"

    if SETTINGS_MODEL_CONFIG is not None:
        model_config = SETTINGS_MODEL_CONFIG
    else:  # pragma: no cover - compatibility with pydantic<2
        class Config:
            env_prefix = "RAG_"
            env_file = ".env"
            env_file_encoding = "utf-8"

    @validator("storage_dir", pre=True)
    def _ensure_path(cls, value: Path | str) -> Path:
        return Path(value)


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""

    settings = Settings()
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    return settings


__all__ = ["Settings", "get_settings"]
