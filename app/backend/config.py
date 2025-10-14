"""Application configuration settings."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseSettings, Field, validator


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
    embedding_model: str = Field("text-embedding-3-large", description="Embedding model name")

    extraction_timeout_s: float = Field(5.0, gt=0, description="Timeout for extraction stage")
    transcription_timeout_s: float = Field(8.0, gt=0, description="Timeout for transcription stage")
    generation_timeout_s: float = Field(6.0, gt=0, description="Timeout for generation stage")

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
