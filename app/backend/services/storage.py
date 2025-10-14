"""File storage utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict
from uuid import UUID

from fastapi import UploadFile

from app.backend.config import get_settings
from app.backend.models.ingestion import FileMetadata


class FileStorage:
    """Manage persisted uploaded files."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._registry: Dict[UUID, FileMetadata] = {}

    @property
    def storage_dir(self) -> Path:
        return self._settings.storage_dir

    def _validate_size(self, size_bytes: int) -> None:
        max_bytes = self._settings.max_upload_mb * 1024 * 1024
        if size_bytes > max_bytes:
            from app.backend import exceptions

            raise exceptions.file_too_large()

    async def save_upload(self, file: UploadFile, kind: str) -> FileMetadata:
        """Persist an uploaded file to disk and return metadata."""

        contents = await file.read()
        self._validate_size(len(contents))

        metadata = FileMetadata(
            filename=file.filename or "upload",
            content_type=file.content_type or "application/octet-stream",
            size_bytes=len(contents),
            kind=kind,
        )

        destination = self.storage_dir / f"{metadata.file_id}"
        destination.write_bytes(contents)
        self._registry[metadata.file_id] = metadata
        return metadata

    def get_file(self, file_id: UUID) -> Path:
        """Return the path for a given file identifier."""

        path = self.storage_dir / f"{file_id}"
        if not path.exists():
            from app.backend import exceptions

            raise exceptions.file_not_found()
        return path

    def get_metadata(self, file_id: UUID) -> FileMetadata:
        if file_id not in self._registry:
            from app.backend import exceptions

            raise exceptions.file_not_found()
        return self._registry[file_id]

    def purge(self, file_id: UUID) -> None:
        path = self.storage_dir / f"{file_id}"
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        self._registry.pop(file_id, None)


__all__ = ["FileStorage"]
