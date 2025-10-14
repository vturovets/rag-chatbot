"""File storage and session persistence utilities."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict
from uuid import UUID

from fastapi import UploadFile

from app.backend import exceptions
from app.backend.config import get_settings
from app.backend.models.ingestion import FileKind, FileMetadata, FileRecord


class FileStorage:
    """Manage persisted uploaded files and their metadata lifecycle."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._registry: Dict[UUID, FileRecord] = {}
        self._retention = timedelta(hours=self._settings.file_retention_hours)

    @property
    def storage_dir(self) -> Path:
        return self._settings.storage_dir

    def _data_path(self, file_id: UUID) -> Path:
        return self.storage_dir / f"{file_id}"

    def _metadata_path(self, file_id: UUID) -> Path:
        return self.storage_dir / f"{file_id}.json"

    def _validate_size(self, size_bytes: int) -> None:
        max_bytes = self._settings.max_upload_mb * 1024 * 1024
        if size_bytes > max_bytes:
            raise exceptions.file_too_large()

    def _purge_expired(self) -> None:
        now = datetime.utcnow()
        for file_id, record in list(self._registry.items()):
            if record.has_expired(now):
                self.purge(file_id)

        for metadata_path in self.storage_dir.glob("*.json"):
            try:
                metadata = FileMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
            except Exception:  # pragma: no cover - corrupt metadata guard
                continue
            if metadata.has_expired(now):
                self.purge(metadata.file_id)

    def _write_metadata(self, record: FileRecord) -> None:
        self._metadata_path(record.metadata.file_id).write_text(
            record.metadata.model_dump_json(), encoding="utf-8"
        )

    def _load_record(self, file_id: UUID) -> FileRecord:
        metadata_path = self._metadata_path(file_id)
        if not metadata_path.exists():
            raise exceptions.file_not_found()
        metadata = FileMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
        record = FileRecord(metadata=metadata, path=self._data_path(file_id))
        self._registry[file_id] = record
        return record

    def _get_record(self, file_id: UUID) -> FileRecord:
        self._purge_expired()
        record = self._registry.get(file_id)
        if record is None:
            record = self._load_record(file_id)
        if record.has_expired():
            self.purge(file_id)
            raise exceptions.file_not_found()
        if not record.path.exists():
            self.purge(file_id)
            raise exceptions.file_not_found()
        return record

    async def save_upload(self, file: UploadFile, kind: FileKind) -> FileMetadata:
        """Persist an uploaded file to disk and return metadata."""

        contents = await file.read()
        self._validate_size(len(contents))
        self._purge_expired()
        uploaded_at = datetime.utcnow()
        metadata = FileMetadata(
            filename=file.filename or "upload",
            content_type=file.content_type or "application/octet-stream",
            size_bytes=len(contents),
            uploaded_at=uploaded_at,
            expires_at=uploaded_at + self._retention,
            kind=kind,
        )

        destination = self._data_path(metadata.file_id)
        destination.write_bytes(contents)
        record = FileRecord(metadata=metadata, path=destination)
        self._registry[metadata.file_id] = record
        self._write_metadata(record)
        return metadata

    def get_file(self, file_id: UUID) -> Path:
        """Return the path for a given file identifier."""

        record = self._get_record(file_id)
        return record.path

    def get_metadata(self, file_id: UUID) -> FileMetadata:
        """Retrieve metadata for the given file identifier."""

        record = self._get_record(file_id)
        return record.metadata

    def purge(self, file_id: UUID) -> None:
        """Remove file contents and metadata."""

        data_path = self._data_path(file_id)
        metadata_path = self._metadata_path(file_id)
        try:
            data_path.unlink()
        except FileNotFoundError:
            pass
        try:
            metadata_path.unlink()
        except FileNotFoundError:
            pass
        self._registry.pop(file_id, None)


__all__ = ["FileStorage"]
