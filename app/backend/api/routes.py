"""FastAPI route definitions for the RAG chatbot."""
from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, File, UploadFile

from app.backend import exceptions
from app.backend.config import Settings, get_settings
from app.backend.models.chat import ChatRequest, ChatResponse, DebugPipelineRequest, DebugPipelineResponse
from app.backend.models.ingestion import FileKind, UploadResponse
from app.backend.services.pipeline import PipelineService
from app.backend.services.session_store import SessionStore
from app.backend.services.storage import FileStorage

PDF_MIME_TYPES = {"application/pdf", "application/x-pdf"}
AUDIO_MIME_TYPES = {"audio/mpeg", "audio/mp3", "audio/mpeg3"}


router = APIRouter()


@lru_cache()
def get_storage() -> FileStorage:
    return FileStorage()


@lru_cache()
def get_session_store() -> SessionStore:
    return SessionStore()


@lru_cache()
def get_pipeline() -> PipelineService:
    return PipelineService(storage=get_storage(), session_store=get_session_store())


def get_app_settings() -> Settings:
    return get_settings()


@router.get("/health", response_model=dict)
async def health(settings: Settings = Depends(get_app_settings)) -> dict:
    return {"status": "ok", "provider": settings.llm_provider.title(), "vector_db": "ChromaDB"}


def _is_pdf_upload(file: UploadFile) -> bool:
    content_type = (file.content_type or "").lower()
    if content_type in PDF_MIME_TYPES:
        return True
    filename = (file.filename or "").lower()
    return filename.endswith(".pdf")


def _is_mp3_upload(file: UploadFile) -> bool:
    content_type = (file.content_type or "").lower()
    if content_type in AUDIO_MIME_TYPES:
        return True
    filename = (file.filename or "").lower()
    return filename.endswith(".mp3")


@router.post("/upload/pdf", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    pipeline: PipelineService = Depends(get_pipeline),
    storage: FileStorage = Depends(get_storage),
) -> UploadResponse:
    if not _is_pdf_upload(file):
        raise exceptions.invalid_file_type()
    metadata = await storage.save_upload(file, FileKind.PDF)
    try:
        extraction = await pipeline.handle_pdf_upload(metadata.file_id)
    except Exception:
        storage.purge(metadata.file_id)
        raise
    return UploadResponse(
        file_id=metadata.file_id,
        filename=metadata.filename,
        kind=metadata.kind,
        page_count=extraction.pages,
        expires_at=metadata.expires_at,
    )


@router.post("/upload/audio", response_model=UploadResponse)
async def upload_audio(
    file: UploadFile = File(...),
    pipeline: PipelineService = Depends(get_pipeline),
    storage: FileStorage = Depends(get_storage),
) -> UploadResponse:
    if not _is_mp3_upload(file):
        raise exceptions.invalid_file_type()
    metadata = await storage.save_upload(file, FileKind.AUDIO)
    try:
        transcription = await pipeline.handle_audio_upload(metadata.file_id)
    except Exception:
        storage.purge(metadata.file_id)
        raise
    return UploadResponse(
        file_id=metadata.file_id,
        filename=metadata.filename,
        kind=metadata.kind,
        duration_seconds=transcription.duration_seconds,
        expires_at=metadata.expires_at,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, pipeline: PipelineService = Depends(get_pipeline)) -> ChatResponse:
    return await pipeline.chat(request)


@router.post("/debug/pipeline", response_model=DebugPipelineResponse)
async def debug_pipeline(
    request: DebugPipelineRequest,
    pipeline: PipelineService = Depends(get_pipeline),
    settings: Settings = Depends(get_app_settings),
) -> DebugPipelineResponse:
    if not settings.debug_mode:
        raise exceptions.unauthorized_debug()
    return await pipeline.debug_pipeline(request.file_id, request.break_at, request.raw)


__all__ = ["router"]
