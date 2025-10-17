"""FastAPI route definitions for the RAG chatbot."""
from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, File, Query, UploadFile

from app.backend import exceptions
from app.backend.config import Settings, get_settings
from app.backend.models.chat import ChatRequest, ChatResponse, DebugPipelineRequest, DebugPipelineResponse
from app.backend.models.ingestion import FileKind, UploadResponse
from app.backend.models.status import HealthLimits, HealthResponse, PurgeResponse
from app.backend.services.lifecycle import BackendLifecycle
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
    return PipelineService(
        storage=get_storage(),
        session_store=get_session_store(),
        lifecycle=BackendLifecycle(),
    )


def get_app_settings() -> Settings:
    return get_settings()


def _format_provider_name(identifier: str) -> str:
    mapping = {"openai": "OpenAI", "google": "Google"}
    return mapping.get(identifier.lower(), identifier)


@router.get("/health", response_model=HealthResponse)
async def health(
    raw: bool = Query(False, description="Return extended diagnostics when debug mode is enabled."),
    settings: Settings = Depends(get_app_settings),
) -> HealthResponse:
    if raw and not settings.debug_mode:
        raise exceptions.unauthorized_debug()

    limits: HealthLimits | None = None
    if raw:
        limits = HealthLimits(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            top_k=settings.top_k,
            max_upload_mb=settings.max_upload_mb,
            max_pdf_pages=settings.max_pdf_pages,
            max_audio_minutes=settings.max_audio_minutes,
        )

    return HealthResponse(
        status="ok",
        provider=_format_provider_name(settings.llm_provider),
        vector_db="ChromaDB",
        environment=settings.environment,
        debug_mode=settings.debug_mode,
        limits=limits,
    )


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
        source=metadata.source or metadata.kind.value,
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
        source=metadata.source or metadata.kind.value,
        duration_seconds=transcription.duration_seconds,
        expires_at=metadata.expires_at,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, pipeline: PipelineService = Depends(get_pipeline)) -> ChatResponse:
    return await pipeline.chat(request)


@router.post("/debug/pipeline", response_model=DebugPipelineResponse)
async def debug_pipeline(
    request: DebugPipelineRequest,
    break_at: str = Query("generate", description="Stage at which to stop the pipeline."),
    raw: bool = Query(False, description="Return raw payloads for each stage when debug mode is enabled."),
    pipeline: PipelineService = Depends(get_pipeline),
    settings: Settings = Depends(get_app_settings),
) -> DebugPipelineResponse:
    if not settings.debug_mode:
        raise exceptions.unauthorized_debug()
    return await pipeline.debug_pipeline(request, break_at, raw)


@router.post("/admin/purge", response_model=PurgeResponse)
async def purge_ingested_content(
    pipeline: PipelineService = Depends(get_pipeline),
    settings: Settings = Depends(get_app_settings),
) -> PurgeResponse:
    if not settings.debug_mode:
        raise exceptions.unauthorized_debug()
    pipeline.purge_all()
    return PurgeResponse(status="purged")


__all__ = ["router"]
