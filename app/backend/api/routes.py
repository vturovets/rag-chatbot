"""FastAPI route definitions for the RAG chatbot."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, UploadFile

from app.backend import exceptions
from app.backend.config import Settings, get_settings
from app.backend.models.chat import ChatRequest, ChatResponse, DebugPipelineRequest, DebugPipelineResponse
from app.backend.models.ingestion import UploadResponse
from app.backend.services.pipeline import PipelineService
from app.backend.services.storage import FileStorage

router = APIRouter()


def get_pipeline() -> PipelineService:
    return PipelineService()


def get_storage() -> FileStorage:
    return FileStorage()


def get_app_settings() -> Settings:
    return get_settings()


@router.get("/health", response_model=dict)
async def health(settings: Settings = Depends(get_app_settings)) -> dict:
    return {"status": "ok", "provider": settings.llm_provider.title(), "vector_db": "ChromaDB"}


@router.post("/upload/pdf", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    pipeline: PipelineService = Depends(get_pipeline),
    storage: FileStorage = Depends(get_storage),
) -> UploadResponse:
    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise exceptions.invalid_file_type()
    metadata = await storage.save_upload(file, "pdf")
    extraction = await pipeline.handle_pdf_upload(metadata.file_id)
    return UploadResponse(
        file_id=metadata.file_id,
        filename=metadata.filename,
        kind="pdf",
        page_count=extraction.pages,
    )


@router.post("/upload/audio", response_model=UploadResponse)
async def upload_audio(
    file: UploadFile = File(...),
    pipeline: PipelineService = Depends(get_pipeline),
    storage: FileStorage = Depends(get_storage),
) -> UploadResponse:
    if file.content_type not in {"audio/mpeg", "audio/mp3", "audio/mpeg3"}:
        raise exceptions.invalid_file_type()
    metadata = await storage.save_upload(file, "audio")
    transcription = await pipeline.handle_audio_upload(metadata.file_id)
    return UploadResponse(
        file_id=metadata.file_id,
        filename=metadata.filename,
        kind="audio",
        duration_seconds=transcription.duration_seconds,
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
