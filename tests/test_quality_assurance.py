from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

import pytest

from app.backend import exceptions
from app.backend.services.extraction import ExtractionService, OpenAIError

FIXTURES_DIR = Path("fixtures")


def _open_fixture(name: str, mode: str = "rb"):
    return (FIXTURES_DIR / name).open(mode)


@pytest.fixture
def stub_transcription(monkeypatch):
    async def _fake_transcribe(self, path):
        assert path.exists()
        return (
            "This recording outlines the launch schedule and readiness review for the Mars habitat program."
        )

    monkeypatch.setattr(ExtractionService, "_transcribe_with_whisper", _fake_transcribe)


def test_pdf_ingestion_and_chat_latency(client):
    with _open_fixture("sample_image_text.pdf") as stream:
        response = client.post(
            "/upload/pdf",
            files={"file": ("sample_image_text.pdf", stream, "application/pdf")},
        )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["page_count"] >= 1

    chat_response = client.post(
        "/chat",
        json={"query": "What does GGUF stand for and how is it used?"},
    )
    assert chat_response.status_code == 200, chat_response.text
    chat_payload = chat_response.json()
    assert chat_payload["latency_ms"] < 15_000
    assert "GGUF" in chat_payload["answer"].upper()
    assert chat_payload["answer"].strip()


def test_audio_ingestion_and_retrieval(client, stub_transcription):
    with _open_fixture("sample_clean.mp3") as stream:
        response = client.post(
            "/upload/audio",
            files={"file": ("sample_clean.mp3", stream, "audio/mpeg")},
        )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["duration_seconds"] > 0

    chat_response = client.post(
        "/chat",
        json={"query": "Summarize the launch readiness update from the recording."},
    )
    assert chat_response.status_code == 200, chat_response.text
    answer = chat_response.json()["answer"].lower()
    assert "launch" in answer and "readiness" in answer


def test_extraction_service_uses_rag_openai_key(monkeypatch):
    from app.backend import config

    captured_kwargs: dict[str, str] = {}

    class DummyAsyncOpenAI:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setenv("RAG_OPENAI_API_KEY", "dummy-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("app.backend.services.extraction.AsyncOpenAI", DummyAsyncOpenAI)
    config.get_settings.cache_clear()

    try:
        service = ExtractionService()

        assert captured_kwargs.get("api_key") == "dummy-key"
        assert isinstance(service._openai_client, DummyAsyncOpenAI)
    finally:
        config.get_settings.cache_clear()


def test_extraction_service_reads_env_file(monkeypatch, tmp_path):
    from app.backend import config

    env_file = tmp_path / ".env"
    env_file.write_text('\n'.join([
        "# comment should be ignored",
        'RAG_OPENAI_API_KEY="file-key"',
    ]))

    captured_kwargs: dict[str, str] = {}

    class DummyAsyncOpenAI:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.delenv("RAG_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(config, "ENV_FILE", env_file)
    monkeypatch.setattr(config.Settings, "env_file", str(env_file))
    monkeypatch.setattr("app.backend.services.extraction.AsyncOpenAI", DummyAsyncOpenAI)
    config.get_settings.cache_clear()

    try:
        service = ExtractionService()

        assert captured_kwargs.get("api_key") == "file-key"
        assert isinstance(service._openai_client, DummyAsyncOpenAI)
    finally:
        config.get_settings.cache_clear()


def test_invalid_pdf_rejected(client, tmp_path):
    import fitz

    image_only_pdf = tmp_path / "image_only.pdf"
    document = fitz.open()
    document.new_page()  # empty page with no text layer
    document.save(image_only_pdf)
    document.close()

    with image_only_pdf.open("rb") as stream:
        response = client.post(
            "/upload/pdf",
            files={"file": ("image_only.pdf", stream, "application/pdf")},
        )
    assert response.status_code == 400
    payload = response.json()
    assert payload["error_code"] == "INVALID_PDF_STRUCTURE"
    assert "cannot be processed" in payload["message"].lower()


@pytest.mark.usefixtures("stub_transcription")
def test_debug_endpoint_requires_debug_mode(client, monkeypatch):
    from app.backend import config
    from app.backend.api import routes

    monkeypatch.setenv("RAG_DEBUG_MODE", "false")
    config.get_settings.cache_clear()
    routes.get_pipeline.cache_clear()
    routes.get_storage.cache_clear()
    routes.get_session_store.cache_clear()

    response = client.post(
        "/debug/pipeline",
        json={"file_id": str(uuid4())},
    )
    assert response.status_code == 401
    payload = response.json()
    assert payload["error_code"] == "UNAUTHORIZED_DEBUG"
    assert "development mode" in payload["message"].lower()


def test_debug_chunk_accepts_inline_text(client):
    response = client.post(
        "/debug/pipeline",
        params={"break_at": "chunk"},
        json={
            "text": "Term Plain definition Analogy GGUF models A new, efficient file format used to store and run AI language models",
            "chunk_size": 200,
            "chunk_overlap": 40,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["stages"]) == 1
    chunk_stage = payload["stages"][0]
    assert chunk_stage["stage"] == "chunk"
    input_payload = chunk_stage["input_payload"]
    assert input_payload["chunk_size"] == 200
    assert input_payload["overlap"] == 40
    output_payload = chunk_stage["output_payload"]
    assert output_payload["counts"]["chunks"] >= 1
    assert output_payload["chunks"], "Expected chunk previews in response"


def test_debug_embed_accepts_chunk_payload(client):
    seed_text = (
        "Term Plain definition Analogy GGUF models A new, efficient file format used to store and run AI language "
        "models like LLaMA with better speed."
    )
    chunk_response = client.post(
        "/debug/pipeline",
        params={"break_at": "chunk", "raw": True},
        json={
            "text": seed_text,
            "chunk_size": 120,
            "chunk_overlap": 20,
        },
    )
    assert chunk_response.status_code == 200, chunk_response.text
    chunk_payload = chunk_response.json()["stages"][0]["output_payload"]["chunks"]
    assert chunk_payload, "Expected chunk payload from chunk stage"

    embed_response = client.post(
        "/debug/pipeline",
        params={"break_at": "embed", "raw": True},
        json={"chunks": chunk_payload},
    )
    assert embed_response.status_code == 200, embed_response.text
    embed_payload = embed_response.json()
    assert len(embed_payload["stages"]) == 2
    embed_stage = embed_payload["stages"][-1]
    assert embed_stage["stage"] == "embed"
    vectors_info = embed_stage["output_payload"]["vectors"]
    assert vectors_info["count"] >= 1
    assert vectors_info["dim"] > 0


def test_chat_missing_query_validation(client):
    response = client.post(
        "/chat",
        json={"query": "   "},
    )
    assert response.status_code == 400
    payload = response.json()
    assert payload["error_code"] == "MISSING_QUERY"
    assert "required" in payload["message"].lower()


def test_transcription_falls_back_to_local_when_openai_rejects(monkeypatch):
    service = ExtractionService()

    class DummyTranscriptions:
        async def create(self, *args, **kwargs):
            raise OpenAIError("Unsupported file format from test")

    class DummyAudio:
        def __init__(self) -> None:
            self.transcriptions = DummyTranscriptions()

    class DummyClient:
        def __init__(self) -> None:
            self.audio = DummyAudio()

    service._openai_client = DummyClient()

    flag = {"local": False}

    async def _fake_local(self, path):
        flag["local"] = True
        return "local transcript"

    monkeypatch.setattr(ExtractionService, "_transcribe_locally", _fake_local)

    transcript = asyncio.run(service._transcribe_with_whisper(FIXTURES_DIR / "sample_clean.mp3"))

    assert transcript == "local transcript"
    assert flag["local"] is True


def test_local_transcription_reports_missing_ffmpeg(monkeypatch):
    service = ExtractionService()

    class DummyLocal:
        def transcribe(self, path):
            return "stub"

    def _raise_ffmpeg(*args, **kwargs):
        raise FileNotFoundError("ffmpeg not found")

    monkeypatch.setattr("app.backend.services.extraction.subprocess.run", _raise_ffmpeg)
    monkeypatch.setattr(ExtractionService, "_get_local_transcriber", lambda self: DummyLocal())

    with pytest.raises(exceptions.RagError) as excinfo:
        asyncio.run(service._transcribe_locally(FIXTURES_DIR / "sample_clean.mp3"))

    payload = excinfo.value.detail
    assert payload["error_code"] == "TRANSCRIPTION_ERROR"
    assert "ffmpeg" in payload.get("hint", "").lower()
