from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from app.backend.services.extraction import ExtractionService

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
    detail = response.json()["detail"]
    assert detail["error_code"] == "INVALID_PDF_STRUCTURE"
    assert "cannot be processed" in detail["message"].lower()


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
    detail = response.json()["detail"]
    assert detail["error_code"] == "UNAUTHORIZED_DEBUG"
    assert "development mode" in detail["message"].lower()


def test_chat_missing_query_validation(client):
    response = client.post(
        "/chat",
        json={"query": "   "},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["error_code"] == "MISSING_QUERY"
    assert "required" in detail["message"].lower()
