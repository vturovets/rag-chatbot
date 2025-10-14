import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure the application package is importable when tests run from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.backend.main import app
from app.backend import config
from app.backend.api import routes


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch, tmp_path):
    storage_root = tmp_path / "storage"
    monkeypatch.setenv("RAG_STORAGE_DIR", str(storage_root))
    monkeypatch.setenv("RAG_ENVIRONMENT", "dev")
    monkeypatch.setenv("RAG_DEBUG_MODE", "true")

    config.get_settings.cache_clear()
    routes.get_pipeline.cache_clear()
    routes.get_storage.cache_clear()
    routes.get_session_store.cache_clear()

    settings = config.get_settings()

    yield settings

    routes.get_pipeline.cache_clear()
    routes.get_storage.cache_clear()
    routes.get_session_store.cache_clear()
    config.get_settings.cache_clear()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
