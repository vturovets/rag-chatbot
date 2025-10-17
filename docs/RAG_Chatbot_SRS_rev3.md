# RAG Chatbot SRS (Implementation Snapshot)

This specification describes the delivered RAG Chatbot system as implemented in the
`rag-chatbot` repository. It captures the architecture, functional scope, configuration, and
operational behaviour of the FastAPI backend and Streamlit frontend that jointly provide the
retrieval‑augmented experience.

---

## 1. Purpose and Context

- Provide an internal assistant that answers questions grounded in uploaded slide decks (PDF) and
  recorded briefings (MP3).
- Support hybrid ingestion, retrieval, and generation while maintaining deterministic fallbacks so
  the product remains usable in constrained environments.
- Offer development tooling (debug pipeline, admin purge, structured logs) to aid operations and
  incident response.

---

## 2. System Overview

| Component | Description |
| --------- | ----------- |
| FastAPI backend (`app/backend`) | Exposes REST endpoints, orchestrates the ingestion and chat pipeline, enforces limits, and translates provider failures into first-class errors. |
| Extraction service | Uses PyMuPDF + pdfplumber to read PDFs and OpenAI Whisper (with optional `faster-whisper` fallback) to transcribe MP3 audio. |
| Chunking + embeddings | Normalises text via `chunk_text`, generates embeddings through OpenAI or Google providers, and falls back to deterministic hashing when APIs are unreachable. |
| Vector store | Persists embeddings in ChromaDB keyed by provider/model fingerprint; provides in-memory storage when Chroma is not installed. |
| Generation service | Calls OpenAI or Google chat models with retry/backoff and degrades to a deterministic summariser when external providers are unavailable. |
| Session store | Tracks uploaded file associations per chat session with 24 h retention. |
| Streamlit frontend (`app/frontend`) | Presents ingestion and chat tabs, surfaces request progress, displays retention windows, and maps error codes to friendly copy. |
| Storage directory (`storage/`) | Retains uploads, extracted metadata (`*.json`), and ChromaDB persistence, auto-purged after retention expires. |

---

## 3. Actors and Responsibilities

| Actor | Responsibilities |
| ----- | ---------------- |
| End user | Upload PDFs/MP3s, initiate chat sessions, review answers via the Streamlit UI. |
| Backend developer | Maintain FastAPI routes, pipeline orchestration, and error translation. |
| ML engineer | Tune chunking/embedding configuration, manage provider selection, validate retrieval quality. |
| Data/Platform engineer | Provision API keys, manage storage volume, monitor logs, and coordinate restarts triggered by purge actions. |
| QA engineer | Execute automated and manual regression tests using the fixtures and Postman collection. |

---

## 4. Functional Requirements

### 4.1 File ingestion
- Accept uploads through `/upload/pdf` and `/upload/audio` with multipart form payloads.
- Validate MIME type or file extension (`.pdf`, `.mp3`). Reject image-only PDFs (`INVALID_PDF_STRUCTURE`) and non‑MP3 audio (`INVALID_FILE_TYPE`).
- Enforce size and duration limits (`RAG_MAX_UPLOAD_MB`, `RAG_MAX_PDF_PAGES`, `RAG_MAX_AUDIO_MINUTES`). Default audio limit is 90 minutes; exceeding it returns `AUDIO_TOO_LONG`.
- Persist the raw file and metadata JSON under `storage/<uuid>`; set `expires_at = uploaded_at + file_retention_hours` (24 h default).
- Return `UploadResponse` including `file_id`, derived `source`, and session association.

### 4.2 Extraction and transcription
- PDFs: use PyMuPDF (`fitz`) to iterate pages, supplement with pdfplumber for resilience, and combine page text. Abort if the concatenated text is empty.
- Audio: use `mutagen` to inspect MP3 metadata, validate duration, then call OpenAI Whisper via `AsyncOpenAI.audio.transcriptions.create` (text response). Retry with exponential backoff on rate limits, and fall back to local transcription when configured.
- Local transcription: convert MP3 to 16 kHz mono WAV via `ffmpeg`, run `faster-whisper` using configured device/compute type, and clean up temporary files.

### 4.3 Chunking and embeddings
- Normalise whitespace and chunk text using `ChunkConfig(chunk_size=500, chunk_overlap=60)` with a tokenizer that prefers `tiktoken` and falls back to whitespace tokenisation.
- Wrap chunks as `Chunk` objects (tracking `chunk_id`, source file, order).
- Generate embeddings using the configured provider:
  - **OpenAI**: `text-embedding-3-large` by default, with retry/backoff on rate limits and provider errors.
  - **Google Generative AI**: calls `genai.embed_content` when configured.
  - **Fallback**: deterministic SHA-256 hash mapped to a unit vector (ensures deterministic behaviour offline).
- Persist embeddings to the vector store via `VectorStore.upsert_vectors`, storing metadata (source file, order, provider fingerprint).

### 4.4 Retrieval and ranking
- When `/chat` is invoked, retrieve the session context (`SessionStore`). Reject empty queries (`MISSING_QUERY`).
- Determine retrieval plan based on `_route_query_sources` keywords to balance PDF vs audio sources.
- Perform similarity search through ChromaDB with optional source filtering and deduplicate chunk hits.
- Limit retrieval depth to `top_k ≤ 8`; fallback to global search if routed hits are empty.
- Record precision sample metrics and latency breakdown in structured logs.

### 4.5 Prompting and generation
- Assemble prompts by flattening retrieved chunks (max 8) into a bullet list. When no context exists, instruct the LLM to acknowledge the lack of evidence.
- Generate answers via provider-specific adapters:
  - Retry/backoff on rate limits and transient connectivity problems.
  - Enforce per-stage timeouts using configuration (`retrieval_timeout_s`, `prompt_timeout_s`, `generation_timeout_s`).
  - Map provider exceptions to domain errors (`RATE_LIMIT_EXCEEDED`, `LLM_PROVIDER_DOWN`, `GENERATION_TIMEOUT`, etc.).
- Ensure responses are concise (≤3 sentences) and citation-free. Return `ChatResponse` with `answer`, `latency_ms`, and `session_id` (to maintain continuity across UI refreshes).
- Fall back to `LocalFallbackProvider` that summarises available snippets if remote providers fail.

### 4.6 Session and retention management
- `SessionStore` associates uploaded file IDs with a session UUID (provided by client or generated server-side).
- Sessions expire after `session_retention_hours` (24 h default) and are lazily purged before use.
- Successful chat responses append newly retrieved file IDs to the session context.

### 4.7 Debugging and administration
- `/debug/pipeline` (debug mode only) replays the pipeline stage-by-stage (`extract → chunk → embed → retrieve → generate`), returning diagnostics for each step. Accepts inline chunks to bypass extraction and optionally emits raw payloads when `raw=true`.
- `/admin/purge` (debug mode only) clears stored files, vector index, and session cache, then invokes `BackendLifecycle.schedule_restart()` when `RAG_AUTO_RESTART_ON_PURGE=true`.

### 4.8 Frontend behaviour
- Streamlit app provides tabs for **Ingest** (file uploader, status indicator, retention summary) and **Chat** (question input, streaming progress messages, answer display, history of ingested items).
- Friendly messages map backend error codes to user-facing copy; technical details and HTTP metadata are surfaced only when `st.secrets["debug_mode"]` is true.
- UI stores ingested file records in Streamlit session state to support subsequent chats without re-uploading within the retention window.

---

## 5. Non-Functional Requirements

| Category | Requirement |
| -------- | ----------- |
| Performance | End-to-end chat latency must remain under 15 seconds (validated in tests). Per-stage timeouts configurable via settings. |
| Scalability | Concurrent requests supported through async FastAPI handlers and thread executors for CPU-bound work. Vector store operations guarded by re-entrant locks. |
| Reliability | Deterministic fallbacks for embeddings and generation maintain baseline functionality when providers are unavailable. File/session retention automatically purges stale artefacts. |
| Observability | Structured JSON logs (`json_log`) emit ingestion counts, timings, and precision metrics. Request ID middleware injects `X-Request-ID` into all responses. |
| Security | Debug endpoints require `RAG_DEBUG_MODE=true` and should be disabled in production. Upload validation prevents unsupported file types and oversize payloads. |
| Maintainability | Configuration centralised in `Settings` (Pydantic), with `.env` support and caching. Services designed for dependency injection to enable testing and overrides. |

---

## 6. Configuration and Limits

| Setting | Default | Description |
| ------- | ------- | ----------- |
| `environment` | `dev` | Runtime environment label. |
| `debug_mode` | `True` | Enables debug endpoints and richer validation hints. |
| `storage_dir` | `storage` | Location for uploads, metadata, and Chroma persistence. |
| `file_retention_hours` | `24` | Hours to retain uploaded files before expiry. |
| `session_retention_hours` | `24` | Hours to retain session context. |
| `chunk_size` | `500` | Target tokens per chunk. |
| `chunk_overlap` | `60` | Token overlap between consecutive chunks. |
| `top_k` | `5` | Default retrieval depth (capped at 8). |
| `max_upload_mb` | `200` | Maximum upload size in megabytes. |
| `max_pdf_pages` | `200` | Maximum allowed pages per PDF. |
| `max_audio_minutes` | `90` | Maximum MP3 duration accepted for transcription. |
| `llm_provider` | `openai` | Generation provider (`openai` / `google`). |
| `llm_model` | `gpt-4o-mini` | Chat model identifier. |
| `embedding_model` | `text-embedding-3-large` | Embedding model identifier. |
| `whisper_model` | `gpt-4o-mini-transcribe` | Whisper transcription model for audio ingestion. |
| `retrieval_timeout_s` | `5.0` | Timeout for similarity search. |
| `prompt_timeout_s` | `5.0` | Timeout for prompt assembly. |
| `generation_timeout_s` | `30.0` | Timeout for LLM generation. |
| `transcription_timeout_s` | `600.0` | Timeout for audio transcription (remote or local). |
| `auto_restart_on_purge` | `True` | Whether to restart backend after purge completes. |
| `restart_grace_seconds` | `0.5` | Delay before executing restart action. |

All settings are exposed via environment variables prefixed with `RAG_` and load from `.env` when present.

---

## 7. API Contract Summary

### 7.1 `GET /health`
- **Query params**: `raw` (bool, default `false`) to include configuration limits (requires debug mode).
- **Response**:
  ```json
  {
    "status": "ok",
    "provider": "OpenAI",
    "vector_db": "ChromaDB",
    "environment": "dev",
    "debug_mode": true,
    "limits": {
      "chunk_size": 500,
      "chunk_overlap": 60,
      "top_k": 5,
      "max_upload_mb": 200,
      "max_pdf_pages": 200,
      "max_audio_minutes": 90
    }
  }
  ```

### 7.2 `POST /upload/pdf`
- **Request**: multipart form with `file` (PDF) and optional `session_id` (UUID string).
- **Response**: `UploadResponse`
  ```json
  {
    "file_id": "...",
    "filename": "slides.pdf",
    "kind": "pdf",
    "source": "pdf",
    "page_count": 12,
    "expires_at": "2024-07-01T18:23:45.000Z",
    "session_id": "..."
  }
  ```

### 7.3 `POST /upload/audio`
- **Request**: multipart form with MP3 `file` and optional `session_id`.
- **Response**: `UploadResponse`
  ```json
  {
    "file_id": "...",
    "filename": "briefing.mp3",
    "kind": "audio",
    "source": "audio",
    "duration_seconds": 354.2,
    "expires_at": "2024-07-01T18:23:45.000Z",
    "session_id": "..."
  }
  ```

### 7.4 `POST /chat`
- **Request body**: `{"query": "string", "session_id": "optional UUID", "top_k": optional int}`.
- **Response**:
  ```json
  {
    "answer": "Concise answer grounded in the retrieved context.",
    "latency_ms": 3120,
    "session_id": "..."
  }
  ```

### 7.5 `POST /debug/pipeline`
- **Guard**: requires `RAG_DEBUG_MODE=true`.
- **Request body**: `DebugPipelineRequest` supporting either `file_id` or inline `text/chunks` payloads. Query params: `break_at` (`extract|chunk|embed|retrieve|generate`), `raw` (bool).
- **Response**: `DebugPipelineResponse` containing a list of `PipelineStageDiagnostics` entries with `stage`, `input_payload`, and `output_payload` (optionally raw vectors).

### 7.6 `POST /admin/purge`
- **Guard**: requires `RAG_DEBUG_MODE=true`.
- **Response**: `{ "status": "purged" }`. When `auto_restart_on_purge` is true, schedules an application restart after `restart_grace_seconds`.

---

## 8. Data Management

- **Storage layout**: each upload stored at `storage/<uuid>` with metadata at `storage/<uuid>.json`.
- **Vector persistence**: ChromaDB collections stored under `storage/chroma/` using fingerprint-based names (provider + embedding model). When Chroma is unavailable the store operates in-memory.
- **Retention**: `FileStorage` and `SessionStore` lazily purge expired artefacts on each operation. `/admin/purge` forces immediate cleanup.
- **Security considerations**: uploads are not encrypted at rest; restrict filesystem permissions and ensure the storage directory is not world-readable in production.

---

## 9. Logging, Monitoring, and Error Handling

- **Request tracing**: `RequestIDMiddleware` injects a UUID as `X-Request-ID`; value is reused if the client supplies one.
- **Structured logging**: `json_log` helper emits events such as `ingestion.complete` and `chat.completed` with counts, timings, provider info, and precision samples.
- **Error payload format**:
  ```json
  {
    "error_code": "STRING",
    "message": "Human readable description",
    "hint": "Optional developer hint"
  }
  ```
- **Standard error catalogue**:

| HTTP Code | Error Code | Description / Typical cause |
| --------- | ---------- | --------------------------- |
| 400 | `INVALID_FILE_TYPE` | Unsupported file format or incorrect MIME type. |
| 400 | `INVALID_PDF_STRUCTURE` | PDF lacks a text layer after extraction. |
| 400 | `FILE_TOO_LARGE` | Upload exceeds `max_upload_mb`. |
| 400 | `AUDIO_TOO_LONG` | MP3 duration exceeds `max_audio_minutes`. |
| 400 | `INVALID_DEBUG_STAGE` | Unsupported stage requested in `/debug/pipeline`. |
| 400 | `MISSING_QUERY` | Chat request contains an empty query. |
| 401 | `UNAUTHORIZED_DEBUG` | Debug endpoint invoked while `debug_mode` is false. |
| 404 | `FILE_NOT_FOUND` | File ID not found or expired. |
| 404 | `RESOURCE_NOT_FOUND` | Generic 404 for other routes/resources. |
| 408 | `TIMEOUT_STAGE` | Stage exceeded configured timeout (extract/transcribe/prompt/retrieve). |
| 422 | `INVALID_REQUEST` | Validation failure with optional hint in debug mode. |
| 422 | `EMBEDDING_ERROR` | Embedding provider returned an error. |
| 422 | `TRANSCRIPTION_ERROR` | Whisper/local transcription failed. |
| 429 | `RATE_LIMIT_EXCEEDED` | Provider rate limit hit (embeddings, generation, transcription). |
| 500 | `INTERNAL_ERROR` | Unhandled server error. |
| 502 | `LLM_PROVIDER_DOWN` | LLM provider unreachable. |
| 503 | `VECTOR_DB_UNAVAILABLE` | Vector store query failed. |
| 504 | `GENERATION_TIMEOUT` | LLM exceeded configured timeout. |

Front-end copies mirror these messages and append contextual hints when debug mode is active.

---

## 10. Testing and Quality Assurance

- Automated pytest suite (`tests/test_quality_assurance.py`) validates:
  - PDF ingestion → chat latency (<15 s) and contextual accuracy.
  - Audio ingestion with stubbed transcription and retrieval quality.
  - Admin purge endpoint (including storage cleanup and lifecycle restart trigger).
  - Query routing heuristics and transcript cleaning behaviour.
  - Configuration sourcing from environment variables and `.env` files.
- Fixtures (`fixtures/sample_image_text.pdf`, `fixtures/sample_clean.mp3`) provide deterministic content for integration tests.
- Postman collection (`docs/RAG Chatbot Debug Pipeline.postman_collection.json`) documents debug endpoints for manual validation.

---

## 11. Deployment Considerations

- Run backend and frontend separately; ensure `storage/` is writable.
- Provide required API keys via environment variables or `.env`. If operating in an offline environment, install `faster-whisper` and enable local transcription and the deterministic generation fallback suffices.
- Disable debug mode (`RAG_DEBUG_MODE=false`) in production to protect admin endpoints.
- Monitor structured logs for ingestion metrics, latency, and error spikes; integrate with log aggregation by parsing JSON payloads.

---

**End of Document**
