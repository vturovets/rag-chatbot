Absolutely ✅ — here’s the **updated version of the “RAG Chatbot SRS (Codex Edition)”** that now includes a structured **Error Handling and Messages** section (Section 9).  
I’ve kept the messages consistent with REST conventions and informative for both developers (in logs/debug) and users (in Streamlit UI).  
All previous content is preserved and aligned with your latest agreements.

---

# **RAG Chatbot SRS (Codex Edition)**

## 1. Purpose and Context

The project delivers a **Retrieval-Augmented Generation (RAG) chatbot** that ingests **PDF presentations** and **audio recordings** to create a searchable knowledge base.  
It combines **retrieval** from a vector database with **generation** via a large language model (LLM), returning concise, contextually grounded answers.  
Target implementation: FastAPI backend + Streamlit UI (2 tabs: *Ingest* / *Chat*).

---

## 2. Business Value

| Objective                      | Business Value                                                        |
| ------------------------------ | --------------------------------------------------------------------- |
| Centralized knowledge access   | Unified conversational access to information from PDFs and MP3 files. |
| Reduced manual effort          | Automated extraction and transcription of documents and recordings.   |
| Improved knowledge utilization | Converts static files into a dynamic, queryable knowledge base.       |
| Scalable knowledge management  | Designed for multi-session operation and future extensibility.        |
| Enhanced decision-making       | Provides fast, contextually grounded responses.                       |

---

## 3. Roles and Responsibilities

| Role                                 | Responsibilities                                                               |
| ------------------------------------ | ------------------------------------------------------------------------------ |
| System Administrator / Data Engineer | Configure and maintain ingestion pipelines (PDF parsing, audio transcription). |
| ML Engineer                          | Implement chunking, embedding, and vector storage logic.                       |
| Backend Developer                    | Build and maintain API endpoints for ingestion, retrieval, and generation.     |
| Frontend Developer                   | Develop Streamlit interface (2 tabs, ephemeral sessions).                      |
| QA / Tester                          | Validate ingestion, retrieval, generation, and debug pipelines.                |
| End User                             | Upload files and query chatbot through web UI.                                 |

---

## 4. Functional Overview

### Workflow

1. **Upload Source Data** – User uploads `.pdf` or `.mp3` file.

2. **Extraction / Transcription** – System extracts text (PDF) or transcribes audio (Whisper).

3. **Chunking** – Text split into semantic units; token-based (size ≤ 500, overlap ≈ 60).

4. **Embedding Generation** – Chunks converted to embeddings (OpenAI or Google).

5. **Vector Storage** – Embeddings stored in **ChromaDB**, metric = cosine.

6. **User Query** – User enters a natural-language question.

7. **Context Retrieval** – Query embedded and matched to relevant chunks (`top_k = 5`).

8. **Answer Generation** – Retrieved chunks + query → LLM (OpenAI or Google) → concise answer.

9. **Delivery** – Answer returned without visible citations/snippets.

---

## 5. Functional and Non-Functional Requirements

### Functional

| Module                  | Requirement                                                          |
| ----------------------- | -------------------------------------------------------------------- |
| **Data Loader**         | Accepts `.pdf` and `.mp3` (reject image-only PDFs / other audio).    |
| **Text Extractor**      | Use `PyMuPDF` or `pdfplumber` for text-layer extraction.             |
| **Audio Transcriber**   | Use OpenAI Whisper API (large-v3).                                   |
| **Chunking Engine**     | Token-based with 10–20 % overlap (default 500 ± 60 tokens).          |
| **Embedding Generator** | Uses OpenAI or Google model; configurable per environment.           |
| **Vector DB**           | ChromaDB only; “index fingerprint = provider + model” enforced.      |
| **Retriever**           | Semantic similarity search (`k ≤ 8`, default 5).                     |
| **Generator**           | OpenAI/Google LLM; concise, neutral answers, no citations shown.     |
| **Frontend (UI)**       | Streamlit 2-tab app (Ingest, Chat); ephemeral 24 h retention banner. |
| **Backend (API)**       | FastAPI service exposing upload, chat, and debug endpoints.          |

### Non-Functional

| Category            | Requirement                                                  |
| ------------------- | ------------------------------------------------------------ |
| **Performance**     | Total query latency ≤ 15 s.                                  |
| **Scalability**     | Handles multiple concurrent chat sessions.                   |
| **Reliability**     | Stable under multi-session load.                             |
| **Maintainability** | Configurable chunk size, model provider, limits.             |
| **Usability**       | Simple Streamlit interface; clear validation/error messages. |
| **Security**        | Debug endpoints only available in DEV (`DEBUG_MODE = True`). |

---

## 6. Configuration and Limits

| Parameter         | Default                                              | Description                   |
| ----------------- | ---------------------------------------------------- | ----------------------------- |
| `CHUNK_SIZE`      | 500                                                  | Tokens per chunk.             |
| `CHUNK_OVERLAP`   | 60                                                   | Token overlap between chunks. |
| `LLM_PROVIDER`    | OpenAI                                               | Options: OpenAI, Google.      |
| `EMBEDDING_MODEL` | `text-embedding-3-large`                             | Default embedding model.      |
| `VECTOR_DB`       | ChromaDB                                             | Fixed option.                 |
| `TOP_K`           | 5                                                    | Retrieved context items.      |
| `MAX_UPLOAD_MB`   | 200                                                  | Max file size.                |
| `MAX_PDF_PAGES`   | 200                                                  | Page limit per PDF.           |
| `MAX_AUDIO_MIN`   | 60                                                   | Max audio length (minutes).   |
| `TIMEOUTS`        | extract 5 s / transcribe 8 s / retrieve+generate 6 s | Stage timeouts.               |

---

## 7. API and Debug Interfaces

**Base URL:** `http://localhost:8000`  
**Environment guard:** active only if `DEBUG_MODE=True`.

### 7.1 Health

`GET /health` → `{ "status": "ok", "provider": "OpenAI", "vector_db": "ChromaDB" }`

### 7.2 Upload

| Endpoint        | Method | Description                                      |
| --------------- | ------ | ------------------------------------------------ |
| `/upload/pdf`   | `POST` | Accept text-layer PDF. Reject image-only PDFs.   |
| `/upload/audio` | `POST` | Accept MP3 only (English Whisper transcription). |

### 7.3 Chat

`POST /chat`  
**Body:** `{ "query": "string" }`  
**Response:** `{ "answer": "string", "latency_ms": n }`

### 7.4 Debug Pipeline (DEV only)

`POST /debug/pipeline?break_at=<stage>`  
Stages: `extract | chunk | embed | retrieve | generate`

| Stage             | Input                                                        | Output                                                              |
| ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------------- |
| `extract`         | `{ "file_id":"UUID", "file_type":"pdf" }`                    | `{ "text":"...", "pages":12 }`                                      |
| `extract (audio)` | `{ "file_id":"UUID", "file_type":"audio", "language":"en" }` | `{ "transcript":"..." }`                                            |
| `chunk`           | `{ "text":"...", "chunk_size":500, "overlap":60 }`           | `{ "chunks":[...], "counts":{...} }`                                |
| `embed`           | `{ "chunks":[{"id":"c1","text":"..."}] }`                    | `{ "vectors":{ "count":N,"dim":1536 }, "index_fingerprint":"..." }` |
| `retrieve`        | `{ "query":"...", "top_k":5 }`                               | `{ "hits":[{"id":"c1","score":0.78},...] }`                         |
| `generate`        | `{ "query":"...", "context":["chunk1","chunk2"] }`           | `{ "prompt":"...", "answer":"..." }`                                |

All responses omit raw vectors by default (`?raw=true` for dev only).

**Fixtures:**

- `fixtures/sample_image_text.pdf`, `fixtures/sample_image.pdf` (negative test)

- `fixtures/sample_clean.mp3`  
  **Postman Collection:** `/docs/debug-collection.json`

---

## 8. Operational Notes

1. **UI framework:** Streamlit (2 tabs: Ingest, Chat).

2. **Provider defaults:** OpenAI embeddings + OpenAI LLM.

3. **Index isolation:** enforce `index_fingerprint = provider + model`; block cross-use.

4. **OCR:** Not supported (text-layer PDFs only).

5. **Audio:** MP3 only; prefer better quality even if slower (< 15 s).

6. **Telemetry:** local JSON logs with ingestion time, chunk/embedding counts, latency breakdown, precision@K spot checks.

7. **Rate limits:** environment-level guardrails with exponential backoff.

8. **Session management:** ephemeral Streamlit session state (auto-purge ≤ 24 h).

9. **Testing fixtures:** sample PDF + MP3 for automated end-to-end tests (latency ≤ 15 s).

10. **Prompt tone:** concise, professional, no explicit citations.

---

## 9. Error Handling and Messages

All errors follow JSON format:  
`{ "error_code": "string", "message": "human-readable explanation", "hint": "optional" }`

| HTTP Code | Error Code              | Message (User-Facing)                                       | Typical Cause / Notes                      |
| --------- | ----------------------- | ----------------------------------------------------------- | ------------------------------------------ |
| **400**   | `INVALID_FILE_TYPE`     | “Unsupported file format. Please upload a PDF or MP3 file.” | Wrong extension or MIME type.              |
| **400**   | `INVALID_PDF_STRUCTURE` | “This PDF contains no text layer and cannot be processed.”  | Image-only PDF rejected.                   |
| **400**   | `FILE_TOO_LARGE`        | “File exceeds size limit (200 MB).”                         | Enforced by config.                        |
| **400**   | `AUDIO_TOO_LONG`        | “Audio exceeds 60-minute limit.”                            | Duration validation.                       |
| **400**   | `MISSING_QUERY`         | “Query text is required.”                                   | Empty `/chat` body.                        |
| **401**   | `UNAUTHORIZED_DEBUG`    | “Debug endpoints are only available in development mode.”   | `DEBUG_MODE=False`.                        |
| **404**   | `FILE_NOT_FOUND`        | “Referenced file not found or expired.”                     | Purged or invalid file ID.                 |
| **408**   | `TIMEOUT_STAGE`         | “Processing timed out while .”                              | Stage exceeded configured timeout.         |
| **422**   | `EMBEDDING_ERROR`       | “Embedding generation failed.”                              | Provider API error / invalid payload.      |
| **422**   | `TRANSCRIPTION_ERROR`   | “Audio transcription failed.”                               | Whisper API failure or bad input.          |
| **429**   | `RATE_LIMIT_EXCEEDED`   | “Service is temporarily busy. Please try again later.”      | Provider rate limit triggered.             |
| **500**   | `INTERNAL_ERROR`        | “Unexpected server error. Please retry or contact support.” | Generic fallback for unhandled exceptions. |
| **502**   | `LLM_PROVIDER_DOWN`     | “Model provider unavailable. Try again shortly.”            | Provider API outage.                       |
| **503**   | `VECTOR_DB_UNAVAILABLE` | “Vector database temporarily unreachable.”                  | DB connection error.                       |
| **504**   | `GENERATION_TIMEOUT`    | “Response generation took too long.”                        | LLM exceeded timeout.                      |

**Developer Logs (internal):** include stack traces, request IDs, and latency metrics (JSON-structured).  
**User UI Feedback (Streamlit):** simplified, friendly messages; technical details hidden unless `DEBUG_MODE=True`.

---

**End of Document**
