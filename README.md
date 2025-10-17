# RAG Chatbot

A production-ready retrieval-augmented generation (RAG) assistant composed of a FastAPI
backend and a Streamlit frontend. Users upload PDF slide decks or MP3 recordings, the backend
extracts or transcribes the content, chunks and embeds the text into a ChromaDB vector store,
and serves grounded answers via the configured LLM provider. The implementation ships with
structured logging, deterministic fallbacks, admin tooling, and an end-to-end test suite.

## Key capabilities

- **Dual ingestion pipeline** – Extracts text from PDFs (PyMuPDF + pdfplumber) and transcribes
  MP3 files via OpenAI Whisper with optional local `faster-whisper` fallback.
- **Configurable embeddings** – Supports OpenAI and Google Generative AI embedding models with
  deterministic hashing fallback when APIs are unreachable.
- **Context-aware retrieval** – Stores chunks in ChromaDB (persistent by default) and routes
  similarity search between PDF and audio sources based on query keywords.
- **LLM generation with guardrails** – Calls OpenAI or Google chat models with automatic retry,
  timeout handling, and a deterministic summariser fallback for offline development.
- **Ephemeral sessions** – Associates uploads with chat sessions for 24 hours, ensuring queries
  only reference authorised documents.
- **Operational tooling** – Request ID middleware, structured JSON logs, pipeline debugging
  endpoint, and an admin purge action that can optionally trigger a process restart.

## Project structure

```
app/
├── backend/      # FastAPI application (routers, services, config, models)
├── common/       # Shared utilities such as chunking helpers and logging
└── frontend/     # Streamlit UI for ingestion, chat, and debug views
docs/             # Project documentation and Postman collection
fixtures/         # Sample documents and audio clips for automated testing
tests/            # Pytest suite covering ingestion, chat, and admin flows
```

## Prerequisites

- Python 3.10 or newer
- (Recommended) [virtualenv](https://virtualenv.pypa.io/) or
  [uv](https://github.com/astral-sh/uv) / [pipenv](https://pipenv.pypa.io/en/latest/)
- An OpenAI or Google Generative AI API key, depending on the provider you select

## Initial setup

1. **Create and activate a virtual environment**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install dependencies**

   You can install with `pip` from `requirements.txt` (fully pinned) or from the
   `pyproject.toml` metadata. The simplest option is:

   ```powershell
   pip install -r requirements.txt
   ```

3. **Provide environment configuration**

   Copy `.env.example` to `.env` if available, or create a new `.env` file in the project root
   with at least the API credentials you plan to use:

   ```env
   # Core credentials (set at least one provider)
   RAG_OPENAI_API_KEY="sk-..."
   # RAG_GOOGLE_API_KEY="..."

   # Optional overrides
   # RAG_LLM_PROVIDER="openai"    # or "google"
   # RAG_LLM_MODEL="gpt-4o-mini"
   # RAG_EMBEDDING_MODEL="text-embedding-3-large"
   # RAG_STORAGE_DIR="storage"     # location for uploaded files + ChromaDB
   # RAG_MAX_AUDIO_MINUTES="90"    # reject longer recordings during ingestion
   # RAG_DEBUG_MODE="true"         # enables debug endpoints (default true in dev)
   # RAG_LOCAL_TRANSCRIPTION_ONLY="false"  # force faster-whisper offline mode
   ```

   Any `RAG_*` variables are picked up automatically by the backend via Pydantic settings.

## Running the applications locally

You can run the backend and frontend in separate terminals. The commands below work in the
IntelliJ IDEA integrated terminal configured to use PowerShell.

### Backend (FastAPI + Uvicorn)

```powershell
# From the project root inside the activated virtual environment
uvicorn app.backend.main:app --reload --host 0.0.0.0 --port 8000
```

The backend exposes REST endpoints under `http://localhost:8000`. Every request receives an
`X-Request-ID` header and emits structured JSON logs detailing timings, chunk counts, and
errors.

### Frontend (Streamlit UI)

Open a second IntelliJ terminal tab (PowerShell) and run:

```powershell
streamlit run app/frontend/app.py --server.port 8501 --server.headless true
```

The Streamlit app proxies requests to the backend using the base URL defined in `st.secrets`
(`api_base`) or the default `http://localhost:8000`. The UI provides dedicated tabs for
ingestion and chat, displays upload retention windows, and surfaces friendly error messages.

### Upload directory

Uploaded files and generated embeddings are stored under `storage/` (configurable through
`RAG_STORAGE_DIR`). Ensure the directory is writable before starting the backend.

### Audio transcription backends

When no external transcription API key is configured the backend falls back to
[`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) for local inference. The default
configuration pins CPU execution (`RAG_LOCAL_TRANSCRIPTION_DEVICE=cpu` and
`RAG_LOCAL_TRANSCRIPTION_COMPUTE_TYPE=int8`) so Windows installations do not require CUDA or
cuDNN DLLs. If you have a working GPU toolchain you can opt back into accelerated inference with:

```powershell
setx RAG_LOCAL_TRANSCRIPTION_DEVICE cuda
setx RAG_LOCAL_TRANSCRIPTION_COMPUTE_TYPE float16
```

Restart your terminal after changing the environment variables so Uvicorn picks them up.

## API quick reference

| Endpoint | Method | Description |
| -------- | ------ | ----------- |
| `/health` | GET | Returns provider, environment, and optional limit diagnostics (`?raw=true` in debug mode). |
| `/upload/pdf` | POST | Accepts text-layer PDFs, extracts pages, chunks text, and indexes it. |
| `/upload/audio` | POST | Accepts MP3 uploads, transcribes audio (OpenAI Whisper or local fallback), and indexes the transcript. |
| `/chat` | POST | Generates an answer grounded in the retrieved context for the active session. |
| `/debug/pipeline` | POST | (Debug only) Runs the ingestion pipeline step-by-step with optional payload introspection. |
| `/admin/purge` | POST | (Debug only) Clears stored files, vector index, and session state; optionally schedules backend restart. |

See `docs/RAG_Chatbot_SRS_rev3.md` for complete request/response schemas and operational notes.

## Configuration matrix

| Environment variable | Default | Purpose |
| -------------------- | ------- | ------- |
| `RAG_ENVIRONMENT` | `dev` | Controls behaviour suitable for development vs production. |
| `RAG_DEBUG_MODE` | `true` | Enables debug endpoints and verbose validation hints. |
| `RAG_STORAGE_DIR` | `storage` | Root directory for uploads, metadata, and ChromaDB persistence. |
| `RAG_FILE_RETENTION_HOURS` | `24` | Hours before uploaded files expire. |
| `RAG_SESSION_RETENTION_HOURS` | `24` | Hours before chat sessions expire. |
| `RAG_CHUNK_SIZE` | `500` | Target tokens per chunk. |
| `RAG_CHUNK_OVERLAP` | `60` | Token overlap between chunks. |
| `RAG_TOP_K` | `5` | Default retrieval depth (capped at 8). |
| `RAG_MAX_UPLOAD_MB` | `200` | Maximum upload size accepted by the backend. |
| `RAG_MAX_PDF_PAGES` | `200` | Hard limit for PDF page count. |
| `RAG_MAX_AUDIO_MINUTES` | `90` | Maximum audio duration permitted for ingestion. |
| `RAG_LLM_PROVIDER` | `openai` | Provider for generation (`openai`, `google`). |
| `RAG_LLM_MODEL` | `gpt-4o-mini` | Chat model identifier used with the provider. |
| `RAG_EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model identifier. |
| `RAG_OPENAI_API_KEY` | _unset_ | Credential for OpenAI APIs (embeddings, Whisper, chat). |
| `RAG_OPENAI_API_BASE` | _unset_ | Optional custom OpenAI base URL. |
| `RAG_GOOGLE_API_KEY` | _unset_ | Credential for Google Generative AI APIs. |
| `RAG_WHISPER_MODEL` | `gpt-4o-mini-transcribe` | Whisper transcription model for audio ingestion. |
| `RAG_LOCAL_TRANSCRIPTION_MODEL` | `base` | Local faster-whisper model size. |
| `RAG_LOCAL_TRANSCRIPTION_DEVICE` | `cpu` | Device hint for faster-whisper. |
| `RAG_LOCAL_TRANSCRIPTION_COMPUTE_TYPE` | `int8` | Compute type for faster-whisper. |
| `RAG_LOCAL_TRANSCRIPTION_ONLY` | `false` | Force local transcription instead of OpenAI Whisper. |
| `RAG_AUTO_RESTART_ON_PURGE` | `true` | Schedule backend restart after purge completes. |
| `RAG_RESTART_GRACE_SECONDS` | `0.5` | Delay before triggering the optional restart action. |

All settings also load from a `.env` file in the project root when present.

## Running tests

```powershell
pytest
```

The suite relies on the virtual environment being active. Sample fixtures in the `fixtures`
folder provide deterministic inputs for unit and integration tests.

## Helpful IntelliJ IDEA tips

- Configure the Python SDK to point to the `.venv` interpreter (File → Settings → Project
  → Python Interpreter).
- Mark the `app` directory as a sources root so imports like `from app.backend...` resolve
  without additional `PYTHONPATH` tweaks.
- Use the "Run/Debug Configuration" templates for Uvicorn and Streamlit to launch the
  services directly from the IDE if you prefer not to use the terminal.

## Troubleshooting

### "Could not locate cudnn_ops64_9.dll" on Windows

This error indicates that PyTorch or TensorFlow attempted to load CUDA/cuDNN kernels but
could not find the `cudnn_ops64_9.dll` runtime in your `PATH`. To resolve it:

1. **Install the matching cuDNN build.**
   - Ensure the CUDA toolkit version you installed matches the version required by your
     deep-learning framework (check the framework's "CUDA support" table).
   - Download the corresponding cuDNN 9 zip from the [NVIDIA Developer downloads](https://developer.nvidia.com/cudnn-downloads) and extract it locally.
2. **Copy the cuDNN binaries into your CUDA toolkit.** From the extracted package copy the
   contents of the `bin`, `include`, and `lib` folders into
   `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\` (replace `vXX.X` with your
   installed CUDA version). Allow Windows to overwrite older files if prompted.
3. **Update the system PATH.** Add the CUDA `bin` directory (for example,
   `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin`) to the *System* PATH so
   shells, IDEs, and services that launch Python inherit the location of the DLLs.
4. **Restart the terminal or IDE** to ensure the updated PATH is picked up, then retry your
   training or inference command.

If you do not intend to run GPU-accelerated workloads, install the CPU-only build of your
framework to bypass the cuDNN dependency entirely. For PyTorch you can run:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Pin the version as needed (for example, `pip install torch==2.3.1 --index-url ...`) to match
your project's compatibility requirements.

