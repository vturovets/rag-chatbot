# RAG Chatbot

Retrieval-augmented generation (RAG) chatbot that couples a FastAPI backend with a Streamlit
frontend. Users can upload PDFs or MP3 files, have them chunked and embedded into a ChromaDB
vector store, and then ask questions that are answered using grounded context plus an LLM.

## Project structure

```
app/
├── backend/      # FastAPI application and supporting services
├── common/       # Shared utilities such as logging helpers
└── frontend/     # Streamlit UI for chat experience
fixtures/         # Sample documents and audio clips for local testing
tests/            # Automated test suite (pytest + HTTPX clients)
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
   RAG_OPENAI_API_KEY="sk-..."
   # Optional transcription model override (defaults to gpt-4o-mini-transcribe)
   # RAG_WHISPER_MODEL="gpt-4o-transcribe"
   # Optional overrides
   # RAG_GOOGLE_API_KEY="your-google-api-key"
   # RAG_LLM_PROVIDER="openai"  # or "google"
   # RAG_STORAGE_DIR="storage"
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

The backend exposes REST endpoints under `http://localhost:8000/api/...`. Logs and structured
error reporting are written to the console.

### Frontend (Streamlit UI)

Open a second IntelliJ terminal tab (PowerShell) and run:

```powershell
streamlit run app/frontend/app.py --server.port 8501 --server.headless true
```

The Streamlit app proxies requests to the backend using the base URL defined in `st.secrets` or
the default `http://localhost:8000`.

### Upload directory

Uploaded files and generated embeddings are stored under `storage/` (configurable through
`RAG_STORAGE_DIR`). Ensure the directory is writable before starting the backend.

### Audio transcription backends

When no external transcription API key is configured the backend falls back to
[`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) for local inference. The default
configuration now forces CPU execution (`RAG_LOCAL_TRANSCRIPTION_DEVICE=cpu` and
`RAG_LOCAL_TRANSCRIPTION_COMPUTE_TYPE=int8`) so Windows installations do not require CUDA or
cuDNN DLLs. If you have a working GPU toolchain you can opt back into accelerated inference with:

```powershell
setx RAG_LOCAL_TRANSCRIPTION_DEVICE cuda
setx RAG_LOCAL_TRANSCRIPTION_COMPUTE_TYPE float16
```

Restart your terminal after changing the environment variables so Uvicorn picks them up.

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

