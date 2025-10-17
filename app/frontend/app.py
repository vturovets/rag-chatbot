"""Streamlit application for the RAG chatbot."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Protocol

import requests
import streamlit as st


def _secret_get(key: str, default: Any) -> Any:
    try:
        value = st.secrets.get(key, default)
    except Exception:  # pragma: no cover - depends on runtime secrets configuration
        return default
    return default if value is None else value


def _secret_bool(key: str, default: bool) -> bool:
    value = _secret_get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _secret_int(key: str, default: int) -> int:
    value = _secret_get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return default


API_BASE = str(_secret_get("api_base", "http://localhost:8000"))
DEBUG_MODE = _secret_bool("debug_mode", False)
RETENTION_HOURS = _secret_int("retention_hours", 24)
# Allow long-running ingestion requests (e.g. large audio transcriptions).
REQUEST_TIMEOUT = _secret_int("request_timeout", 600)

FRIENDLY_MESSAGES: Dict[str, str] = {
    "INVALID_FILE_TYPE": "Unsupported file format. Please upload a PDF or MP3 file.",
    "INVALID_PDF_STRUCTURE": "This PDF does not contain a text layer and cannot be processed.",
    "FILE_TOO_LARGE": "The file exceeds the 200 MB size limit.",
    "AUDIO_TOO_LONG": "Audio exceeds the 60-minute limit.",
    "MISSING_QUERY": "Please enter a question before sending.",
    "UNAUTHORIZED_DEBUG": "Debug features are available only in development mode.",
    "FILE_NOT_FOUND": "The selected file is no longer available. Please re-upload it.",
    "TIMEOUT_STAGE": "Processing took too long. Please try again in a moment.",
    "EMBEDDING_ERROR": "We were unable to embed the content. Please retry shortly.",
    "TRANSCRIPTION_ERROR": "The audio transcription failed. Please check the recording and retry.",
    "RATE_LIMIT_EXCEEDED": "The service is temporarily busy. Please try again shortly.",
    "INTERNAL_ERROR": "Something unexpected happened. Please retry or contact support.",
    "LLM_PROVIDER_DOWN": "The language model provider is unavailable right now. Please try again soon.",
    "VECTOR_DB_UNAVAILABLE": "The search index is unavailable. Please retry shortly.",
    "GENERATION_TIMEOUT": "Generating a response took too long. Please try again.",
    "INVALID_REQUEST": "The request payload is invalid. Please review your input and try again.",
    "HTTP_ERROR": "We were unable to complete that request due to an HTTP error.",
    "RESOURCE_NOT_FOUND": "The resource you requested could not be found.",
}

GENERIC_ERROR_MESSAGE = "We ran into a problem while completing that action. Please try again."


class ProgressUpdater(Protocol):
    def __call__(self, label: str, state: str = "running") -> None:  # pragma: no cover - protocol definition
        ...


@contextmanager
def _progress_status(initial_label: str) -> Iterator[ProgressUpdater]:
    """Render a progress indicator with graceful fallback."""

    status_factory = getattr(st, "status", None)
    if status_factory:
        with status_factory(initial_label, expanded=True) as status:
            def _update(label: str, state: str = "running") -> None:
                status.update(label=label, state=state)

            yield _update
    else:  # pragma: no cover - requires older Streamlit versions
        placeholder = st.empty()
        placeholder.info(initial_label)

        def _update(label: str, state: str = "running") -> None:
            if state == "complete":
                placeholder.success(label)
            elif state == "error":
                placeholder.error(label)
            else:
                placeholder.info(label)

        try:
            yield _update
        finally:
            pass


def _post_multipart(endpoint: str, files: dict[str, tuple[str, bytes, str]]) -> dict[str, Any]:
    url = f"{API_BASE}{endpoint}"
    response = requests.post(url, files=files, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _post_json(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{API_BASE}{endpoint}"
    response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _parse_error_response(response: requests.Response) -> tuple[str, str | None, Dict[str, Any] | None]:
    """Translate backend error payload into user-facing content."""

    try:
        payload = response.json()
    except ValueError:  # pragma: no cover - defensive branch
        payload = {}
    error_code = payload.get("error_code")
    hint = payload.get("hint")
    message = FRIENDLY_MESSAGES.get(error_code) or payload.get("message") or GENERIC_ERROR_MESSAGE

    debug_payload: Dict[str, Any] | None = None
    if DEBUG_MODE:
        debug_payload = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "payload": payload,
        }
    return message, hint, debug_payload


def _display_error(message: str, *, hint: str | None = None, debug: Dict[str, Any] | None = None) -> None:
    """Render a friendly error message with optional debug context."""

    st.error(message)
    if hint:
        st.caption(hint)
    if DEBUG_MODE and debug:
        with st.expander("Debug details"):
            st.json(debug)


def _handle_request_error(error: Exception) -> None:
    """Normalize request exceptions into consistent UI feedback."""

    if isinstance(error, requests.HTTPError) and error.response is not None:
        message, hint, debug_payload = _parse_error_response(error.response)
        _display_error(message, hint=hint, debug=debug_payload)
        return

    fallback_message = "We couldn't reach the service. Please check your connection and try again."
    debug_payload: Dict[str, Any] | None = None
    if DEBUG_MODE:
        debug_payload = {"exception": repr(error)}
    _display_error(fallback_message, debug=debug_payload)


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:  # pragma: no cover - defensive
        return None


def _format_expiration(expires_at: str | None) -> str | None:
    parsed = _parse_timestamp(expires_at)
    if not parsed:
        return None
    local_dt = parsed.astimezone()
    return local_dt.strftime("%b %d, %Y %H:%M %Z")


def _register_ingested_file(record: dict[str, Any]) -> None:
    files = st.session_state.setdefault("ingested_files", [])
    files.append(record)


def _render_ingestion_success(payload: dict[str, Any]) -> None:
    filename = payload.get("filename", "file")
    kind = payload.get("kind", "pdf")
    expires_text = _format_expiration(payload.get("expires_at"))

    details: list[str] = []
    if kind == "pdf" and payload.get("page_count") is not None:
        details.append(f"{payload['page_count']} pages")
    if kind == "audio" and payload.get("duration_seconds") is not None:
        duration = int(round(payload["duration_seconds"]))
        details.append(f"{duration} seconds")
    if expires_text:
        details.append(f"retained until {expires_text}")

    summary = ", ".join(details)
    if summary:
        st.success(f"Processed **{filename}** ({summary}).")
    else:
        st.success(f"Processed **{filename}**.")

    _register_ingested_file(
        {
            "filename": filename,
            "kind": kind,
            "details": summary,
            "file_id": payload.get("file_id"),
            "expires_at": payload.get("expires_at"),
        }
    )


def _render_recent_ingestions() -> None:
    files: list[dict[str, Any]] = st.session_state.get("ingested_files", [])
    if not files:
        return

    st.subheader("Recently processed files")
    for item in reversed(files[-5:]):
        badge = "PDF" if item.get("kind") == "pdf" else "Audio"
        filename = item.get("filename", "Unknown file")
        details = item.get("details") or "Ready for questions."
        st.markdown(f"- **{filename}** · {badge} · {details}")


def _ensure_chat_session() -> None:
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = {
            "session_id": None,
            "conversation": [],
        }


def _reset_chat_session() -> None:
    st.session_state.chat_session = {
        "session_id": None,
        "conversation": [],
    }


def _render_chat_history() -> None:
    _ensure_chat_session()
    for message in st.session_state.chat_session["conversation"]:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant" and message.get("latency_ms") is not None:
                st.caption(f"Responded in {message['latency_ms']} ms")


def _handle_ingestion(file, endpoint: str) -> None:
    buffer = file.getvalue()
    mime_type = file.type or "application/octet-stream"
    with _progress_status("Preparing upload...") as update_status:
        update_status("Uploading file to the server...")
        try:
            response = _post_multipart(endpoint, {"file": (file.name, buffer, mime_type)})
        except requests.RequestException as exc:  # pragma: no cover - UI only
            update_status("Upload failed.", state="error")
            _handle_request_error(exc)
            return

        update_status("Processing file with the knowledge pipeline...")

        update_status("File is ready for chat.", state="complete")

    _render_ingestion_success(response)


def _render_ingest_tab() -> None:
    st.subheader("Add source material")
    st.write("Upload PDF presentations or MP3 recordings to make them searchable in chat.")

    with st.form("pdf_ingest_form"):
        pdf_file = st.file_uploader("PDF document", type=["pdf"], key="pdf_uploader")
        submitted = st.form_submit_button("Process PDF", use_container_width=True)
        if submitted:
            if pdf_file is None:
                st.warning("Please choose a PDF file before processing.")
            else:
                _handle_ingestion(pdf_file, "/upload/pdf")

    with st.form("audio_ingest_form"):
        audio_file = st.file_uploader("Audio recording (MP3)", type=["mp3"], key="audio_uploader")
        submitted = st.form_submit_button("Process Audio", use_container_width=True)
        if submitted:
            if audio_file is None:
                st.warning("Please choose an MP3 file before processing.")
            else:
                _handle_ingestion(audio_file, "/upload/audio")

    _render_recent_ingestions()


def _render_chat_tab() -> None:
    _ensure_chat_session()
    st.subheader("Chat with your knowledge base")
    st.write("Ask questions about the files you've ingested during this session.")

    action_col = st.columns([3, 1])[1]
    with action_col:
        if st.button("Start new chat", use_container_width=True):
            _reset_chat_session()
            toast = getattr(st, "toast", None)
            if callable(toast):
                toast("Started a fresh chat session.")
            st.experimental_rerun()

    _render_chat_history()

    if prompt := st.chat_input("Ask a question about your uploaded content"):
        st.session_state.chat_session["conversation"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Thinking...")
            try:
                response = _post_json(
                    "/chat",
                    {
                        "query": prompt,
                        **(
                            {"session_id": st.session_state.chat_session["session_id"]}
                            if st.session_state.chat_session.get("session_id")
                            else {}
                        ),
                    },
                )
            except requests.RequestException as exc:  # pragma: no cover - UI only
                placeholder.empty()
                _handle_request_error(exc)
                return

            answer = response.get("answer", "") or "No answer returned."
            latency_ms = response.get("latency_ms")
            st.session_state.chat_session["session_id"] = response.get("session_id")
            placeholder.markdown(answer)
            if latency_ms is not None:
                st.caption(f"Responded in {latency_ms} ms")
            st.session_state.chat_session["conversation"].append(
                {"role": "assistant", "content": answer, "latency_ms": latency_ms}
            )


st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot")
st.info(
    f"Uploads and chat history are retained for up to {RETENTION_HOURS} hours for this session before being purged."
)

ingest_tab, chat_tab = st.tabs(["Ingest", "Chat"])

with ingest_tab:
    _render_ingest_tab()

with chat_tab:
    _render_chat_tab()
