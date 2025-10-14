"""Streamlit application for the RAG chatbot."""
from __future__ import annotations

from typing import Any

import requests
import streamlit as st

API_BASE = st.secrets.get("api_base", "http://localhost:8000")


def _post_multipart(endpoint: str, files: dict[str, tuple[str, bytes, str]]) -> dict[str, Any]:
    url = f"{API_BASE}{endpoint}"
    response = requests.post(url, files=files, timeout=30)
    response.raise_for_status()
    return response.json()


def _post_json(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{API_BASE}{endpoint}"
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot")
st.caption("Uploads are stored for up to 24 hours for session continuity.")

ingest_tab, chat_tab = st.tabs(["Ingest", "Chat"])

with ingest_tab:
    st.header("Ingest")
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf")
    if pdf_file and st.button("Process PDF"):
        try:
            response = _post_multipart("/upload/pdf", {"file": (pdf_file.name, pdf_file.getvalue(), pdf_file.type)})
            st.success(f"Processed {response['filename']} ({response.get('page_count', 0)} pages)")
        except requests.HTTPError as exc:  # pragma: no cover - UI only
            payload = exc.response.json()
            st.error(payload.get("message", "Upload failed."))

    audio_file = st.file_uploader("Upload Audio", type=["mp3"], key="audio")
    if audio_file and st.button("Process Audio"):
        try:
            response = _post_multipart("/upload/audio", {"file": (audio_file.name, audio_file.getvalue(), audio_file.type)})
            st.success(f"Transcribed {response['filename']} ({int(response.get('duration_seconds', 0))} s)")
        except requests.HTTPError as exc:  # pragma: no cover
            payload = exc.response.json()
            st.error(payload.get("message", "Upload failed."))

with chat_tab:
    st.header("Chat")
    query = st.text_input("Ask a question")
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if st.button("Send") and query:
        try:
            response = _post_json("/chat", {"query": query})
            st.session_state.conversation.append((query, response["answer"]))
        except requests.HTTPError as exc:  # pragma: no cover
            payload = exc.response.json()
            st.error(payload.get("message", "Chat failed."))
    for prompt, answer in reversed(st.session_state.conversation):
        st.markdown(f"**You:** {prompt}")
        st.markdown(f"**Bot:** {answer}")
