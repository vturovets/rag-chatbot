"""Utilities for splitting text into normalized, token-aware chunks."""
from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
import re
from typing import List, Sequence


@dataclass(frozen=True)
class ChunkConfig:
    """Configuration governing token sized chunks and overlap."""

    chunk_size: int
    chunk_overlap: int

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")


@dataclass(frozen=True)
class NormalizedChunk:
    """A normalized representation of chunked text."""

    text: str
    token_count: int
    start_token: int
    end_token: int


class _Tokenizer:
    """Wrapper that prefers a tiktoken tokenizer with whitespace fallback."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding = None
        if importlib.util.find_spec("tiktoken") is not None:
            tiktoken = importlib.import_module("tiktoken")
            get_encoding = getattr(tiktoken, "get_encoding", None)
            if callable(get_encoding):
                try:
                    self._encoding = get_encoding(encoding_name)
                except Exception:
                    self._encoding = None

    def encode(self, text: str) -> Sequence[int | str]:
        if self._encoding is not None:
            return self._encoding.encode(text)
        if not text:
            return []
        return text.split()

    def decode(self, tokens: Sequence[int | str]) -> str:
        if self._encoding is not None:
            return self._encoding.decode(tokens)  # type: ignore[arg-type]
        return " ".join(str(token) for token in tokens)


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text)


def chunk_text(text: str, config: ChunkConfig, *, tokenizer: _Tokenizer | None = None) -> List[NormalizedChunk]:
    """Split *text* into normalized chunks adhering to the config constraints."""

    normalized_source = _normalize_text(text)
    if not normalized_source:
        return []

    tokenizer = tokenizer or _Tokenizer()
    tokens = list(tokenizer.encode(normalized_source))
    if not tokens:
        return []

    step = config.chunk_size - config.chunk_overlap
    chunks: List[NormalizedChunk] = []
    for start in range(0, len(tokens), step):
        end = min(start + config.chunk_size, len(tokens))
        if end <= start:
            break
        chunk_tokens = tokens[start:end]
        decoded = _normalize_text(tokenizer.decode(chunk_tokens))
        if not decoded:
            continue
        chunks.append(
            NormalizedChunk(
                text=decoded,
                token_count=len(chunk_tokens),
                start_token=start,
                end_token=end,
            )
        )
        if end >= len(tokens):
            break
    return chunks


__all__ = ["ChunkConfig", "NormalizedChunk", "chunk_text"]
