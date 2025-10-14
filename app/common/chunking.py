"""Utilities for splitting text into overlapping chunks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class ChunkConfig:
    chunk_size: int
    chunk_overlap: int


def _tokenize(text: str) -> List[str]:
    return text.split()


def chunk_text(text: str, config: ChunkConfig) -> Iterable[str]:
    tokens = _tokenize(text)
    if not tokens:
        return []
    step = max(1, config.chunk_size - config.chunk_overlap)
    for start in range(0, len(tokens), step):
        slice_tokens = tokens[start : start + config.chunk_size]
        if not slice_tokens:
            continue
        yield " ".join(slice_tokens)


__all__ = ["ChunkConfig", "chunk_text"]
