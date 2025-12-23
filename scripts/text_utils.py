"""Shared text utilities for preprocessing and RAG."""
from __future__ import annotations

import re
from typing import Iterable, List


def clean_text(text: str) -> str:
    """Normalize whitespace and remove control characters."""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks to improve retrieval recall.

    Args:
        text: input string
        chunk_size: target length of each chunk
        overlap: number of characters to overlap between chunks
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0:
        raise ValueError("overlap must be non-negative.")

    words = text.split()
    chunks: List[str] = []
    current: List[str] = []

    for word in words:
        current.append(word)
        if len(" ".join(current)) >= chunk_size:
            chunks.append(" ".join(current))
            # start next chunk with overlap
            overlap_words = []
            while current and len(" ".join(overlap_words)) < overlap:
                overlap_words.insert(0, current.pop())  # pop from end to keep order
            current = overlap_words

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


def flatten(list_of_lists: Iterable[Iterable[str]]) -> List[str]:
    """Flatten an iterable of iterables into a list."""
    out: List[str] = []
    for sub in list_of_lists:
        out.extend(sub)
    return out


