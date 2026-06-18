"""
Chunking primitives, kept dependency-light (no torch / no qdrant).

Extracted from ingestor.py so the serverless static-index builder can chunk text
without importing the heavy embedding/vector-store stack. ingestor.py re-imports
these names, so existing behavior and imports are unchanged.
"""
import hashlib
from typing import List

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + size])
        chunks.append(chunk)
        i += size - overlap
    return chunks


def _doc_id(file_path: str, chunk_idx: int) -> str:
    raw = f"{file_path}::{chunk_idx}"
    return hashlib.md5(raw.encode()).hexdigest()
