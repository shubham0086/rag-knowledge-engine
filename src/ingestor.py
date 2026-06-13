"""
Ingestor: reads files from a directory, chunks them, embeds them,
and upserts into Qdrant.
"""
import os
import hashlib
from pathlib import Path
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

COLLECTION = "knowledge"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE  = 512
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


class Ingestor:
    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self.client  = QdrantClient(url=qdrant_url)
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION not in existing:
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    def ingest_directory(self, directory: str, extensions: List[str] = None) -> int:
        extensions = extensions or [".txt", ".md", ".py", ".js", ".ts", ".json"]
        total = 0
        for path in Path(directory).rglob("*"):
            if path.suffix in extensions and path.is_file():
                total += self.ingest_file(str(path))
        return total

    def ingest_file(self, file_path: str) -> int:
        try:
            text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return 0

        chunks  = _chunk_text(text)
        vectors = self.embedder.encode(chunks).tolist()
        points  = []

        for idx, (chunk, vector) in enumerate(zip(chunks, vectors)):
            points.append(PointStruct(
                id=_doc_id(file_path, idx),
                vector=vector,
                payload={
                    "file": file_path,
                    "chunk_idx": idx,
                    "text": chunk,
                }
            ))

        if points:
            self.client.upsert(collection_name=COLLECTION, points=points)
        return len(points)
