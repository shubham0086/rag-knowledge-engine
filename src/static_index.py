"""
Static index builder (serverless mode).

Builds a portable `index.json` (chunks + embeddings + metadata) from a directory of
documents, using any embedding backend. No Qdrant, no running server: the output is a
single JSON file that ships with a static site and is queried by a lightweight runtime
(e.g. a Vercel function doing cosine similarity). This is what makes the RAG engine
edge-deployable.

The Qdrant-backed Ingestor/Retriever path is unchanged; this is an additional export.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from .chunking import _chunk_text
from .embeddings import get_embedder

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_html(text: str) -> str:
    """Minimal HTML -> text (drops script/style and tags). Avoids a bs4 dependency."""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.I | re.S)
    text = _TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def build_static_index(
    directory: str,
    out_path: str = "index.json",
    embedder=None,
    embedder_name: str = "gemini",
    extensions: Optional[List[str]] = None,
    source_base_url: str = "",
    exclude_dirs: Optional[set] = None,
) -> Dict:
    """
    Chunk + embed every matching file under `directory`, write a static index.json.

    Returns a small summary dict: { count, dim, model, out_path }.
    `source_base_url` is stored in the index so a runtime can turn `file` into a link.
    `exclude_dirs` skips any file whose relative path contains one of those dir names
    (e.g. embedded repos, asset folders, build noise).
    """
    extensions = extensions or [".txt", ".md", ".html"]
    embedder = embedder or get_embedder(embedder_name)
    exclude_dirs = exclude_dirs or set()

    records: List[Dict] = []
    for path in sorted(Path(directory).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if exclude_dirs and (set(path.relative_to(directory).parts[:-1]) & exclude_dirs):
            continue
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore").lstrip("﻿")
        except Exception:
            continue
        text = _strip_html(raw) if path.suffix.lower() in (".html", ".htm") else raw
        rel = str(path.relative_to(directory)).replace("\\", "/")
        for idx, chunk in enumerate(_chunk_text(text)):
            records.append({"file": rel, "chunk_idx": idx, "text": chunk})

    vectors = embedder.encode([r["text"] for r in records]) if records else []
    for rec, vec in zip(records, vectors):
        rec["vector"] = vec

    index = {
        "model": getattr(embedder, "MODEL", embedder.__class__.__name__),
        "dim": getattr(embedder, "dim", None),
        "source_base_url": source_base_url,
        "count": len(records),
        "records": records,
    }
    Path(out_path).write_text(json.dumps(index), encoding="utf-8")
    return {"count": len(records), "dim": index["dim"], "model": index["model"], "out_path": out_path}
