"""Tests for Retriever — mock Qdrant, embedder, and reranker."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ------------------------------------------------------------------
# Existing tests (v1 path: pure vector, no hybrid, no rerank)
# ------------------------------------------------------------------

@patch("src.retriever.QdrantClient")
@patch("src.retriever.SentenceTransformer")
def test_search_returns_formatted_results(mock_transformer, mock_qdrant_cls):
    from src.retriever import Retriever

    mock_hit = MagicMock()
    mock_hit.score = 0.92
    mock_hit.payload = {"file": "src/main.py", "chunk_idx": 0, "text": "def run(): pass"}

    mock_qdrant = MagicMock()
    mock_qdrant.search.return_value = [mock_hit]
    mock_qdrant_cls.return_value = mock_qdrant

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.zeros(384)
    mock_transformer.return_value = mock_embedder

    retriever = Retriever.__new__(Retriever)
    retriever.client       = mock_qdrant
    retriever.embedder     = mock_embedder
    retriever.reranker     = None
    retriever._bm25        = None
    retriever._bm25_corpus = []

    results = retriever.search("how does run work?", top_k=1, hybrid=False, rerank=False)

    assert len(results) == 1
    assert results[0]["file"]  == "src/main.py"
    assert results[0]["score"] == 0.92
    assert "def run" in results[0]["text"]


@patch("src.retriever.QdrantClient")
@patch("src.retriever.SentenceTransformer")
def test_search_empty_returns_empty_list(mock_transformer, mock_qdrant_cls):
    from src.retriever import Retriever

    mock_qdrant = MagicMock()
    mock_qdrant.search.return_value = []
    mock_qdrant_cls.return_value = mock_qdrant

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.zeros(384)
    mock_transformer.return_value = mock_embedder

    retriever = Retriever.__new__(Retriever)
    retriever.client       = mock_qdrant
    retriever.embedder     = mock_embedder
    retriever.reranker     = None
    retriever._bm25        = None
    retriever._bm25_corpus = []

    results = retriever.search("nonexistent topic", hybrid=False, rerank=False)
    assert results == []


@patch("src.retriever.QdrantClient")
@patch("src.retriever.SentenceTransformer")
def test_format_context_includes_citations(mock_transformer, mock_qdrant_cls):
    from src.retriever import Retriever

    retriever = Retriever.__new__(Retriever)
    retriever.client       = mock_qdrant_cls.return_value
    retriever.embedder     = mock_transformer.return_value
    retriever.reranker     = None
    retriever._bm25        = None
    retriever._bm25_corpus = []

    chunks = [
        {"file": "README.md",    "chunk": 0, "score": 0.9, "text": "This project does X."},
        {"file": "src/core.py",  "chunk": 1, "score": 0.8, "text": "Core logic here."},
    ]
    context = retriever.format_context(chunks)

    assert "[1]" in context
    assert "[2]" in context
    assert "README.md"   in context
    assert "src/core.py" in context


# ------------------------------------------------------------------
# v2 tests: RRF, hybrid search, reranking
# ------------------------------------------------------------------

def test_rrf_merges_two_ranked_lists():
    from src.retriever import Retriever

    vec_hits = [
        {"file": "a.py", "chunk": 0, "score": 0.9, "text": "vec only"},
        {"file": "b.py", "chunk": 0, "score": 0.8, "text": "shared doc"},
    ]
    bm25_hits = [
        {"file": "b.py", "chunk": 0, "score": 5.0, "text": "shared doc"},
        {"file": "c.py", "chunk": 0, "score": 4.0, "text": "bm25 only"},
    ]
    fused = Retriever._rrf(vec_hits, bm25_hits)
    files = [r["file"] for r in fused]

    assert "a.py" in files
    assert "b.py" in files
    assert "c.py" in files
    # b.py appears in both lists → highest RRF score
    assert fused[0]["file"] == "b.py"


@patch("src.retriever.QdrantClient")
@patch("src.retriever.SentenceTransformer")
def test_hybrid_search_builds_and_uses_bm25(mock_transformer, mock_qdrant_cls):
    from src.retriever import Retriever

    mock_hit = MagicMock()
    mock_hit.score = 0.85
    mock_hit.payload = {"file": "doc.md", "chunk_idx": 0, "text": "vector match keyword"}

    mock_qdrant = MagicMock()
    mock_qdrant.search.return_value = [mock_hit]
    corpus_record = MagicMock()
    corpus_record.payload = {"text": "vector match keyword", "file": "doc.md", "chunk_idx": 0}
    mock_qdrant.scroll.return_value = ([corpus_record], None)
    mock_qdrant_cls.return_value = mock_qdrant

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.zeros(384)
    mock_transformer.return_value = mock_embedder

    retriever = Retriever.__new__(Retriever)
    retriever.client       = mock_qdrant
    retriever.embedder     = mock_embedder
    retriever.reranker     = None
    retriever._bm25        = None
    retriever._bm25_corpus = []

    results = retriever.search("keyword", top_k=1, hybrid=True, rerank=False)

    mock_qdrant.scroll.assert_called_once()
    assert len(results) >= 1


@patch("src.retriever.QdrantClient")
@patch("src.retriever.SentenceTransformer")
def test_search_calls_reranker_and_attaches_rerank_score(mock_transformer, mock_qdrant_cls):
    from src.retriever import Retriever

    mock_hit = MagicMock()
    mock_hit.score = 0.85
    mock_hit.payload = {"file": "doc.md", "chunk_idx": 0, "text": "reranked result"}

    mock_qdrant = MagicMock()
    mock_qdrant.search.return_value = [mock_hit]
    mock_qdrant_cls.return_value = mock_qdrant

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.zeros(384)
    mock_transformer.return_value = mock_embedder

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        {"file": "doc.md", "chunk": 0, "score": 0.85, "text": "reranked result", "rerank_score": 9.5}
    ]

    retriever = Retriever.__new__(Retriever)
    retriever.client       = mock_qdrant
    retriever.embedder     = mock_embedder
    retriever.reranker     = mock_reranker
    retriever._bm25        = None
    retriever._bm25_corpus = []

    results = retriever.search("query", top_k=1, hybrid=False, rerank=True)

    mock_reranker.rerank.assert_called_once()
    assert results[0]["rerank_score"] == 9.5
