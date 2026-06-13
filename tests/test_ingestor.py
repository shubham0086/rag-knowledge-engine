"""
Tests for Ingestor — uses a mock Qdrant client to avoid needing a live server.
"""
import pytest
from unittest.mock import MagicMock, patch
from src.ingestor import Ingestor, _chunk_text, _doc_id


def test_chunk_text_basic():
    text  = " ".join([f"word{i}" for i in range(100)])
    chunks = _chunk_text(text, size=20, overlap=5)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 20 for c in chunks)


def test_chunk_text_single_chunk():
    text   = "short text"
    chunks = _chunk_text(text, size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == "short text"


def test_doc_id_deterministic():
    id1 = _doc_id("path/to/file.py", 0)
    id2 = _doc_id("path/to/file.py", 0)
    assert id1 == id2


def test_doc_id_different_inputs():
    id1 = _doc_id("file.py", 0)
    id2 = _doc_id("file.py", 1)
    assert id1 != id2


@patch("src.ingestor.QdrantClient")
@patch("src.ingestor.SentenceTransformer")
def test_ingest_file_returns_chunk_count(mock_transformer, mock_qdrant_cls, tmp_path):
    mock_qdrant    = MagicMock()
    mock_qdrant.get_collections.return_value.collections = []
    mock_qdrant_cls.return_value = mock_qdrant

    mock_embedder  = MagicMock()
    import numpy as np
    mock_embedder.encode.return_value = np.zeros((3, 384))
    mock_transformer.return_value = mock_embedder

    test_file = tmp_path / "test.txt"
    test_file.write_text(" ".join([f"token{i}" for i in range(600)]))

    ingestor = Ingestor.__new__(Ingestor)
    ingestor.client   = mock_qdrant
    ingestor.embedder = mock_embedder

    count = ingestor.ingest_file(str(test_file))
    assert count > 0
    mock_qdrant.upsert.assert_called_once()


@patch("src.ingestor.QdrantClient")
@patch("src.ingestor.SentenceTransformer")
def test_ingest_empty_file_returns_zero(mock_transformer, mock_qdrant_cls, tmp_path):
    mock_qdrant = MagicMock()
    mock_qdrant.get_collections.return_value.collections = []
    mock_qdrant_cls.return_value = mock_qdrant
    mock_transformer.return_value = MagicMock()

    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")

    ingestor = Ingestor.__new__(Ingestor)
    ingestor.client   = mock_qdrant
    ingestor.embedder = mock_transformer.return_value

    count = ingestor.ingest_file(str(empty_file))
    assert count == 0
