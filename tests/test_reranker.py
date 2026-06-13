"""Tests for Reranker — mock CrossEncoder to avoid loading the model."""
import numpy as np
from unittest.mock import MagicMock, patch


@patch("src.reranker.CrossEncoder")
def test_rerank_sorts_by_score(mock_ce_cls):
    from src.reranker import Reranker

    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.array([0.3, 0.9, 0.1])
    mock_ce_cls.return_value = mock_ce

    reranker = Reranker.__new__(Reranker)
    reranker.model = mock_ce

    candidates = [
        {"text": "low relevance",    "file": "a.md", "chunk": 0, "score": 0.5},
        {"text": "high relevance",   "file": "b.md", "chunk": 0, "score": 0.4},
        {"text": "lowest relevance", "file": "c.md", "chunk": 0, "score": 0.3},
    ]
    results = reranker.rerank("test query", candidates, top_k=3)

    assert results[0]["file"] == "b.md"
    assert results[0]["rerank_score"] == 0.9
    assert results[1]["file"] == "a.md"
    assert results[2]["file"] == "c.md"


@patch("src.reranker.CrossEncoder")
def test_rerank_returns_top_k(mock_ce_cls):
    from src.reranker import Reranker

    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.array([float(i) for i in range(10)])
    mock_ce_cls.return_value = mock_ce

    reranker = Reranker.__new__(Reranker)
    reranker.model = mock_ce

    candidates = [
        {"text": f"doc {i}", "file": f"{i}.md", "chunk": 0, "score": 0.5}
        for i in range(10)
    ]
    results = reranker.rerank("query", candidates, top_k=3)

    assert len(results) == 3
    assert all("rerank_score" in r for r in results)


@patch("src.reranker.CrossEncoder")
def test_rerank_empty_candidates_returns_empty(mock_ce_cls):
    from src.reranker import Reranker

    reranker = Reranker.__new__(Reranker)
    reranker.model = mock_ce_cls.return_value

    results = reranker.rerank("any query", [], top_k=5)
    assert results == []
    mock_ce_cls.return_value.predict.assert_not_called()
