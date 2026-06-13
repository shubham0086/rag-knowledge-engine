"""Tests for Evaluator — mock Anthropic and SentenceTransformer."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


@patch("src.evaluator.anthropic.Anthropic")
@patch("src.evaluator.SentenceTransformer")
def test_faithfulness_returns_llm_score(mock_st, mock_anthropic_cls):
    from src.evaluator import Evaluator

    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(
        text='{"total_claims":2,"supported_claims":2,"faithfulness_score":1.0,"unsupported":[]}'
    )]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_resp
    mock_anthropic_cls.return_value = mock_client

    evaluator = Evaluator.__new__(Evaluator)
    evaluator.client  = mock_client
    evaluator.model   = "claude-haiku-4-5-20251001"
    evaluator.embedder = mock_st.return_value

    chunks = [{"text": "The sky is blue.", "file": "doc.md", "chunk": 0, "score": 0.9}]
    score = evaluator._faithfulness("The sky is blue. [1]", chunks)

    assert score == 1.0
    mock_client.messages.create.assert_called_once()


@patch("src.evaluator.SentenceTransformer")
def test_answer_relevance_identical_strings_is_near_one(mock_st):
    from src.evaluator import Evaluator

    vec = np.random.rand(384).astype(np.float32)
    vec /= np.linalg.norm(vec)

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.stack([vec, vec])

    evaluator = Evaluator.__new__(Evaluator)
    evaluator.embedder = mock_embedder
    evaluator.client   = MagicMock()
    evaluator.model    = "claude-haiku-4-5-20251001"

    score = evaluator._answer_relevance("same question", "same question")
    assert score > 0.99


@patch("src.evaluator.SentenceTransformer")
def test_context_precision_prefers_rerank_score(mock_st):
    from src.evaluator import Evaluator

    evaluator = Evaluator.__new__(Evaluator)
    evaluator.embedder = mock_st.return_value
    evaluator.client   = MagicMock()
    evaluator.model    = "claude-haiku-4-5-20251001"

    chunks = [
        {"text": "a", "file": "f.md", "chunk": 0, "score": 0.5, "rerank_score": 0.8},
        {"text": "b", "file": "g.md", "chunk": 0, "score": 0.4, "rerank_score": 0.6},
    ]
    precision = evaluator._context_precision(chunks)
    # should use rerank_score (0.8 + 0.6) / 2 = 0.7, not raw score
    assert precision == pytest.approx(0.7, abs=0.01)
