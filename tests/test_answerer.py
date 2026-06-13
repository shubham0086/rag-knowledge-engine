"""
Tests for Answerer — mock Anthropic client.
"""
import pytest
from unittest.mock import MagicMock, patch


@patch("src.answerer.anthropic.Anthropic")
def test_answer_with_context(mock_anthropic_cls):
    from src.answerer import Answerer

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="The answer is 42. [1]")]
    mock_response.usage.input_tokens  = 100
    mock_response.usage.output_tokens = 20

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_cls.return_value = mock_client

    answerer = Answerer()
    chunks = [{"file": "doc.md", "chunk": 0, "score": 0.9, "text": "The answer is 42."}]
    result = answerer.answer("What is the answer?", chunks)

    assert "42" in result["answer"]
    assert result["sources"] == ["doc.md"]
    assert result["tokens"]["input"]  == 100
    assert result["tokens"]["output"] == 20


@patch("src.answerer.anthropic.Anthropic")
def test_empty_chunks_returns_not_found(mock_anthropic_cls):
    from src.answerer import Answerer

    answerer = Answerer()
    result = answerer.answer("any question", [])

    assert "No relevant documents" in result["answer"]
    mock_anthropic_cls.return_value.messages.create.assert_not_called()
