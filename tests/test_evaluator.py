"""Tests for the answer quality evaluator."""

import json
from unittest.mock import MagicMock, patch

from active_rag.config import Config
from active_rag.evaluator import AnswerEvaluator


def _mock_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@patch("active_rag.evaluator.OpenAI")
def test_evaluate_good_answer(mock_openai):
    """A good answer scores high quality."""
    client = MagicMock()
    mock_openai.return_value = client
    client.chat.completions.create.return_value = _mock_response(
        json.dumps({"quality": 0.9, "issues": [], "suggestion": ""})
    )
    evaluator = AnswerEvaluator(Config())
    result = evaluator.evaluate("What is Python?", "Python is a programming language.")
    assert result.quality >= 0.8
    assert result.is_acceptable is True


@patch("active_rag.evaluator.OpenAI")
def test_evaluate_bad_answer(mock_openai):
    """A bad answer scores low quality."""
    client = MagicMock()
    mock_openai.return_value = client
    client.chat.completions.create.return_value = _mock_response(
        json.dumps({"quality": 0.2, "issues": ["off-topic"], "suggestion": "Address the question"})
    )
    evaluator = AnswerEvaluator(Config())
    result = evaluator.evaluate("What is Python?", "The sky is blue.")
    assert result.quality < 0.5
    assert result.is_acceptable is False


@patch("active_rag.evaluator.OpenAI")
def test_evaluate_malformed_json_defaults_to_acceptable(mock_openai):
    """If the LLM returns invalid JSON, the evaluation defaults to neutral/acceptable."""
    client = MagicMock()
    mock_openai.return_value = client
    client.chat.completions.create.return_value = _mock_response(
        "I couldn't evaluate this, sorry."
    )
    evaluator = AnswerEvaluator(Config())
    result = evaluator.evaluate("Q", "A")
    assert result.quality == 0.5
    assert result.is_acceptable is True
    assert "Failed to parse" in result.issues[0]
