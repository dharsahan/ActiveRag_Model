"""Tests for the confidence checker module."""

import json
from unittest.mock import MagicMock, patch

from active_rag.confidence_checker import ConfidenceChecker
from active_rag.config import Config


def _mock_openai_response(content: str) -> MagicMock:
    """Build a mock OpenAI chat completion response."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


@patch("active_rag.confidence_checker.OpenAI")
def test_high_confidence(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_openai_response(
        json.dumps({"confidence": 0.95, "reasoning": "Common knowledge"})
    )

    config = Config(confidence_threshold=0.7)
    checker = ConfidenceChecker(config)
    result = checker.check("What is 2+2?")

    assert result.confidence == 0.95
    assert result.is_high_confidence is True


@patch("active_rag.confidence_checker.OpenAI")
def test_low_confidence(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_openai_response(
        json.dumps({"confidence": 0.2, "reasoning": "Obscure topic"})
    )

    config = Config(confidence_threshold=0.7)
    checker = ConfidenceChecker(config)
    result = checker.check("What was the weather in Tokyo on Jan 1 1823?")

    assert result.confidence == 0.2
    assert result.is_high_confidence is False


@patch("active_rag.confidence_checker.OpenAI")
def test_malformed_json_defaults_to_low(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_openai_response(
        "not json at all"
    )

    config = Config(confidence_threshold=0.7)
    checker = ConfidenceChecker(config)
    result = checker.check("anything")

    assert result.confidence == 0.0
    assert result.is_high_confidence is False
