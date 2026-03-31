"""Tests for the full Active RAG pipeline."""

import json
from unittest.mock import MagicMock, patch

from openai import APIConnectionError
from tenacity import RetryError

from active_rag.config import Config
from active_rag.pipeline import ActiveRAGPipeline


def _mock_openai_response(content: str) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


@patch("active_rag.answer_generator.OpenAI")
@patch("active_rag.confidence_checker.OpenAI")
def test_high_confidence_path(mock_conf_openai, mock_ans_openai):
    """High-confidence queries bypass RAG entirely."""
    mock_conf_client = MagicMock()
    mock_conf_openai.return_value = mock_conf_client
    mock_conf_client.chat.completions.create.return_value = (
        _mock_openai_response(
            json.dumps({"confidence": 0.95, "reasoning": "Well-known fact"})
        )
    )

    mock_ans_client = MagicMock()
    mock_ans_openai.return_value = mock_ans_client
    mock_ans_client.chat.completions.create.return_value = (
        _mock_openai_response("Direct answer here.")
    )

    config = Config(confidence_threshold=0.7)
    pipeline = ActiveRAGPipeline(config)
    result = pipeline.run("What is 2+2?")

    assert result.path == "direct"
    assert result.answer.source == "direct"
    assert result.answer.text == "Direct answer here."


@patch("active_rag.web_search.DDGS")
@patch("active_rag.answer_generator.OpenAI")
@patch("active_rag.confidence_checker.OpenAI")
def test_low_confidence_empty_store_triggers_web_search(
    mock_conf_openai, mock_ans_openai, mock_ddgs_cls
):
    """Low confidence + empty vector store → web search path."""
    mock_conf_client = MagicMock()
    mock_conf_openai.return_value = mock_conf_client
    mock_conf_client.chat.completions.create.return_value = (
        _mock_openai_response(
            json.dumps({"confidence": 0.1, "reasoning": "No idea"})
        )
    )

    mock_ans_client = MagicMock()
    mock_ans_openai.return_value = mock_ans_client
    mock_ans_client.chat.completions.create.return_value = (
        _mock_openai_response("Answer from web context.")
    )

    # Mock web search to return no results (to avoid real HTTP calls)
    mock_ddgs_cls.return_value.text.return_value = []

    config = Config(confidence_threshold=0.7)
    pipeline = ActiveRAGPipeline(config)
    result = pipeline.run("Some obscure question")

    assert result.path == "rag_web"
    assert result.confidence is not None
    assert result.confidence.is_high_confidence is False


@patch("active_rag.answer_generator.OpenAI")
@patch("active_rag.confidence_checker.OpenAI")
def test_connection_error_returns_error_result(mock_conf_openai, mock_ans_openai):
    """Pipeline returns a friendly error when LLM API is unreachable."""
    mock_conf_client = MagicMock()
    mock_conf_openai.return_value = mock_conf_client
    # The retry decorator will wrap APIConnectionError in RetryError after attempts
    mock_conf_client.chat.completions.create.side_effect = APIConnectionError(
        request=MagicMock()
    )

    config = Config(confidence_threshold=0.7)
    # Disable cache/memory to simplify test
    pipeline = ActiveRAGPipeline(config, enable_cache=False, enable_memory=False)
    result = pipeline.run("anything")

    # After retries exhaust, pipeline should catch the error and return error result
    assert result.path == "error"
    assert result.answer.source == "error"
    assert "error" in result.answer.text.lower() or "Error" in result.answer.text
