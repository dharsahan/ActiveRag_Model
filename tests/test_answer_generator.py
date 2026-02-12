"""Tests for the answer generator module."""

from unittest.mock import MagicMock, patch

from active_rag.answer_generator import AnswerGenerator
from active_rag.config import Config
from active_rag.vector_store import RetrievalResult


def _mock_openai_response(content: str) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


@patch("active_rag.answer_generator.OpenAI")
def test_generate_direct(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_openai_response(
        "The answer is 42."
    )

    config = Config()
    generator = AnswerGenerator(config)
    answer = generator.generate_direct("What is the meaning of life?")

    assert answer.text == "The answer is 42."
    assert answer.source == "direct"
    assert answer.citations == []


@patch("active_rag.answer_generator.OpenAI")
def test_generate_with_citations(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _mock_openai_response(
        "Python is great (source: python.org)."
    )

    config = Config()
    generator = AnswerGenerator(config)
    context = [
        RetrievalResult(
            content="Python is a programming language",
            source_url="https://python.org",
            score=0.9,
        ),
    ]
    answer = generator.generate_with_citations("What is Python?", context)

    assert answer.source == "rag"
    assert "https://python.org" in answer.citations
    assert answer.text != ""
