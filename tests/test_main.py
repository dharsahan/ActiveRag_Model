"""Tests for the CLI entry point (main.py)."""

from unittest.mock import patch

import httpx
import pytest

from main import _check_llm_backend
from active_rag.config import Config


def test_check_llm_backend_exits_on_connection_error():
    """_check_llm_backend exits with code 1 and a clear message when unreachable."""
    config = Config(ollama_base_url="http://localhost:11434/v1")
    with patch("main.httpx.get", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(SystemExit) as exc_info:
            _check_llm_backend(config)
        assert exc_info.value.code == 1


def test_check_llm_backend_exits_on_timeout():
    """_check_llm_backend exits with code 1 on timeout."""
    config = Config(ollama_base_url="http://localhost:11434/v1")
    with patch("main.httpx.get", side_effect=httpx.TimeoutException("timeout")):
        with pytest.raises(SystemExit) as exc_info:
            _check_llm_backend(config)
        assert exc_info.value.code == 1
