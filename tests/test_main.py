"""Tests for the CLI entry point (main.py)."""

from unittest.mock import patch

import httpx
import pytest

from main import _check_ollama
from active_rag.config import Config


def test_check_ollama_exits_on_connection_error():
    """_check_ollama exits with code 1 and a clear message when unreachable."""
    config = Config()
    with patch("main.httpx.get", side_effect=httpx.ConnectError("refused")):
        with pytest.raises(SystemExit) as exc_info:
            _check_ollama(config)
        assert exc_info.value.code == 1


def test_check_ollama_exits_on_timeout():
    """_check_ollama exits with code 1 on timeout."""
    config = Config()
    with patch("main.httpx.get", side_effect=httpx.TimeoutException("timeout")):
        with pytest.raises(SystemExit) as exc_info:
            _check_ollama(config)
        assert exc_info.value.code == 1
