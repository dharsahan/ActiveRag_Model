"""Tests for the LLM provider abstraction."""

import pytest
from active_rag.providers import get_provider_config, list_providers


def test_nvidia_provider():
    config = get_provider_config("nvidia")
    assert "nvidia.com" in config["base_url"]
    assert config["api_key_env"] == "NVIDIA_API_KEY"


def test_ollama_provider():
    config = get_provider_config("ollama")
    assert "localhost" in config["base_url"]
    assert config["api_key"] == "ollama"


def test_openai_provider():
    config = get_provider_config("openai")
    assert "openai.com" in config["base_url"]
    assert config["api_key_env"] == "OPENAI_API_KEY"


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_provider_config("unknown_provider")


def test_list_providers():
    providers = list_providers()
    assert "nvidia" in providers
    assert "openai" in providers
    assert "ollama" in providers
