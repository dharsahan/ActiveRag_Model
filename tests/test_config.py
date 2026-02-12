"""Tests for the configuration module."""

from active_rag.config import Config


def test_default_config_values():
    """Default configuration uses sensible defaults."""
    config = Config()
    assert config.confidence_threshold == 0.7
    assert config.top_k == 3
    assert config.max_search_results == 3
    assert config.collection_name == "active_rag"
    assert config.model_name == "llama3.2"
    assert config.ollama_base_url == "http://localhost:11434/v1"


def test_custom_config_values():
    """Configuration can be overridden."""
    config = Config(
        confidence_threshold=0.9,
        top_k=5,
        max_search_results=10,
        model_name="mistral",
        ollama_base_url="http://myhost:11434/v1",
    )
    assert config.confidence_threshold == 0.9
    assert config.top_k == 5
    assert config.max_search_results == 10
    assert config.model_name == "mistral"
    assert config.ollama_base_url == "http://myhost:11434/v1"
