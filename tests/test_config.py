"""Tests for the configuration module."""

from active_rag.config import Config


def test_default_config_values(monkeypatch):
    """Default configuration uses sensible defaults."""
    for key in (
        "LLM_PROVIDER",
        "MODEL_NAME",
        "OLLAMA_BASE_URL",
        "LLM_API_KEY",
        "NVIDIA_API_KEY",
        "LLM_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)

    config = Config()
    assert config.confidence_threshold == 0.7
    assert config.top_k == 3
    assert config.max_search_results == 3
    assert config.collection_name == "active_rag"
    # Default provider is "local" → model = "gpt-5.2", base_url = "http://localhost:4141"
    assert config.model_name == "gpt-5.2"
    assert config.ollama_base_url == "http://localhost:4141"


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
