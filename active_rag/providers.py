"""Multi-provider LLM configuration."""

from __future__ import annotations

_PROVIDERS: dict[str, dict[str, str]] = {
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "default_model": "nvidia/llama-3.1-70b-instruct",
        "api_key_env": "NVIDIA_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3.2",
        "api_key": "ollama",
    },
    "openai": {
        "base_url": "http://localhost:4141/v1",
        "default_model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.1-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "default_model": "meta-llama/Llama-3-70b-chat-hf",
        "api_key_env": "TOGETHER_API_KEY",
    },
    "local":{
        "base_url": "http://localhost:4141/v1",
        "default_model": "gpt-4o",
        "api_key": "ollama",
    }
}


def get_provider_config(provider: str) -> dict[str, str]:
    """Return base URL, default model, and API key info for *provider*."""
    provider = provider.lower()
    if provider not in _PROVIDERS:
        available = ", ".join(sorted(_PROVIDERS.keys()))
        raise ValueError(f"Unknown LLM provider: {provider}. Available: {available}")
    return _PROVIDERS[provider]


def list_providers() -> list[str]:
    """Return all available provider names."""
    return sorted(_PROVIDERS.keys())
