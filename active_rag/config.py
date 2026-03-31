"""Centralized configuration for the Active RAG system."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Configuration for the Active RAG pipeline."""

    # Ollama / LLM settings
    provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "nvidia").lower()
    )
    ollama_base_url: str | None = None
    model_name: str | None = None
    api_key: str | None = None

    def __post_init__(self) -> None:
        from active_rag.providers import get_provider_config
        cfg = get_provider_config(self.provider)
        
        if self.ollama_base_url is None:
            self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", cfg["base_url"])
            
        if self.model_name is None:
            self.model_name = os.getenv("MODEL_NAME", cfg["default_model"])
            
        if self.api_key is None:
            if "api_key_env" in cfg:
                default_key = os.getenv(cfg["api_key_env"], "ollama")
            else:
                default_key = cfg.get("api_key", "ollama")
            self.api_key = os.getenv("LLM_API_KEY", default_key)

    # Confidence threshold (0.0–1.0). Scores at or above this value are
    # considered "high confidence" and skip RAG retrieval.
    confidence_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("CONFIDENCE_THRESHOLD", "0.7")
        )
    )

    # ChromaDB / vector store settings
    chroma_persist_dir: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv("COLLECTION_NAME", "active_rag")
    )
    top_k: int = field(
        default_factory=lambda: int(os.getenv("TOP_K", "3"))
    )

    # Web search settings
    max_search_results: int = field(
        default_factory=lambda: int(os.getenv("MAX_SEARCH_RESULTS", "3"))
    )

    # Time-sensitive query settings (seconds). Documents older than this
    # are skipped when the query is about current/recent events.
    time_sensitive_max_age: int = field(
        default_factory=lambda: int(os.getenv("TIME_SENSITIVE_MAX_AGE", "3600"))
    )
