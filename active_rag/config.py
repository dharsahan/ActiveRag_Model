"""Centralized configuration for the Active RAG system."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class Config:
    """Configuration for the Active RAG pipeline."""

    # Ollama / LLM settings
    provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "local").lower()
    )
    ollama_base_url: str | None = None
    model_name: str | None = None
    api_key: str | None = None

    # Cache settings
    cache_dir: str | None = None

    # Legacy compatibility fields from pre-Neo4j releases.
    # These are kept so older tests/scripts can still construct Config.
    collection_name: str | None = None
    chroma_persist_dir: str | None = None

    # Vector store settings (Neo4j-backed)
    vector_index_name: str = field(
        default_factory=lambda: os.getenv(
            "VECTOR_INDEX_NAME",
            os.getenv("COLLECTION_NAME", "active_rag"),
        )
    )
    top_k: int = field(
        default_factory=lambda: int(os.getenv("TOP_K", "3"))
    )

    def __post_init__(self) -> None:
        from active_rag.providers import get_provider_config
        cfg = get_provider_config(self.provider)
        
        if self.ollama_base_url is None:
            self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", cfg["base_url"])
            
        if self.model_name is None:
            val = os.getenv("MODEL_NAME", cfg["default_model"])
            self.model_name = val.strip() if val else val
            
        if self.api_key is None:
            if "api_key_env" in cfg:
                default_key = os.getenv(cfg["api_key_env"], "ollama")
            else:
                default_key = cfg.get("api_key", "ollama")
            self.api_key = os.getenv("LLM_API_KEY", default_key)

        if self.collection_name is None:
            self.collection_name = os.getenv("COLLECTION_NAME", self.vector_index_name)

        # If legacy collection_name is explicitly set, mirror it to vector index.
        if self.collection_name:
            self.vector_index_name = self.collection_name

        if self.chroma_persist_dir is None:
            self.chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR", ".cache/chroma")

        if self.cache_dir is None:
            self.cache_dir = os.getenv("CACHE_DIR", ".cache")

    # Confidence threshold (0.0–1.0). Scores at or above this value are
    # considered "high confidence" and skip RAG retrieval.
    confidence_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("CONFIDENCE_THRESHOLD", "0.7")
        )
    )

    # Web search settings
    max_search_results: int = field(
        default_factory=lambda: int(os.getenv("MAX_SEARCH_RESULTS", "3"))
    )
    headless: bool = field(
        default_factory=lambda: os.getenv("HEADLESS", "true").lower() == "true"
    )

    # Time-sensitive query settings (seconds). Documents older than this
    # are skipped when the query is about current/recent events.
    time_sensitive_max_age: int = field(
        default_factory=lambda: int(os.getenv("TIME_SENSITIVE_MAX_AGE", "3600"))
    )

    # Neo4j / Knowledge Graph settings
    neo4j_uri: str = field(
        default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687")
    )
    neo4j_username: str = field(
        default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j")
    )
    neo4j_password: str = field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", "activerag123")
    )

    # Graph feature toggles
    enable_graph_features: bool = field(
        default_factory=lambda: os.getenv("ENABLE_GRAPH_FEATURES", "true").lower() == "true"
    )
    max_graph_hops: int = field(
        default_factory=lambda: int(os.getenv("MAX_GRAPH_HOPS", "3"))
    )

    # NLP Pipeline settings
    spacy_model: str = field(
        default_factory=lambda: os.getenv("SPACY_MODEL", "en_core_web_sm")
    )
    enable_relation_extraction: bool = field(
        default_factory=lambda: os.getenv("ENABLE_RELATION_EXTRACTION", "true").lower() == "true"
    )
