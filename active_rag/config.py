"""Centralized configuration for the Active RAG system."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Configuration for the Active RAG pipeline."""

    # Ollama / LLM settings
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434/v1"
        )
    )
    model_name: str = field(
        default_factory=lambda: os.getenv("MODEL_NAME", "llama3.2")
    )

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
