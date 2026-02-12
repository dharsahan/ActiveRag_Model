"""Tests for the vector store module."""

import hashlib
import uuid

import chromadb

from active_rag.config import Config
from active_rag.vector_store import VectorStore


class _SimpleEmbeddingFunction(chromadb.EmbeddingFunction):
    """Deterministic hash-based embedding for offline testing."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = []
        for text in input:
            digest = hashlib.sha256(text.encode()).digest()
            # Repeat digest to produce a 64-dimensional vector
            vec = [float(b) / 255.0 for b in (digest + digest)]
            embeddings.append(vec)
        return embeddings


def _make_store() -> VectorStore:
    """Create an in-memory vector store for testing."""
    config = Config(
        chroma_persist_dir="",
        collection_name=f"test_{uuid.uuid4().hex[:8]}",
    )
    return VectorStore(config, embedding_function=_SimpleEmbeddingFunction())


def test_empty_store_returns_not_found():
    """Searching an empty store should return found=False."""
    store = _make_store()
    result = store.search("anything")
    assert result.found is False
    assert result.results == []


def test_add_and_search():
    """Documents added to the store can be retrieved."""
    store = _make_store()
    store.add_documents(
        contents=["Python is a programming language"],
        source_urls=["https://python.org"],
    )
    # Use the exact same text to guarantee a high-similarity match
    result = store.search("Python is a programming language")
    assert result.found is True
    assert len(result.results) >= 1
    assert result.results[0].source_url == "https://python.org"


def test_add_multiple_documents():
    """Multiple documents can be indexed and searched."""
    store = _make_store()
    store.add_documents(
        contents=[
            "Rust is a systems programming language",
            "JavaScript is used for web development",
        ],
        source_urls=[
            "https://rust-lang.org",
            "https://developer.mozilla.org",
        ],
    )
    # Use exact document text to guarantee match
    result = store.search("Rust is a systems programming language")
    assert result.found is True
    assert len(result.results) >= 1


def test_add_empty_list_is_noop():
    """Adding an empty list should not raise."""
    store = _make_store()
    store.add_documents(contents=[], source_urls=[])
    result = store.search("anything")
    assert result.found is False
