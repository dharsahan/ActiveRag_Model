"""Tests for the Neo4j-backed vector store module."""

from unittest.mock import MagicMock, patch

import numpy as np

from active_rag.config import Config
from active_rag.vector_store import VectorStore


def _build_store(mock_driver, top_k: int = 3) -> VectorStore:
    config = Config(top_k=top_k, collection_name="test_index")
    with patch("active_rag.vector_store.Neo4jClient") as mock_client_cls, \
         patch("active_rag.vector_store.SentenceTransformer") as mock_st_cls, \
         patch("active_rag.vector_store.CrossEncoder") as mock_ce_cls:
        mock_client = MagicMock()
        mock_client._driver = mock_driver
        mock_client_cls.return_value = mock_client

        embedder = MagicMock()

        def _encode(value):
            if isinstance(value, str):
                return np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
            return np.array([[0.1, 0.2, 0.3, 0.4] for _ in value], dtype=float)

        embedder.encode.side_effect = _encode
        mock_st_cls.return_value = embedder

        reranker = MagicMock()
        reranker.predict.side_effect = lambda pairs: [0.95 - i * 0.1 for i in range(len(pairs))]
        mock_ce_cls.return_value = reranker

        return VectorStore(config)


def test_empty_store_returns_not_found():
    """Searching with no records should return found=False."""
    driver = MagicMock()
    session = MagicMock()
    session.run.side_effect = [iter(()), iter(())]  # index setup, then search
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__.return_value = None

    store = _build_store(driver)
    result = store.search("anything")
    assert result.found is False
    assert result.results == []


def test_add_and_search_returns_results():
    """Indexed documents should be returned by search."""
    driver = MagicMock()
    session = MagicMock()
    session.run.side_effect = [
        iter(()),  # _setup_vector_index
        iter(()),  # add_documents insert
        iter([
            {"content": "Python is great", "source_url": "https://python.org", "indexed_at": 10_000.0, "score": 0.91},
            {"content": "Python docs", "source_url": "https://docs.python.org", "indexed_at": 10_001.0, "score": 0.75},
        ]),  # search query
    ]
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__.return_value = None

    store = _build_store(driver, top_k=2)
    ids = store.add_documents(["Python is great"], ["https://python.org"])
    assert len(ids) == 1

    result = store.search("Python is great")
    assert result.found is True
    assert len(result.results) >= 1
    assert result.results[0].source_url in {"https://python.org", "https://docs.python.org"}


def test_add_empty_list_is_noop():
    """Adding empty document list is a no-op."""
    driver = MagicMock()
    session = MagicMock()
    session.run.side_effect = [iter(())]
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__.return_value = None

    store = _build_store(driver)
    ids = store.add_documents([], [])
    assert ids == []


def test_max_age_filters_stale_documents():
    """max_age_seconds should exclude stale records."""
    driver = MagicMock()
    session = MagicMock()
    session.run.side_effect = [
        iter(()),
        iter([
            {"content": "old", "source_url": "https://old", "indexed_at": 1.0, "score": 0.9},
            {"content": "new", "source_url": "https://new", "indexed_at": 9999999999.0, "score": 0.8},
        ]),
    ]
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__.return_value = None

    store = _build_store(driver)

    with patch("active_rag.vector_store.time.time", return_value=10_000.0):
        result = store.search("query", max_age_seconds=3600)

    assert result.found is True
    assert all(r.source_url != "https://old" for r in result.results)


def test_time_sensitive_detection_keywords():
    """Pipeline should detect freshness-oriented phrasing."""
    from active_rag.pipeline import ActiveRAGPipeline

    assert ActiveRAGPipeline._is_time_sensitive("current news in USA") is True
    assert ActiveRAGPipeline._is_time_sensitive("latest updates on elections") is True
    assert ActiveRAGPipeline._is_time_sensitive("today's weather") is True
    assert ActiveRAGPipeline._is_time_sensitive("What is Python?") is False
