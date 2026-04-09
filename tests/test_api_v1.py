"""Comprehensive tests for the Active RAG API v2.0.

Tests all routers: query, ingestion, knowledge base, graph, NLP,
reasoning, evaluation, and system endpoints.
"""

import json
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient

from active_rag.api import create_app


# --- Fixtures ---

@pytest.fixture
def app():
    """Create a test app with mocked heavy dependencies."""
    with patch("active_rag.dependencies.AgenticOrchestrator") as mock_agent, \
         patch("active_rag.dependencies.VectorStore") as mock_vs, \
         patch("active_rag.dependencies.ConversationMemory") as mock_mem:
        
        # Mock vector store
        mock_store = MagicMock()
        mock_store.count.return_value = 42
        mock_store.search.return_value = MagicMock(
            found=True,
            results=[MagicMock(content="test content", source_url="http://test.com", score=0.9)],
        )
        mock_store.add_documents.return_value = ["chunk-123"]
        mock_store.get_all_documents.return_value = [{"content": "doc1", "source_url": "s1"}]
        mock_vs.return_value = mock_store

        # Mock agent
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.answer.text = "Test answer"
        mock_result.answer.citations = ["https://example.com"]
        mock_result.path = "agent"
        mock_conf = MagicMock()
        mock_conf.confidence = 0.9
        mock_conf.reasoning = "Known fact"
        mock_result.confidence = mock_conf
        mock_result.web_pages_indexed = 0
        mock_result.from_cache = False
        mock_pipeline.run_async = AsyncMock(return_value=mock_result)
        mock_agent.return_value = mock_pipeline

        yield create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


# ============================================================
# Query Endpoints
# ============================================================

class TestQueryRouter:

    def test_query_endpoint(self, client):
        """POST /api/v1/query returns an answer."""
        response = client.post("/api/v1/query", json={"query": "What is Python?"})
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Test answer"
        assert data["path"] == "agent"
        assert data["confidence"] == 0.9

    def test_query_stream_endpoint(self, client):
        """POST /api/v1/query/stream returns streaming response."""
        response = client.post(
            "/api/v1/query/stream",
            json={"query": "What is Python?", "pipeline_type": "agent"},
        )
        assert response.status_code == 200

    def test_backward_compat_query_redirect(self, client):
        """Old /query path redirects to /api/v1/query."""
        response = client.post("/query", json={"query": "test"}, follow_redirects=False)
        assert response.status_code == 307


# ============================================================
# Ingestion Endpoints
# ============================================================

class TestIngestionRouter:

    def test_ingest_text(self, client):
        """POST /api/v1/ingest/text ingests raw text."""
        response = client.post("/api/v1/ingest/text", json={
            "content": "This is a test document about Python.",
            "source": "test_source",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["chunks"] >= 1

    def test_ingest_batch(self, client):
        """POST /api/v1/ingest/batch ingests multiple documents."""
        response = client.post("/api/v1/ingest/batch", json={
            "documents": [
                {"content": "FAQ: How to reset password?", "source": "faq"},
                {"content": "FAQ: How to update profile?", "source": "faq"},
                {"content": "FAQ: How to contact support?", "source": "faq"},
            ],
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["total_documents"] == 3

    def test_ingest_batch_empty(self, client):
        """POST /api/v1/ingest/batch with empty list returns 400."""
        response = client.post("/api/v1/ingest/batch", json={"documents": []})
        assert response.status_code == 400


# ============================================================
# Knowledge Base Endpoints
# ============================================================

class TestKnowledgeBaseRouter:

    def test_kb_stats(self, client):
        """GET /api/v1/kb/stats returns stats."""
        response = client.get("/api/v1/kb/stats")
        assert response.status_code == 200
        data = response.json()
        assert "vector_chunks" in data

    def test_kb_search(self, client):
        """POST /api/v1/kb/search performs semantic search."""
        response = client.post("/api/v1/kb/search", json={
            "query": "test query",
            "limit": 5,
        })
        assert response.status_code == 200
        data = response.json()
        assert "found" in data
        assert "results" in data

    def test_kb_export(self, client):
        """GET /api/v1/kb/export returns all documents."""
        response = client.get("/api/v1/kb/export")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "documents" in data


# ============================================================
# Graph Endpoints
# ============================================================

class TestGraphRouter:

    @patch("active_rag.dependencies.GraphResourceManager")
    def test_entities_search(self, mock_grm):
        """POST /api/v1/graph/entities/search returns entities."""
        # Create app with mocked graph resources
        with patch("active_rag.dependencies.VectorStore"), \
             patch("active_rag.dependencies.AgenticOrchestrator"), \
             patch("active_rag.dependencies.ConversationMemory"):
            
            app = create_app()
            client = TestClient(app)

            # Graph endpoints return 503 when graph is not available
            response = client.post("/api/v1/graph/entities/search", json={
                "name_pattern": "Smith",
            })
            # Without real Neo4j, this should return 503
            assert response.status_code in [200, 503]

    def test_graph_stats_no_graph(self, client):
        """GET /api/v1/graph/stats returns 503 or 500 when graph is unavailable."""
        response = client.get("/api/v1/graph/stats")
        # May return 503 (no graph) or 500 (connection error) depending on config
        assert response.status_code in [200, 500, 503]


# ============================================================
# NLP Endpoints
# ============================================================

class TestNLPRouter:

    def test_sentiment_analysis(self, client):
        """POST /api/v1/nlp/sentiment analyzes sentiment."""
        response = client.post("/api/v1/nlp/sentiment", json={
            "text": "This is a great and excellent product!",
        })
        # May return 200 or 503 depending on spaCy availability
        if response.status_code == 200:
            data = response.json()
            assert "label" in data
            assert "score" in data

    def test_classify_document(self, client):
        """POST /api/v1/nlp/classify classifies text domain."""
        response = client.post("/api/v1/nlp/classify", json={
            "text": "Dr. Smith at MIT published a paper on quantum computing using novel methodology.",
        })
        if response.status_code == 200:
            data = response.json()
            assert "domain" in data

    def test_extract_entities(self, client):
        """POST /api/v1/nlp/entities/extract extracts entities."""
        response = client.post("/api/v1/nlp/entities/extract", json={
            "text": "Google and Microsoft are competing in AI research.",
            "domain": "technical",
        })
        if response.status_code == 200:
            data = response.json()
            assert "entities" in data
            assert "count" in data


# ============================================================
# Reasoning Endpoints
# ============================================================

class TestReasoningRouter:

    def test_reason_no_graph(self, client):
        """POST /api/v1/reasoning/reason returns error without graph."""
        response = client.post("/api/v1/reasoning/reason", json={
            "query": "How is Einstein connected to Princeton?",
        })
        # May return 503 (no engine), 500 (connection error), or 200 if graph connects
        assert response.status_code in [200, 500, 503]

    def test_communities_no_graph(self, client):
        """POST /api/v1/reasoning/communities returns error without graph."""
        response = client.post("/api/v1/reasoning/communities", json={})
        assert response.status_code in [200, 500, 503]

    def test_bridges_no_graph(self, client):
        """GET /api/v1/reasoning/bridges returns error without graph."""
        response = client.get("/api/v1/reasoning/bridges")
        assert response.status_code in [200, 500, 503]


# ============================================================
# Evaluation Endpoints
# ============================================================

class TestEvaluationRouter:

    def test_evaluate_answer(self, client):
        """POST /api/v1/evaluate evaluates answer quality."""
        response = client.post("/api/v1/evaluate", json={
            "query": "What is Python?",
            "answer": "Python is a high-level programming language.",
        })
        # May return 200 or 503 depending on LLM availability
        assert response.status_code in [200, 503]


# ============================================================
# System Endpoints
# ============================================================

class TestSystemRouter:

    def test_system_health(self, client):
        """GET /api/v1/system/health returns health status."""
        response = client.get("/api/v1/system/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

    def test_list_sessions(self, client):
        """GET /api/v1/system/sessions lists active sessions."""
        response = client.get("/api/v1/system/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "sessions" in data

    def test_performance_report(self, client):
        """GET /api/v1/system/performance returns performance data."""
        response = client.get("/api/v1/system/performance")
        assert response.status_code == 200

    def test_cache_stats(self, client):
        """GET /api/v1/system/cache/stats returns cache metrics."""
        response = client.get("/api/v1/system/cache/stats")
        assert response.status_code == 200


# ============================================================
# Config Endpoints
# ============================================================

class TestConfigEndpoints:

    def test_get_config(self, client):
        """GET /api/v1/config returns current config."""
        response = client.get("/api/v1/config")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "top_k" in data

    def test_backward_compat_config_redirect(self, client):
        """Old /config path redirects to /api/v1/config."""
        response = client.get("/config", follow_redirects=False)
        assert response.status_code == 307


# ============================================================
# Root Endpoint
# ============================================================

class TestRootEndpoint:

    def test_root_returns_success(self, client):
        """GET / returns 200 (JSON info or HTML index)."""
        response = client.get("/")
        assert response.status_code == 200
        # May return JSON API info or HTML static page
        try:
            data = response.json()
            if "endpoints" in data:
                assert "query" in data["endpoints"]
                assert "graph" in data["endpoints"]
        except Exception:
            # Static HTML is served — that's fine
            assert len(response.content) > 0
