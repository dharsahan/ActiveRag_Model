"""Compatibility tests for legacy API paths and auth behavior."""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from active_rag.api import create_app


@patch("active_rag.dependencies.AgenticOrchestrator")
@patch("active_rag.dependencies.VectorStore")
@patch("active_rag.dependencies.ConversationMemory")
def test_query_endpoint_legacy_redirect(mock_mem_cls, mock_vs_cls, mock_agent_cls):
    """POST /query should redirect and still return a valid query response."""
    mock_store = MagicMock()
    mock_store.count.return_value = 0
    mock_store.search.return_value = MagicMock(found=False, results=[])
    mock_vs_cls.return_value = mock_store

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
    mock_agent_cls.return_value = mock_pipeline

    app = create_app()
    client = TestClient(app)

    response = client.post("/query", json={"query": "What is Python?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Test answer"
    assert data["path"] == "agent"
    assert data["confidence"] == 0.9


def test_health_endpoint_legacy_redirect():
    """GET /system/health should redirect and return health JSON."""
    app = create_app()
    client = TestClient(app)

    response = client.get("/system/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_api_key_auth_is_enforced_when_configured(monkeypatch):
    """If ACTIVE_RAG_API_KEY is set, requests must include X-API-Key."""
    monkeypatch.setenv("ACTIVE_RAG_API_KEY", "secret-key")
    app = create_app()
    client = TestClient(app)

    missing = client.get("/api/v1/config")
    assert missing.status_code == 401

    wrong = client.get("/api/v1/config", headers={"X-API-Key": "wrong"})
    assert wrong.status_code == 401

    ok = client.get("/api/v1/config", headers={"X-API-Key": "secret-key"})
    assert ok.status_code == 200
