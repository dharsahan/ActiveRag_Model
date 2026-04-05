"""Tests for the REST API."""

import json
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

from active_rag.api import create_app


@patch("active_rag.api.AgenticOrchestrator")
def test_query_endpoint(mock_pipeline_cls):
    """POST /query returns an answer."""
    mock_pipeline = MagicMock()
    mock_result = MagicMock()
    mock_result.answer.text = "Test answer"
    mock_result.answer.citations = ["https://example.com"]
    mock_result.path = "direct"
    
    # Mock confidence object
    mock_conf = MagicMock()
    mock_conf.confidence = 0.9
    mock_conf.reasoning = "Known fact"
    mock_result.confidence = mock_conf
    
    mock_result.web_pages_indexed = 0
    mock_result.from_cache = False
    # API uses await pipeline.run_async(), so we need AsyncMock
    mock_pipeline.run_async = AsyncMock(return_value=mock_result)
    mock_pipeline_cls.return_value = mock_pipeline

    app = create_app()
    client = TestClient(app)

    response = client.post("/query", json={"query": "What is Python?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Test answer"
    assert data["path"] == "direct"
    assert data["confidence"] == 0.9


@patch("active_rag.api.AgenticOrchestrator")
def test_health_endpoint(mock_pipeline_cls):
    """GET /health returns ok."""
    app = create_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@patch("active_rag.api.AgenticOrchestrator")
def test_clear_endpoints(mock_pipeline_cls):
    """POST /clear-memory and /clear-cache work."""
    mock_pipeline = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline

    app = create_app()
    client = TestClient(app)

    res1 = client.post("/clear-memory")
    assert res1.status_code == 200
    mock_pipeline.clear_memory.assert_called_once()

    res2 = client.post("/clear-cache")
    assert res2.status_code == 200
    mock_pipeline.clear_cache.assert_called_once()
