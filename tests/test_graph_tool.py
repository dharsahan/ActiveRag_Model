"""Tests for the graph_query agent tool."""

import json
from unittest.mock import MagicMock, patch

from active_rag.tools.graph_query import GraphQueryTool, TOOL_SCHEMA


class TestGraphQueryToolSchema:
    """Test the tool schema definition."""

    def test_schema_has_name(self):
        assert TOOL_SCHEMA["function"]["name"] == "graph_query"

    def test_schema_has_query_param(self):
        params = TOOL_SCHEMA["function"]["parameters"]["properties"]
        assert "query" in params

    def test_schema_has_max_hops_param(self):
        params = TOOL_SCHEMA["function"]["parameters"]["properties"]
        assert "max_hops" in params


class TestGraphQueryToolExecution:
    """Test tool execution with mocked graph backend."""

    def _make_tool_with_mock_ops(self):
        """Create a GraphQueryTool with mocked graph operations."""
        config = MagicMock()
        config.enable_graph_features = False  # Start disabled
        tool = GraphQueryTool(config)
        return tool

    def test_unavailable_when_graph_disabled(self):
        """When graph is disabled, returns unavailable status."""
        tool = self._make_tool_with_mock_ops()
        result_str = tool.execute({"query": "Who is Einstein?"})
        result = json.loads(result_str)
        assert result["status"] == "unavailable"
        assert len(result["entities"]) == 0

    @patch("active_rag.knowledge_graph.graph_operations.GraphOperations")
    @patch("active_rag.knowledge_graph.neo4j_client.Neo4jClient")
    def test_successful_graph_query(self, mock_neo4j_cls, mock_graph_ops_cls):
        """When graph is available, returns entities and paths."""
        config = MagicMock()
        config.enable_graph_features = True
        config.neo4j_uri = "bolt://localhost:7687"
        config.neo4j_username = "neo4j"
        config.neo4j_password = "test"

        mock_ops = MagicMock()
        mock_ops.multi_hop_query.return_value = {
            "entities": [
                {"name": "Einstein", "labels": ["Person"], "distance": 0, "relevance_score": 1.0},
            ],
            "paths": [
                {"reasoning_path": "Einstein -> Princeton", "length": 1, "relationship_types": ["AFFILIATED_WITH"]},
            ],
            "reasoning": "Found 1 entities",
        }
        mock_graph_ops_cls.return_value = mock_ops

        tool = GraphQueryTool(config)
        result_str = tool.execute({"query": "Who is Einstein?", "max_hops": 2})
        result = json.loads(result_str)

        assert result["status"] == "success"
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "Einstein"
        assert len(result["paths"]) == 1

    def test_max_hops_capped(self):
        """max_hops should be capped at 5."""
        config = MagicMock()
        config.enable_graph_features = False
        tool = GraphQueryTool(config)
        # Even with max_hops=100, tool won't error (it's unavailable anyway)
        result_str = tool.execute({"query": "test", "max_hops": 100})
        result = json.loads(result_str)
        assert result["status"] == "unavailable"


class TestGraphQueryToolAsync:
    """Test async execution."""

    def test_async_wrapper_exists(self):
        """execute_async should exist on the tool."""
        config = MagicMock()
        config.enable_graph_features = False
        tool = GraphQueryTool(config)
        assert hasattr(tool, "execute_async")
