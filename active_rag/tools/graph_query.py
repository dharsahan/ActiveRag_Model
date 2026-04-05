"""Graph Query tool for the autonomous agent.

Enables the LLM agent to query the Neo4j knowledge graph
for entity relationships and multi-hop reasoning.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from active_rag.config import Config

logger = logging.getLogger(__name__)

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "graph_query",
        "description": (
            "Search the knowledge graph for entity relationships and connections. "
            "Use this tool when the user asks about relationships between people, "
            "organizations, concepts, or when multi-hop reasoning is needed. "
            "Examples: 'Who collaborated with Einstein?', 'Which components depend on the auth API?'"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The natural language query to search the knowledge graph for.",
                },
                "max_hops": {
                    "type": "integer",
                    "description": "Maximum graph traversal depth (1-5). Use 1 for direct relationships, 2-3 for indirect connections.",
                    "default": 2,
                },
            },
            "required": ["query"],
        },
    },
}


class GraphQueryTool:
    """Agent tool wrapping graph operations for knowledge graph queries."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._graph_ops = None
        self._initialize()

    @property
    def schema(self) -> dict:
        return TOOL_SCHEMA

    def _initialize(self) -> None:
        """Lazily initialize graph operations (skip if graph disabled or Neo4j unavailable)."""
        if not self._config.enable_graph_features:
            logger.info("Graph features disabled — graph_query tool will return fallback responses.")
            return

        try:
            from active_rag.knowledge_graph.neo4j_client import Neo4jClient
            from active_rag.knowledge_graph.graph_operations import GraphOperations

            client = Neo4jClient(
                self._config.neo4j_uri,
                self._config.neo4j_username,
                self._config.neo4j_password,
            )
            self._graph_ops = GraphOperations(client)
            logger.info("GraphQueryTool initialized with Neo4j connection.")
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j — graph_query tool in fallback mode: {e}")
            self._graph_ops = None

    def execute(self, args: Dict[str, Any]) -> str:
        """Execute a graph query synchronously.

        Args:
            args: {"query": "...", "max_hops": 2}

        Returns:
            JSON string with entities, paths, and reasoning.
        """
        query = args.get("query", "")
        max_hops = min(int(args.get("max_hops", 2)), 5)

        if not self._graph_ops:
            return json.dumps({
                "status": "unavailable",
                "message": "Knowledge graph is not available. Try using web_browser or query_memory instead.",
                "entities": [],
                "paths": [],
            })

        try:
            result = self._graph_ops.multi_hop_query(query, max_hops=max_hops)

            # Format entities for readability
            formatted_entities = []
            for entity in result.get("entities", []):
                formatted_entities.append({
                    "name": entity.get("name", "Unknown"),
                    "labels": entity.get("labels", []),
                    "distance": entity.get("distance", 0),
                    "relevance": entity.get("relevance_score", 0.0),
                })

            # Format paths for readability
            formatted_paths = []
            for path in result.get("paths", []):
                formatted_paths.append({
                    "reasoning_path": path.get("reasoning_path", ""),
                    "length": path.get("length", 0),
                    "relationships": path.get("relationship_types", []),
                })

            return json.dumps({
                "status": "success",
                "reasoning": result.get("reasoning", ""),
                "entities": formatted_entities[:10],
                "paths": formatted_paths[:5],
            }, indent=2)

        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return json.dumps({
                "status": "error",
                "message": f"Graph query failed: {str(e)}",
                "entities": [],
                "paths": [],
            })

    async def execute_async(self, args: Dict[str, Any]) -> str:
        """Async wrapper — graph queries are synchronous under the hood."""
        return self.execute(args)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from the graph backend."""
        if not self._graph_ops:
            return {"status": "unavailable"}
        return self._graph_ops.get_graph_stats()
