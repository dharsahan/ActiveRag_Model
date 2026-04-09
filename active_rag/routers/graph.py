"""Knowledge Graph router — /api/v1/graph endpoints.

Exposes entity search, neighborhood exploration, path finding,
multi-hop reasoning, and graph statistics.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from active_rag.dependencies import GraphResourceManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/graph", tags=["Knowledge Graph"])


# --- Models ---

class EntitySearchRequest(BaseModel):
    name_pattern: str = Field(..., description="Name or partial name to search for")
    entity_types: Optional[List[str]] = Field(None, description="Filter by entity types (e.g., ['Person', 'Organization'])")


class FindPathsRequest(BaseModel):
    start_id: str = Field(..., description="Starting entity ID")
    end_id: str = Field(..., description="Target entity ID")
    max_depth: int = Field(default=3, ge=1, le=10)


class MultiHopRequest(BaseModel):
    query: str = Field(..., description="Natural language query for multi-hop reasoning")
    max_hops: int = Field(default=2, ge=1, le=5)


def register(r: APIRouter, graph_resources: GraphResourceManager):
    """Register knowledge graph endpoints."""

    def _require_graph():
        if graph_resources.graph_ops is None:
            raise HTTPException(
                status_code=503,
                detail="Knowledge graph is not available. Enable graph features and ensure Neo4j is running.",
            )
        return graph_resources.graph_ops

    @r.post("/entities/search")
    async def search_entities(req: EntitySearchRequest):
        """Search for entities by name pattern, optionally filtered by type.

        Use cases:
        - Find all people named "Smith" in legal records
        - Search for organizations matching "MIT"
        - Look up technical components like "Neo4j"
        """
        ops = _require_graph()
        try:
            results = ops.search_entities_by_name(
                req.name_pattern,
                entity_types=req.entity_types,
            )
            return {
                "count": len(results),
                "entities": results,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @r.get("/entities/{entity_id}/neighborhood")
    async def get_neighborhood(
        entity_id: str,
        radius: int = Query(default=2, ge=1, le=10, description="Hop radius"),
    ):
        """Get the neighborhood of entities around a given entity.

        Returns all entities within the specified radius, sorted by distance.
        Useful for exploring context around a person, organization, or concept.
        """
        ops = _require_graph()
        try:
            neighbors = ops.get_entity_neighborhood(entity_id, radius=radius)
            return {
                "entity_id": entity_id,
                "radius": radius,
                "count": len(neighbors),
                "neighbors": neighbors,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @r.get("/entities/{entity_id}/related")
    async def get_related_entities(
        entity_id: str,
        depth: int = Query(default=1, ge=1, le=5),
    ):
        """Find entities directly related to a given entity.

        Returns related entities with their relationship types.
        """
        ops = _require_graph()
        try:
            related = ops.find_related_entities(entity_id, [], depth=depth)
            return {
                "entity_id": entity_id,
                "depth": depth,
                "count": len(related),
                "related": related,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @r.post("/paths")
    async def find_paths(req: FindPathsRequest):
        """Find paths between two entities.

        Discovers how two entities are connected through the knowledge graph.
        Essential for legal case analysis, research citation chains, etc.
        """
        ops = _require_graph()
        try:
            paths = ops.find_paths(req.start_id, req.end_id, max_depth=req.max_depth)
            return {
                "start_id": req.start_id,
                "end_id": req.end_id,
                "max_depth": req.max_depth,
                "count": len(paths),
                "paths": paths,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @r.post("/multi-hop")
    async def multi_hop_query(req: MultiHopRequest):
        """Execute a multi-hop reasoning query using NLP entity extraction.

        Automatically extracts entities from the query, searches the graph,
        and returns relevant entities and reasoning paths.
        """
        ops = _require_graph()
        try:
            result = ops.multi_hop_query(req.query, max_hops=req.max_hops)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @r.get("/stats")
    async def graph_stats():
        """Get detailed knowledge graph statistics.

        Returns total nodes, relationships, node types, and relationship types.
        """
        ops = _require_graph()
        try:
            stats = ops.get_graph_stats()
            return stats
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
