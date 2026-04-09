"""Reasoning & Analytics router — /api/v1/reasoning endpoints.

Exposes multi-hop reasoning, community detection, and cross-domain discovery.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from active_rag.dependencies import GraphResourceManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/reasoning", tags=["Reasoning & Analytics"])


# --- Models ---

class ReasonRequest(BaseModel):
    query: str = Field(..., description="Natural language question for multi-hop reasoning")
    max_hops: int = Field(default=2, ge=1, le=5)


class CommunityDetectionRequest(BaseModel):
    entity_type: Optional[str] = Field(None, description="Filter by entity type (e.g., 'Person')")
    max_entities: int = Field(default=200, ge=1, le=1000)


class CrossDomainRequest(BaseModel):
    entity_id: str = Field(..., description="Starting entity ID")
    source_domain: Optional[str] = Field(None, description="Domain of the starting entity")
    max_depth: int = Field(default=2, ge=1, le=5)


def register(r: APIRouter, graph_resources: GraphResourceManager):
    """Register reasoning and analytics endpoints."""

    @r.post("/reason")
    async def reason(req: ReasonRequest):
        """Execute full multi-hop reasoning pipeline.

        Steps:
        1. Extract seed entities from the query (NLP)
        2. Expand entity neighborhoods in the graph
        3. Extract relevant subgraph (nodes + edges)
        4. Find and rank reasoning paths by relevance
        5. Return structured result with confidence score

        Use cases:
        - "How is Dr. Smith connected to the Stanford AI Lab?"
        - "What links Client X to Case Y through intermediaries?"
        - "Trace the citation chain between these two papers"
        """
        engine = graph_resources.reasoning_engine
        if engine is None:
            raise HTTPException(
                status_code=503,
                detail="Reasoning engine is not available. Ensure graph features are enabled.",
            )

        try:
            result = engine.reason(req.query, max_hops=req.max_hops)
            return {
                "query": result.query,
                "confidence": result.confidence,
                "reasoning_summary": result.reasoning_summary,
                "has_results": result.has_results,
                "seed_entities": result.seed_entities,
                "ranked_paths": [
                    {
                        "reasoning_text": p.reasoning_text,
                        "score": p.score,
                        "length": p.length,
                        "start_entity": p.start_entity,
                        "end_entity": p.end_entity,
                        "relationships": p.relationships,
                    }
                    for p in result.ranked_paths
                ],
                "supporting_entities": result.supporting_entities,
                "subgraph": {
                    "node_count": result.subgraph.node_count,
                    "edge_count": result.subgraph.edge_count,
                    "seed_entity_ids": result.subgraph.seed_entity_ids,
                },
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @r.post("/communities")
    async def detect_communities(req: CommunityDetectionRequest):
        """Detect entity communities using label propagation.

        Groups related entities into clusters based on their graph connections.

        Use cases:
        - Identify research collaboration groups
        - Find organizational clusters in legal records
        - Discover topic clusters in FAQ knowledge bases
        """
        detector = graph_resources.community_detector
        if detector is None:
            raise HTTPException(
                status_code=503,
                detail="Community detector is not available. Ensure graph features are enabled.",
            )

        ops = graph_resources.graph_ops
        if ops is None:
            raise HTTPException(status_code=503, detail="Graph backend is not available.")

        try:
            communities = detector.detect_communities(
                ops,
                entity_type=req.entity_type,
                max_entities=req.max_entities,
            )
            return {
                "count": len(communities),
                "communities": [
                    {
                        "community_id": c.community_id,
                        "size": c.size,
                        "dominant_label": c.dominant_label,
                        "entity_names": c.entity_names,
                    }
                    for c in communities
                ],
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @r.post("/cross-domain")
    async def find_cross_domain_links(req: CrossDomainRequest):
        """Find cross-domain entity connections.

        Discovers how an entity in one domain (e.g., Research) connects
        to entities in other domains (e.g., Technical, Business).

        Use cases:
        - Find how a researcher connects to industry organizations
        - Discover how legal cases span different practice areas
        """
        discovery = graph_resources.cross_domain
        if discovery is None:
            raise HTTPException(
                status_code=503,
                detail="Cross-domain discovery is not available. Ensure graph features are enabled.",
            )

        ops = graph_resources.graph_ops
        if ops is None:
            raise HTTPException(status_code=503, detail="Graph backend is not available.")

        try:
            links = discovery.find_cross_domain_links(
                ops,
                entity_id=req.entity_id,
                source_domain=req.source_domain,
                max_depth=req.max_depth,
            )
            return {
                "entity_id": req.entity_id,
                "count": len(links),
                "links": [
                    {
                        "source_domain": l.source_domain,
                        "target_entity": l.target_entity,
                        "target_domain": l.target_domain,
                        "relationship_type": l.relationship_type,
                        "path_length": l.path_length,
                    }
                    for l in links
                ],
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @r.get("/bridges")
    async def discover_bridges(
        max_entities: int = Query(default=50, ge=1, le=500),
    ):
        """Discover bridge entities that span multiple domains.

        Bridge entities are connected to entities across different content domains,
        making them key connectors in the knowledge graph.

        Use cases:
        - Find key people who bridge research and industry
        - Identify technologies that connect multiple business units
        """
        discovery = graph_resources.cross_domain
        if discovery is None:
            raise HTTPException(
                status_code=503,
                detail="Cross-domain discovery is not available. Ensure graph features are enabled.",
            )

        ops = graph_resources.graph_ops
        if ops is None:
            raise HTTPException(status_code=503, detail="Graph backend is not available.")

        try:
            bridges = discovery.discover_bridges(ops, max_entities=max_entities)
            return {
                "count": len(bridges),
                "bridges": bridges,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
