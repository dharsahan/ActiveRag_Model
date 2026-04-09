"""Knowledge Base router — /api/v1/kb endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from active_rag.config import Config
from active_rag.dependencies import ResourceManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/kb", tags=["Knowledge Base"])


class KBSearchRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    limit: int = Field(default=5, ge=1, le=100)


def register(r: APIRouter, resources: ResourceManager, cfg: Config):
    """Register knowledge base management endpoints."""

    @r.get("/stats")
    async def kb_stats():
        """Get statistics for the Knowledge Base (vector + graph)."""
        count = resources.vector_store.count()
        graph_stats = {}
        try:
            if cfg.enable_graph_features:
                from active_rag.tools.graph_query import GraphQueryTool
                tool = GraphQueryTool(cfg)
                graph_stats = tool.get_stats()
        except Exception:
            pass

        return {
            "vector_chunks": count,
            "graph_nodes": graph_stats.get("total_nodes", 0),
            "graph_relations": graph_stats.get("total_relationships", 0),
            "index_name": cfg.vector_index_name,
        }

    @r.post("/search")
    async def kb_search(req: KBSearchRequest):
        """Perform raw semantic search in the Knowledge Base."""
        old_k = cfg.top_k
        cfg.top_k = req.limit
        try:
            result = resources.vector_store.search(req.query)
            return {
                "found": result.found,
                "results": [
                    {"content": r.content, "source": r.source_url, "score": r.score}
                    for r in result.results
                ],
            }
        finally:
            cfg.top_k = old_k

    @r.get("/export")
    async def kb_export():
        """Export all chunks from the knowledge base."""
        docs = resources.vector_store.get_all_documents()
        return {"total": len(docs), "documents": docs}

    @r.delete("/reset")
    async def kb_reset():
        """Wipe the Knowledge Base completely."""
        resources.vector_store.clear()
        if cfg.enable_graph_features:
            try:
                resources.vector_store._neo4j.clear_all_data()
            except Exception:
                pass
        return {"status": "cleared"}
