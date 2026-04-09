"""System router — /api/v1/system endpoints.

Health checks, session management, performance monitoring, and cache control.
"""

from __future__ import annotations

import time
import logging
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from active_rag.config import Config
from active_rag.dependencies import SessionManager, ResourceManager, GraphResourceManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/system", tags=["System"])


# --- Models ---

class CacheInvalidateRequest(BaseModel):
    query_type: Optional[str] = None


def register(
    r: APIRouter,
    sessions: SessionManager,
    resources: ResourceManager,
    graph_resources: GraphResourceManager,
    cfg: Config,
):
    """Register system endpoints."""

    @r.get("/health")
    async def system_health():
        """Check status of all external dependencies."""
        health = {"status": "ok", "timestamp": time.time()}

        # Check LLM
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(cfg.ollama_base_url, timeout=2.0)
                health["llm"] = "connected" if resp.status_code < 500 else "error"
        except Exception:
            health["llm"] = "disconnected"

        # Check Neo4j
        try:
            resources.vector_store.count()
            health["neo4j"] = "connected"
        except Exception:
            health["neo4j"] = "disconnected"

        # Check graph features
        health["graph_features_enabled"] = cfg.enable_graph_features
        if cfg.enable_graph_features and graph_resources.graph_ops:
            health["graph"] = "connected"
        elif cfg.enable_graph_features:
            health["graph"] = "disconnected"

        return health

    @r.get("/sessions")
    async def list_sessions():
        """List all active sessions."""
        session_list = sessions.list_sessions()
        return {
            "count": len(session_list),
            "sessions": session_list,
        }

    @r.get("/memory/{session_id}")
    async def get_memory(session_id: str):
        """Dump memory for a specific session."""
        memory = sessions.get_memory(session_id)
        return {
            "session_id": session_id,
            "message_count": len(memory.messages),
            "summary": memory.summary,
            "history": [m.__dict__ for m in memory.messages],
        }

    @r.delete("/memory/{session_id}")
    async def clear_session(session_id: str):
        """Clear memory for a specific session."""
        sessions.clear_session(session_id)
        return {"status": f"session {session_id} cleared"}

    @r.get("/performance")
    async def performance_report():
        """Get query performance report.

        Returns execution time stats, cache hit rates, and per-query-type breakdown.
        """
        monitor = graph_resources.query_monitor
        if monitor is None:
            return {"message": "Query monitor is not available", "total_queries": 0}

        return monitor.get_performance_report()

    @r.get("/cache/stats")
    async def cache_stats():
        """Get graph cache performance metrics."""
        cache = graph_resources.graph_cache
        if cache is None:
            return {"message": "Graph cache is not available"}

        return {
            "size": cache.size,
            "metrics": cache.metrics.to_dict(),
        }

    @r.post("/cache/invalidate")
    async def cache_invalidate(req: CacheInvalidateRequest):
        """Invalidate graph cache entries.

        If query_type is provided, only entries for that type are cleared.
        If omitted, the entire cache is cleared.
        """
        cache = graph_resources.graph_cache
        if cache is None:
            raise HTTPException(status_code=503, detail="Graph cache is not available")

        count = cache.invalidate(req.query_type)
        return {
            "status": "invalidated",
            "entries_cleared": count,
            "query_type": req.query_type or "all",
        }
