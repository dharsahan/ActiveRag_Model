"""FastAPI REST API for the Active RAG pipeline.

Router-based architecture exposing all features:
- Query (agent/hybrid/ultimate/legacy pipelines)
- Document ingestion (file upload, text, batch, URL)
- Knowledge Base management (stats, search, export, reset)
- Knowledge Graph operations (entity search, paths, neighborhoods, multi-hop)
- NLP Pipeline (entity extraction, relations, classification, sentiment)
- Reasoning & Analytics (multi-hop reasoning, communities, cross-domain, bridges)
- Answer Evaluation (quality scoring)
- System management (health, sessions, performance, cache)
"""

from __future__ import annotations

import os
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

from active_rag.config import Config
from active_rag.dependencies import (
    SessionManager,
    ResourceManager,
    GraphResourceManager,
)

# Import routers
from active_rag.routers import (
    query as query_router,
    ingestion as ingestion_router,
    knowledge_base as kb_router,
    graph as graph_router,
    nlp as nlp_router,
    reasoning as reasoning_router,
    evaluation as evaluation_router,
    system as system_router,
)

logger = logging.getLogger(__name__)


def create_app(config: Config | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Active RAG API",
        description=(
            "Production-ready RAG API with Knowledge Graph, NLP Pipeline, "
            "Multi-hop Reasoning, and Analytics. "
            "Build FAQ bots, legal record systems, research assistants, and more."
        ),
        version="2.0.0",
        openapi_tags=[
            {"name": "Query", "description": "Execute queries against the RAG pipelines"},
            {"name": "Ingestion", "description": "Upload files, text, batch data, or URLs"},
            {"name": "Knowledge Base", "description": "Manage the vector knowledge base"},
            {"name": "Knowledge Graph", "description": "Search entities, explore paths, and multi-hop queries"},
            {"name": "NLP Pipeline", "description": "Entity extraction, relation extraction, classification, sentiment"},
            {"name": "Reasoning & Analytics", "description": "Multi-hop reasoning, community detection, cross-domain discovery"},
            {"name": "Evaluation", "description": "Answer quality evaluation"},
            {"name": "System", "description": "Health checks, sessions, performance, cache management"},
            {"name": "Config", "description": "View and update system configuration"},
        ],
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Initialize shared resources ---
    cfg = config or Config()
    sessions = SessionManager(cfg)
    resources = ResourceManager(cfg)
    graph_resources = GraphResourceManager(cfg)

    # --- Register all routers ---

    # 1. Query
    query_r = query_router.router
    query_router.register(query_r, sessions, resources)
    app.include_router(query_r)

    # 2. Ingestion
    ingest_r = ingestion_router.router
    ingestion_router.register(ingest_r, resources)
    app.include_router(ingest_r)

    # 3. Knowledge Base
    kb_r = kb_router.router
    kb_router.register(kb_r, resources, cfg)
    app.include_router(kb_r)

    # 4. Knowledge Graph
    graph_r = graph_router.router
    graph_router.register(graph_r, graph_resources)
    app.include_router(graph_r)

    # 5. NLP Pipeline
    nlp_r = nlp_router.router
    nlp_router.register(nlp_r, graph_resources)
    app.include_router(nlp_r)

    # 6. Reasoning & Analytics
    reasoning_r = reasoning_router.router
    reasoning_router.register(reasoning_r, graph_resources)
    app.include_router(reasoning_r)

    # 7. Evaluation
    eval_r = evaluation_router.router
    evaluation_router.register(eval_r, graph_resources)
    app.include_router(eval_r)

    # 8. System
    system_r = system_router.router
    system_router.register(system_r, sessions, resources, graph_resources, cfg)
    app.include_router(system_r)

    # --- Config endpoints (kept inline as they're simple) ---

    @app.get("/api/v1/config", tags=["Config"])
    async def get_config():
        """View current configuration."""
        return {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")}

    @app.patch("/api/v1/config", tags=["Config"])
    async def update_config(
        top_k: int | None = None,
        confidence_threshold: float | None = None,
        max_search_results: int | None = None,
        enable_graph_features: bool | None = None,
    ):
        """Update system configuration dynamically."""
        updates = {
            "top_k": top_k, "confidence_threshold": confidence_threshold,
            "max_search_results": max_search_results, "enable_graph_features": enable_graph_features,
        }
        for key, value in updates.items():
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)
        return {"status": "updated", "new_config": await get_config()}

    # --- Backward-compatible redirects ---
    # Old paths redirect to new /api/v1/ paths

    @app.post("/query", include_in_schema=False)
    async def _redirect_query():
        return RedirectResponse(url="/api/v1/query", status_code=307)

    @app.post("/query/stream", include_in_schema=False)
    async def _redirect_query_stream():
        return RedirectResponse(url="/api/v1/query/stream", status_code=307)

    @app.post("/ingest/upload", include_in_schema=False)
    async def _redirect_ingest_upload():
        return RedirectResponse(url="/api/v1/ingest/upload", status_code=307)

    @app.post("/ingest/text", include_in_schema=False)
    async def _redirect_ingest_text():
        return RedirectResponse(url="/api/v1/ingest/text", status_code=307)

    @app.get("/kb/stats", include_in_schema=False)
    async def _redirect_kb_stats():
        return RedirectResponse(url="/api/v1/kb/stats", status_code=307)

    @app.post("/kb/search", include_in_schema=False)
    async def _redirect_kb_search():
        return RedirectResponse(url="/api/v1/kb/search", status_code=307)

    @app.get("/kb/export", include_in_schema=False)
    async def _redirect_kb_export():
        return RedirectResponse(url="/api/v1/kb/export", status_code=307)

    @app.delete("/kb/reset", include_in_schema=False)
    async def _redirect_kb_reset():
        return RedirectResponse(url="/api/v1/kb/reset", status_code=307)

    @app.get("/system/health", include_in_schema=False)
    async def _redirect_health():
        return RedirectResponse(url="/api/v1/system/health", status_code=307)

    @app.get("/system/memory/{session_id}", include_in_schema=False)
    async def _redirect_memory(session_id: str):
        return RedirectResponse(url=f"/api/v1/system/memory/{session_id}", status_code=307)

    @app.delete("/system/memory/{session_id}", include_in_schema=False)
    async def _redirect_clear_memory(session_id: str):
        return RedirectResponse(url=f"/api/v1/system/memory/{session_id}", status_code=307)

    @app.get("/config", include_in_schema=False)
    async def _redirect_config():
        return RedirectResponse(url="/api/v1/config", status_code=307)

    @app.patch("/config", include_in_schema=False)
    async def _redirect_config_update():
        return RedirectResponse(url="/api/v1/config", status_code=307)

    # --- Static Files ---
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def serve_index():
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {
            "message": "Active RAG API v2.0 is running.",
            "docs": "/docs",
            "endpoints": {
                "query": "/api/v1/query",
                "ingest": "/api/v1/ingest/",
                "knowledge_base": "/api/v1/kb/",
                "graph": "/api/v1/graph/",
                "nlp": "/api/v1/nlp/",
                "reasoning": "/api/v1/reasoning/",
                "evaluate": "/api/v1/evaluate",
                "system": "/api/v1/system/",
                "config": "/api/v1/config",
            },
        }

    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
