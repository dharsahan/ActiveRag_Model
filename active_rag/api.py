"""FastAPI REST API for the Active RAG pipeline."""

from __future__ import annotations

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from active_rag.config import Config
from active_rag.agent import AgenticOrchestrator


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import os
import json
import asyncio

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    path: str
    confidence: float | None = None
    reasoning: str | None = None
    web_pages_indexed: int = 0
    from_cache: bool = False


def create_app(config: Config | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Active RAG API",
        description="Refined Retrieval-Augmented Generation",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    pipeline = AgenticOrchestrator(config or Config())

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResponse)
    async def query(req: QueryRequest):
        result = await pipeline.run_async(req.query)
        return QueryResponse(
            answer=result.answer.text,
            citations=result.answer.citations,
            path=result.path,
            confidence=result.confidence.confidence if result.confidence else None,
            reasoning=result.confidence.reasoning if result.confidence else None,
            web_pages_indexed=result.web_pages_indexed,
            from_cache=result.from_cache,
        )

    @app.post("/query/stream")
    async def query_stream(req: QueryRequest):
        """Stream the agentic response token by token."""
        async def event_generator():
            # Use the streaming method from the orchestrator
            # We need to wrap it to yield JSON strings for the frontend
            try:
                # AgenticOrchestrator.run_stream is now an async generator
                async for event in pipeline.run_stream(req.query):
                    # We send each event as a JSON line
                    yield json.dumps(event) + "\n"
                    # Small sleep to ensure smooth streaming in local dev
                    await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield json.dumps({"type": "error", "content": str(e)}) + "\n"

        return StreamingResponse(event_generator(), media_type="application/x-ndjson")

    @app.post("/clear-memory")
    def clear_memory():
        pipeline.clear_memory()
        return {"status": "cleared"}

    @app.post("/clear-cache")
    def clear_cache():
        pipeline.clear_cache()
        return {"status": "cleared"}

    # --- Hybrid RAG endpoint ---
    hybrid_pipeline = None

    @app.post("/query/hybrid", response_model=QueryResponse)
    async def query_hybrid(req: QueryRequest):
        nonlocal hybrid_pipeline
        if hybrid_pipeline is None:
            from active_rag.hybrid_pipeline import HybridRAGPipeline
            hybrid_pipeline = HybridRAGPipeline(config or Config())
        result = hybrid_pipeline.run(req.query)
        return QueryResponse(
            answer=result.answer.text,
            citations=result.answer.citations,
            path=result.path,
        )

    @app.post("/query/explain")
    async def query_explain(req: QueryRequest):
        """Hybrid query with full reasoning explanation."""
        nonlocal hybrid_pipeline
        if hybrid_pipeline is None:
            from active_rag.hybrid_pipeline import HybridRAGPipeline
            hybrid_pipeline = HybridRAGPipeline(config or Config())
        result = hybrid_pipeline.run(req.query, explain=True)
        explanation = result.diagnostics.get("explanation", {})
        return {
            "answer": result.answer.text,
            "citations": result.answer.citations,
            "path": result.path,
            "explanation": explanation,
        }

    # Mount static files and serve index.html at root
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def serve_index():
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "Active RAG API is running. UI not found in /static."}

    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
