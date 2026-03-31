"""FastAPI REST API for the Active RAG pipeline."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from active_rag.config import Config
from active_rag.pipeline import ActiveRAGPipeline


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

    pipeline = ActiveRAGPipeline(config or Config())

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest):
        result = pipeline.run(req.query)
        return QueryResponse(
            answer=result.answer.text,
            citations=result.answer.citations,
            path=result.path,
            confidence=result.confidence.confidence if result.confidence else None,
            reasoning=result.confidence.reasoning if result.confidence else None,
            web_pages_indexed=result.web_pages_indexed,
            from_cache=result.from_cache,
        )

    @app.post("/clear-memory")
    def clear_memory():
        pipeline.clear_memory()
        return {"status": "cleared"}

    @app.post("/clear-cache")
    def clear_cache():
        pipeline.clear_cache()
        return {"status": "cleared"}

    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
