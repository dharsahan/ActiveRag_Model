"""Query router — /api/v1/query endpoints."""

from __future__ import annotations

import json
import asyncio
import logging
from typing import List, Optional, Literal

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from active_rag.dependencies import (
    SessionManager, ResourceManager, verify_api_key,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Query"])


# --- Models ---

class QueryRequest(BaseModel):
    query: str
    session_id: str = Field(default="default", description="Unique session ID for conversation history")
    pipeline_type: Literal["agent", "hybrid", "ultimate", "legacy"] = "agent"
    explain: bool = False


class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
    path: str
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    web_pages_indexed: int = 0
    from_cache: bool = False
    session_id: str


def register(r: APIRouter, sessions: SessionManager, resources: ResourceManager):
    """Register query endpoints on the given router."""

    @r.post("/query", response_model=QueryResponse)
    async def query(req: QueryRequest):
        pipeline = resources.get_pipeline(req.pipeline_type)
        memory = sessions.get_memory(req.session_id)

        if hasattr(pipeline, "run_async"):
            result = await pipeline.run_async(req.query, memory=memory)
        else:
            result = pipeline.run(req.query, memory=memory)

        return QueryResponse(
            answer=result.answer.text,
            citations=result.answer.citations,
            path=result.path,
            confidence=result.confidence.confidence if result.confidence else None,
            reasoning=result.confidence.reasoning if result.confidence else None,
            web_pages_indexed=result.web_pages_indexed,
            from_cache=result.from_cache,
            session_id=req.session_id,
        )

    @r.post("/query/stream")
    async def query_stream(req: QueryRequest):
        """Stream response for the given session and pipeline."""
        pipeline = resources.get_pipeline(req.pipeline_type)
        memory = sessions.get_memory(req.session_id)

        async def event_generator():
            try:
                if req.pipeline_type == "agent":
                    async for event in pipeline.run_stream(req.query, memory=memory):
                        yield json.dumps(event) + "\n"
                        await asyncio.sleep(0.01)
                else:
                    def run_sync_stream():
                        return pipeline.run_stream(req.query, memory=memory)

                    loop = asyncio.get_event_loop()
                    gen = await loop.run_in_executor(None, run_sync_stream)

                    for event in gen:
                        if isinstance(event, str):
                            yield json.dumps({"type": "token", "content": event}) + "\n"
                        else:
                            yield json.dumps({
                                "type": "result",
                                "answer": event.answer.text,
                                "path": event.path,
                                "citations": event.answer.citations,
                            }) + "\n"
                        await asyncio.sleep(0.01)

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield json.dumps({"type": "error", "content": str(e)}) + "\n"

        return StreamingResponse(event_generator(), media_type="application/x-ndjson")
