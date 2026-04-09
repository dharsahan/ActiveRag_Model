"""Evaluation router — /api/v1/evaluate endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from active_rag.dependencies import GraphResourceManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Evaluation"])


# --- Models ---

class EvaluateRequest(BaseModel):
    query: str = Field(..., description="The question that was asked")
    answer: str = Field(..., description="The answer to evaluate")


def register(r: APIRouter, graph_resources: GraphResourceManager):
    """Register evaluation endpoints."""

    @r.post("/evaluate")
    async def evaluate_answer(req: EvaluateRequest):
        """Evaluate the quality of an answer for a given question.

        Returns a quality score (0.0-1.0), identified issues,
        improvement suggestions, and whether the answer is acceptable.

        Use cases:
        - QA pipeline quality gates
        - Automated FAQ answer validation
        - Answer ranking in search results
        """
        evaluator = graph_resources.evaluator
        if evaluator is None:
            raise HTTPException(
                status_code=503,
                detail="Answer evaluator is not available. Ensure LLM backend is configured.",
            )

        try:
            result = evaluator.evaluate(req.query, req.answer)
            return {
                "quality": result.quality,
                "is_acceptable": result.is_acceptable,
                "issues": result.issues,
                "suggestion": result.suggestion,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
