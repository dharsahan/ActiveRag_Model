"""Active RAG Pipeline – main orchestrator.

Implements the full flow from the architecture diagram:

  User Query
      │
      ▼
  AI Model Confidence Check
      ├── High Confidence ──► Generate Answer Directly ──► Final Answer
      │
      ▼  Don't Know / Hallucination Risk
  Check Vector Memory / RAG
      ├── Data Found ──► Retrieve Context & Citations
      │                       ──► Generate Answer with Citations
      │
      ▼  Data Missing
  Search Data Online
      │
      ▼
  Scrape & Extract Content
      │
      ▼
  Update Vector DB (Content + Source URL)
      │
      └──► (closed loop) back to Retrieve Context & Citations
                              ──► Generate Answer with Citations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from active_rag.answer_generator import Answer, AnswerGenerator
from active_rag.confidence_checker import ConfidenceChecker, ConfidenceResult
from active_rag.config import Config
from active_rag.vector_store import VectorStore
from active_rag.web_search import WebSearcher

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Full result returned by the pipeline, including diagnostics."""

    answer: Answer
    confidence: ConfidenceResult | None = None
    path: str = ""  # "direct" | "rag_memory" | "rag_web"
    web_pages_indexed: int = 0
    diagnostics: dict = field(default_factory=dict)


class ActiveRAGPipeline:
    """Orchestrates the Refined Active RAG Architecture."""

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or Config()
        self._confidence_checker = ConfidenceChecker(self._config)
        self._vector_store = VectorStore(self._config)
        self._web_searcher = WebSearcher(self._config)
        self._answer_generator = AnswerGenerator(self._config)

    def run(self, query: str) -> PipelineResult:
        """Execute the full Active RAG pipeline for the given *query*."""
        # ── Step 1: AI Model Confidence Check ────────────────────────
        confidence = self._confidence_checker.check(query)
        logger.info(
            "Confidence: %.2f (%s) – %s",
            confidence.confidence,
            "HIGH" if confidence.is_high_confidence else "LOW",
            confidence.reasoning,
        )

        if confidence.is_high_confidence:
            # ── High Confidence → Generate Answer Directly ───────────
            answer = self._answer_generator.generate_direct(query)
            return PipelineResult(
                answer=answer,
                confidence=confidence,
                path="direct",
            )

        # ── Step 2: Check Vector Memory / RAG ────────────────────────
        vector_result = self._vector_store.search(query)

        if vector_result.found:
            # ── Data Found → Retrieve Context & Citations → Answer ───
            answer = self._answer_generator.generate_with_citations(
                query, vector_result.results
            )
            return PipelineResult(
                answer=answer,
                confidence=confidence,
                path="rag_memory",
            )

        # ── Step 3: Data Missing → Search Data Online ────────────────
        logger.info("Vector memory miss – searching the web…")
        pages = self._web_searcher.search_and_scrape(query)

        # ── Step 4: Scrape & Extract Content  (done inside search_and_scrape)

        # ── Step 5: Update Vector DB (Content + Source URL) ──────────
        if pages:
            self._vector_store.add_documents(
                contents=[p.content for p in pages],
                source_urls=[p.url for p in pages],
            )
            logger.info("Indexed %d new pages into vector store.", len(pages))

        # ── Closed Loop: Retrieve Context & Citations → Answer ───────
        vector_result = self._vector_store.search(query)
        answer = self._answer_generator.generate_with_citations(
            query, vector_result.results
        )
        return PipelineResult(
            answer=answer,
            confidence=confidence,
            path="rag_web",
            web_pages_indexed=len(pages),
        )
