"""Active RAG Pipeline – enhanced orchestrator with streaming and caching.

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
from typing import Generator, Callable

from openai import APIConnectionError
from tenacity import RetryError

from active_rag.answer_generator import Answer, AnswerGenerator
from active_rag.confidence_checker import ConfidenceChecker, ConfidenceResult
from active_rag.config import Config
from active_rag.vector_store import VectorStore, VectorSearchResult
from active_rag.web_search import WebSearcher
from active_rag.memory import ConversationMemory
from active_rag.cache import ResponseCache, CachedResponse

logger = logging.getLogger(__name__)

# Keywords that indicate the user wants fresh / recent information.
_TIME_SENSITIVE_KEYWORDS = [
    "current", "latest", "today", "recent", "breaking",
    "now", "this week", "this month", "new ", "news",
    "update", "happening", "right now", "2026", "yesterday",
    "last night", "this morning", "tonight", "price", "search online",
    "search the web", "look up", "lookup", "google", "find online",
]


@dataclass
class PipelineResult:
    """Full result returned by the pipeline, including diagnostics."""

    answer: Answer
    confidence: ConfidenceResult | None = None
    path: str = ""  # "direct" | "rag_memory" | "rag_web" | "cached"
    web_pages_indexed: int = 0
    from_cache: bool = False
    diagnostics: dict = field(default_factory=dict)


class ActiveRAGPipeline:
    """Orchestrates the Refined Active RAG Architecture."""

    def __init__(
        self,
        config: Config | None = None,
        enable_cache: bool = True,
        enable_memory: bool = True,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._config = config or Config()
        self._progress_callback = progress_callback or (lambda x: None)
        
        self._confidence_checker = ConfidenceChecker(self._config)
        self._vector_store = VectorStore(self._config)
        self._web_searcher = WebSearcher(self._config, self._progress_callback)
        self._answer_generator = AnswerGenerator(self._config)
        
        # Optional features
        self._memory = ConversationMemory() if enable_memory else None
        self._cache = ResponseCache(self._config) if enable_cache else None

    @property
    def memory(self) -> ConversationMemory | None:
        """Access conversation memory."""
        return self._memory

    def clear_memory(self) -> None:
        """Clear conversation history."""
        if self._memory:
            self._memory.clear()

    def clear_cache(self) -> None:
        """Clear response cache."""
        if self._cache:
            self._cache.clear()

    @staticmethod
    def _is_time_sensitive(query: str) -> bool:
        """Detect if a query is asking about recent / current events."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in _TIME_SENSITIVE_KEYWORDS)

    def run(self, query: str, use_cache: bool = True) -> PipelineResult:
        """Execute the full Active RAG pipeline for the given *query*."""
        try:
            # Check cache first
            if use_cache and self._cache:
                cached = self._cache.get(query)
                if cached:
                    self._progress_callback("Found cached response!")
                    return PipelineResult(
                        answer=Answer(
                            text=cached.answer_text,
                            citations=cached.answer_citations,
                            source=cached.answer_source,
                        ),
                        confidence=ConfidenceResult(
                            confidence=cached.confidence_score or 0.0,
                            reasoning=cached.confidence_reasoning or "",
                            is_high_confidence=(cached.confidence_score or 0.0) >= self._config.confidence_threshold,
                        ) if cached.confidence_score is not None else None,
                        path=cached.path,
                        from_cache=True,
                    )
            
            result = self._run(query)
            
            # Cache the result
            if use_cache and self._cache and result.path != "error":
                self._cache.set(query, CachedResponse(
                    answer_text=result.answer.text,
                    answer_citations=result.answer.citations,
                    answer_source=result.answer.source,
                    confidence_score=result.confidence.confidence if result.confidence else None,
                    confidence_reasoning=result.confidence.reasoning if result.confidence else None,
                    path=result.path,
                ))
            
            # Update memory
            if self._memory:
                self._memory.add_user_message(query)
                self._memory.add_assistant_message(result.answer.text)
            
            return result
            
        except (APIConnectionError, RetryError) as e:
            logger.error(
                "Cannot connect to LLM at %s. Is the server running? Error: %s",
                self._config.ollama_base_url,
                str(e)[:100],
            )
            return PipelineResult(
                answer=Answer(
                    text=(
                        "Error: Could not connect to the LLM API. "
                        "Please check your connection and try again."
                    ),
                    citations=[],
                    source="error",
                ),
                path="error",
            )

    def run_stream(self, query: str) -> Generator[str | PipelineResult, None, None]:
        """Execute pipeline with streaming response (falls back to sync if streaming fails)."""
        try:
            # Get conversation context
            conversation_context = ""
            if self._memory and self._memory.is_followup_question(query):
                conversation_context = self._memory.get_conversation_summary()
            
            # Step 1: Confidence check
            self._progress_callback("Checking confidence...")
            confidence = self._confidence_checker.check(query)
            
            # Step 1b: Force low confidence for time-sensitive/explicit search queries
            time_sensitive = self._is_time_sensitive(query)
            if time_sensitive and confidence.is_high_confidence:
                confidence.is_high_confidence = False
                confidence.reasoning = "Query is time-sensitive or requested search; overriding to low confidence."
            
            yield f"__confidence__:{confidence.confidence:.2f}"
            
            if confidence.is_high_confidence:
                self._progress_callback("Generating direct answer...")
                yield "__path__:direct"
                
                # Try streaming, fall back to sync if it fails
                try:
                    full_text = ""
                    for token in self._answer_generator.generate_direct_stream(query, conversation_context):
                        full_text += token
                        yield token
                    answer = Answer(text=full_text, citations=[], source="direct")
                except Exception as e:
                    logger.debug("Streaming failed, falling back to sync: %s", e)
                    answer = self._answer_generator.generate_direct(query, conversation_context)
                    yield answer.text
                
            else:
                vector_result = VectorSearchResult(found=False)

                if not time_sensitive:
                    # Only check vector store for non-time-sensitive queries
                    self._progress_callback("Searching memory...")
                    vector_result = self._vector_store.search(query)
                
                if vector_result.found:
                    yield "__path__:rag_memory"
                    self._progress_callback("Generating answer from memory...")
                    
                    try:
                        full_text = ""
                        for token in self._answer_generator.generate_with_citations_stream(
                            query, vector_result.results, conversation_context
                        ):
                            full_text += token
                            yield token
                        citations = [r.source_url for r in vector_result.results if r.source_url]
                        answer = Answer(text=full_text, citations=citations, source="rag")
                    except Exception as e:
                        logger.debug("Streaming failed, falling back to sync: %s", e)
                        answer = self._answer_generator.generate_with_citations(
                            query, vector_result.results, conversation_context
                        )
                        yield answer.text
                    
                else:
                    yield "__path__:rag_web"
                    
                    # Web search
                    pages = self._web_searcher.search_and_scrape(query)
                    
                    if pages:
                        self._vector_store.add_documents(
                            contents=[p.content for p in pages],
                            source_urls=[p.url for p in pages],
                        )
                        yield f"__indexed__:{len(pages)}"
                    
                    # Use return_all=True since we just indexed pages for this query
                    vector_result = self._vector_store.search(query, return_all=True)
                    self._progress_callback("Generating answer...")
                    
                    try:
                        full_text = ""
                        for token in self._answer_generator.generate_with_citations_stream(
                            query, vector_result.results, conversation_context
                        ):
                            full_text += token
                            yield token
                        citations = [r.source_url for r in vector_result.results if r.source_url]
                        answer = Answer(text=full_text, citations=citations, source="rag")
                    except Exception as e:
                        logger.debug("Streaming failed, falling back to sync: %s", e)
                        answer = self._answer_generator.generate_with_citations(
                            query, vector_result.results, conversation_context
                        )
                        yield answer.text
            
            # Update memory
            if self._memory:
                self._memory.add_user_message(query)
                self._memory.add_assistant_message(answer.text)
            
            # Yield final result
            yield PipelineResult(
                answer=answer,
                confidence=confidence,
                path="direct" if confidence.is_high_confidence else (
                    "rag_memory" if vector_result.found else "rag_web"
                ),
            )
            
        except Exception as e:
            logger.exception("Pipeline error")
            yield PipelineResult(
                answer=Answer(text=f"Error: {str(e)}", citations=[], source="error"),
                path="error",
            )

    def _run(self, query: str) -> PipelineResult:
        """Internal pipeline logic."""
        # Get conversation context for follow-ups
        conversation_context = ""
        if self._memory and self._memory.is_followup_question(query):
            conversation_context = self._memory.get_conversation_summary()
            logger.debug("Using conversation context for follow-up question")
        
        # ── Step 1: AI Model Confidence Check ────────────────────────
        self._progress_callback("Checking confidence...")
        confidence = self._confidence_checker.check(query)
        
        # Step 1b: Force low confidence for time-sensitive/explicit search queries
        time_sensitive = self._is_time_sensitive(query)
        if time_sensitive and confidence.is_high_confidence:
            confidence.is_high_confidence = False
            confidence.reasoning = "Query is time-sensitive or requested search; overriding to low confidence."
            
        logger.info(
            "Confidence: %.2f (%s) – %s",
            confidence.confidence,
            "HIGH" if confidence.is_high_confidence else "LOW",
            confidence.reasoning,
        )

        if confidence.is_high_confidence:
            # ── High Confidence → Generate Answer Directly ───────────
            self._progress_callback("Generating direct answer...")
            answer = self._answer_generator.generate_direct(query, conversation_context)
            return PipelineResult(
                answer=answer,
                confidence=confidence,
                path="direct",
            )

        # ── Step 2: Check Vector Memory / RAG ────────────────────────
        vector_result = VectorSearchResult(found=False)

        if not time_sensitive:
            # Only check vector store for non-time-sensitive queries
            self._progress_callback("Searching memory...")
            vector_result = self._vector_store.search(query)

        if vector_result.found:
            # ── Data Found → Retrieve Context & Citations → Answer ───
            self._progress_callback("Generating answer from memory...")
            answer = self._answer_generator.generate_with_citations(
                query, vector_result.results, conversation_context
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
        self._progress_callback("Generating answer...")
        # Use return_all=True since we just indexed pages specifically for this query
        vector_result = self._vector_store.search(query, return_all=True)
        answer = self._answer_generator.generate_with_citations(
            query, vector_result.results, conversation_context
        )
        return PipelineResult(
            answer=answer,
            confidence=confidence,
            path="rag_web",
            web_pages_indexed=len(pages),
        )
