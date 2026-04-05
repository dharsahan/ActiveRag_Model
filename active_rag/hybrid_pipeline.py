"""Hybrid Vector-Graph RAG Pipeline.

Orchestrates intelligent routing between vector, graph, and hybrid
retrieval strategies, then combines results for answer generation.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Generator

from openai import OpenAI

from active_rag.config import Config
from active_rag.pipeline import PipelineResult
from active_rag.answer_generator import Answer
from active_rag.memory import ConversationMemory
from active_rag.vector_store import VectorStore
from active_rag.routing.query_classifier import QueryClassifier
from active_rag.routing.strategy_selector import StrategySelector, RetrievalStrategy
from active_rag.routing.result_combiner import ResultCombiner, SourcedChunk, CombinedResult
from active_rag.reasoning.reasoning_engine import ReasoningEngine
from active_rag.reasoning.explainability import ExplainabilityFormatter

logger = logging.getLogger(__name__)


class HybridRAGPipeline:
    """Pipeline with intelligent routing between vector and graph retrieval."""

    def __init__(
        self,
        config: Config | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._config = config or Config()
        self._progress_callback = progress_callback or (lambda _: None)

        # LLM client
        self._client = OpenAI(
            base_url=self._config.ollama_base_url,
            api_key=self._config.api_key,
        )

        # Routing components
        self._classifier = QueryClassifier()
        self._selector = StrategySelector(self._config)
        self._combiner = ResultCombiner()

        # Retrieval backends
        self._vector_store = VectorStore(self._config)
        self._graph_ops = None
        self._memory = ConversationMemory(self._config)

        # Reasoning & explainability (Phase 3)
        self._reasoning_engine = None
        self._explainability = ExplainabilityFormatter()

        # Lazy-init graph ops
        if self._config.enable_graph_features:
            try:
                from active_rag.knowledge_graph.neo4j_client import Neo4jClient
                from active_rag.knowledge_graph.graph_operations import GraphOperations

                client = Neo4jClient(
                    self._config.neo4j_uri,
                    self._config.neo4j_username,
                    self._config.neo4j_password,
                )
                self._graph_ops = GraphOperations(client)
                self._progress_callback("Graph backend connected.")

                # Init reasoning engine with graph ops
                try:
                    from active_rag.nlp_pipeline.entity_extractor import EntityExtractor
                    extractor = EntityExtractor()
                except Exception:
                    extractor = None
                self._reasoning_engine = ReasoningEngine(self._graph_ops, extractor)
            except Exception as e:
                logger.warning(f"Graph backend unavailable: {e}")
                self._graph_ops = None

    def run(self, query: str, explain: bool = False) -> PipelineResult:
        """Execute hybrid retrieval and generate an answer.

        Args:
            query: The user's question.
            explain: If True, attach reasoning explanations to diagnostics.

        Returns:
            PipelineResult with answer, path info, and citations.
        """
        # 1. Classify
        self._progress_callback("Classifying query...")
        classification = self._classifier.classify(query)

        # 2. Select strategy
        decision = self._selector.select(classification)
        strategy = decision.strategy
        self._progress_callback(f"Strategy: {strategy.value} — {decision.reason}")

        # 3. Retrieve
        vector_chunks: list[SourcedChunk] = []
        graph_chunks: list[SourcedChunk] = []

        if strategy in (RetrievalStrategy.VECTOR, RetrievalStrategy.HYBRID):
            vector_chunks = self._retrieve_vector(query)

        if strategy in (RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID) and self._graph_ops:
            graph_chunks = self._retrieve_graph(query, classification)

        # 4. Combine
        combined = self._combiner.combine(
            vector_results=vector_chunks,
            graph_results=graph_chunks,
            top_k=self._config.top_k,
        )

        # 5. Generate answer
        self._progress_callback("Generating answer...")
        answer_text = self._generate_answer(query, combined)
        citations = self._extract_citations(vector_chunks, graph_chunks)

        # Update memory
        self._memory.add_user_message(query)
        self._memory.add_assistant_message(answer_text)

        path_label = f"hybrid_{combined.strategy_used}"
        diagnostics: dict = {}

        # 6. Explainability (Phase 3)
        if explain:
            self._progress_callback("Generating explanation...")
            reasoning_result = None
            if self._reasoning_engine:
                reasoning_result = self._reasoning_engine.reason(
                    query, max_hops=self._config.max_graph_hops
                )
            explanation = self._explainability.format_reasoning(
                reasoning=reasoning_result,
                combined=combined,
                strategy=strategy.value,
            )
            diagnostics["explanation"] = {
                "reasoning_text": explanation.reasoning_text,
                "confidence_explanation": explanation.confidence_explanation,
                "path_visualization": explanation.path_visualization,
                "source_breakdown": explanation.source_breakdown,
                "top_paths": explanation.top_paths,
            }

        return PipelineResult(
            answer=Answer(text=answer_text, citations=citations, source=path_label),
            path=path_label,
            diagnostics=diagnostics,
        )

    def run_stream(self, query: str, explain: bool = False) -> Generator[str | PipelineResult, None, None]:
        """Execute hybrid retrieval with streaming response.

        Args:
            query: The user's question.
            explain: If True, attach reasoning explanations to diagnostics.

        Yields:
            str: Progress tokens and metadata
            PipelineResult: Final result
        """
        try:
            # 1. Classify
            self._progress_callback("Classifying query...")
            classification = self._classifier.classify(query)

            # 2. Select strategy
            decision = self._selector.select(classification)
            strategy = decision.strategy
            self._progress_callback(f"Strategy: {strategy.value} — {decision.reason}")

            # Emit strategy path
            yield f"__path__:hybrid_{strategy.value}"

            # 3. Retrieve
            vector_chunks: list[SourcedChunk] = []
            graph_chunks: list[SourcedChunk] = []

            if strategy in (RetrievalStrategy.VECTOR, RetrievalStrategy.HYBRID):
                yield "Searching vector store..."
                vector_chunks = self._retrieve_vector(query)

            if strategy in (RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID) and self._graph_ops:
                yield "Querying knowledge graph..."
                graph_chunks = self._retrieve_graph(query, classification)

            # 4. Combine
            yield "Combining results..."
            combined = self._combiner.combine(
                vector_results=vector_chunks,
                graph_results=graph_chunks,
                top_k=self._config.top_k,
            )

            # 5. Generate answer with streaming
            yield "Generating answer..."
            context = combined.context_text
            messages = self._memory.get_context_messages()

            system_content = (
                "You are a helpful assistant with access to a hybrid knowledge system. "
                "Use the provided context to answer the user's question accurately. "
                "If a reasoning path is provided, explain how the entities are connected. "
                "Format your response clearly using markdown."
            )

            if not messages or messages[0].get("role") != "system":
                messages.insert(0, {"role": "system", "content": system_content})
            else:
                messages[0] = {"role": "system", "content": system_content}

            user_content = query
            if context.strip():
                user_content = f"Context:\n{context}\n\nQuestion: {query}"

            messages.append({"role": "user", "content": user_content})

            # Stream the LLM response
            answer_text = ""
            try:
                response = self._client.chat.completions.create(
                    model=self._config.model_name,
                    messages=messages,
                    temperature=0.2,
                    stream=True,
                )

                for chunk in response:
                    if chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        answer_text += token
                        yield token

            except Exception as e:
                logger.error(f"Streaming LLM failed, falling back to sync: {e}")
                # Fallback to sync generation
                try:
                    response = self._client.chat.completions.create(
                        model=self._config.model_name,
                        messages=messages,
                        temperature=0.2,
                    )
                    answer_text = response.choices[0].message.content or ""
                    yield answer_text
                except Exception as sync_e:
                    answer_text = f"Error generating answer: {sync_e}"
                    yield answer_text

            # Extract citations
            citations = self._extract_citations(vector_chunks, graph_chunks)

            # Update memory
            self._memory.add_user_message(query)
            self._memory.add_assistant_message(answer_text)

            path_label = f"hybrid_{combined.strategy_used}"
            diagnostics: dict = {}

            # 6. Explainability (if requested)
            if explain:
                yield "\nGenerating explanation..."
                reasoning_result = None
                if self._reasoning_engine:
                    reasoning_result = self._reasoning_engine.reason(
                        query, max_hops=self._config.max_graph_hops
                    )
                explanation = self._explainability.format_reasoning(
                    reasoning=reasoning_result,
                    combined=combined,
                    strategy=strategy.value,
                )
                diagnostics["explanation"] = {
                    "reasoning_text": explanation.reasoning_text,
                    "confidence_explanation": explanation.confidence_explanation,
                    "path_visualization": explanation.path_visualization,
                    "source_breakdown": explanation.source_breakdown,
                    "top_paths": explanation.top_paths,
                }

            # Yield final result
            yield PipelineResult(
                answer=Answer(text=answer_text, citations=citations, source=path_label),
                path=path_label,
                diagnostics=diagnostics,
            )

        except Exception as e:
            logger.error(f"Streaming hybrid pipeline failed: {e}")
            # Emergency fallback - call sync version
            yield from [self.run(query, explain)]

    # --- Private retrieval helpers ---

    def _retrieve_vector(self, query: str) -> list[SourcedChunk]:
        """Search ChromaDB vector store."""
        self._progress_callback("Searching vector store...")
        result = self._vector_store.search(query)
        if not result.found:
            return []

        chunks: list[SourcedChunk] = []
        for r in result.results:
            chunks.append(SourcedChunk(
                content=r.content,
                source="vector",
                score=max(0.0, 1.0 - r.score),  # ChromaDB distance → similarity
                metadata={"source_url": r.source_url},
            ))
        return chunks

    def _retrieve_graph(self, query: str, classification) -> list[SourcedChunk]:
        """Query the knowledge graph."""
        if not self._graph_ops:
            return []

        self._progress_callback("Querying knowledge graph...")
        try:
            max_hops = min(self._config.max_graph_hops, 5)
            result = self._graph_ops.multi_hop_query(query, max_hops=max_hops)

            chunks: list[SourcedChunk] = []
            # Convert graph entities into context chunks
            for entity in result.get("entities", [])[:5]:
                name = entity.get("name", "Unknown")
                labels = entity.get("labels", [])
                relevance = entity.get("relevance_score", 0.5)
                chunks.append(SourcedChunk(
                    content=f"Entity: {name} (Type: {', '.join(labels)})",
                    source="graph",
                    score=relevance,
                    metadata={"entity_id": entity.get("id", "")},
                ))

            # Convert paths into context chunks
            for path in result.get("paths", [])[:3]:
                reasoning = path.get("reasoning_path", "")
                if reasoning:
                    chunks.append(SourcedChunk(
                        content=f"Reasoning path: {reasoning}",
                        source="graph",
                        score=0.8,
                        reasoning_path=reasoning,
                    ))

            return chunks
        except Exception as e:
            logger.warning(f"Graph retrieval failed: {e}")
            return []

    def _generate_answer(self, query: str, combined: CombinedResult) -> str:
        """Generate an answer using the LLM with combined context."""
        context = combined.context_text

        messages = self._memory.get_context_messages()

        system_content = (
            "You are a helpful assistant with access to a hybrid knowledge system. "
            "Use the provided context to answer the user's question accurately. "
            "If a reasoning path is provided, explain how the entities are connected. "
            "Format your response clearly using markdown."
        )

        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": system_content})
        else:
            messages[0] = {"role": "system", "content": system_content}

        user_content = query
        if context.strip():
            user_content = f"Context:\n{context}\n\nQuestion: {query}"

        messages.append({"role": "user", "content": user_content})

        try:
            response = self._client.chat.completions.create(
                model=self._config.model_name,
                messages=messages,
                temperature=0.2,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM answer generation failed: {e}")
            return f"Error generating answer: {e}"

    @staticmethod
    def _extract_citations(
        vector_chunks: list[SourcedChunk],
        graph_chunks: list[SourcedChunk],
    ) -> list[str]:
        """Extract unique citation URLs from retrieved chunks."""
        urls: list[str] = []
        for chunk in vector_chunks:
            url = chunk.metadata.get("source_url", "")
            if url and url not in urls:
                urls.append(url)
        return urls

    def clear_memory(self) -> None:
        """Clear conversation memory."""
        self._memory.clear()

    def clear_cache(self) -> None:
        """Compatibility stub."""
        pass
