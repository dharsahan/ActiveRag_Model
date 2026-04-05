"""Strategy selector for hybrid vector-graph RAG routing.

Takes a ClassificationResult and selects the retrieval strategy:
VECTOR, GRAPH, or HYBRID.  Respects the `enable_graph_features` config toggle.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from active_rag.config import Config
from active_rag.routing.query_classifier import (
    ClassificationResult,
    QueryComplexity,
    QueryIntent,
)


class RetrievalStrategy(Enum):
    """Which retrieval backend(s) to use."""
    VECTOR = "vector"   # ChromaDB similarity search only
    GRAPH = "graph"     # Neo4j graph traversal only
    HYBRID = "hybrid"   # Both, then combine results


@dataclass
class RoutingDecision:
    """Outcome of strategy selection."""
    strategy: RetrievalStrategy
    classification: ClassificationResult
    reason: str


class StrategySelector:
    """Selects retrieval strategy based on query classification and config."""

    def __init__(self, config: Config | None = None) -> None:
        self._config = config or Config()

    @property
    def graph_enabled(self) -> bool:
        return self._config.enable_graph_features

    def select(self, classification: ClassificationResult) -> RoutingDecision:
        """Choose a retrieval strategy.

        Args:
            classification: Output from QueryClassifier.classify().

        Returns:
            RoutingDecision with strategy and reasoning.
        """
        # If graph features are disabled, always fall back to vector
        if not self.graph_enabled:
            return RoutingDecision(
                strategy=RetrievalStrategy.VECTOR,
                classification=classification,
                reason="Graph features disabled — using vector-only retrieval.",
            )

        intent = classification.intent
        complexity = classification.complexity

        # Pure semantic queries → vector
        if intent == QueryIntent.SEMANTIC and complexity == QueryComplexity.SIMPLE:
            return RoutingDecision(
                strategy=RetrievalStrategy.VECTOR,
                classification=classification,
                reason="Simple semantic query — vector similarity is sufficient.",
            )

        # Pure relational queries → graph
        if intent == QueryIntent.RELATIONAL and complexity == QueryComplexity.SIMPLE:
            return RoutingDecision(
                strategy=RetrievalStrategy.GRAPH,
                classification=classification,
                reason="Relational query — routing to knowledge graph.",
            )

        # Multi-hop or hybrid intent → hybrid
        if complexity == QueryComplexity.MULTI_HOP or intent == QueryIntent.HYBRID:
            return RoutingDecision(
                strategy=RetrievalStrategy.HYBRID,
                classification=classification,
                reason="Complex or hybrid query — combining vector + graph retrieval.",
            )

        # Relational multi-hop → hybrid
        if intent == QueryIntent.RELATIONAL and complexity == QueryComplexity.MULTI_HOP:
            return RoutingDecision(
                strategy=RetrievalStrategy.HYBRID,
                classification=classification,
                reason="Multi-hop relational query — needs both traversal and context.",
            )

        # Default fallback → vector (safe default)
        return RoutingDecision(
            strategy=RetrievalStrategy.VECTOR,
            classification=classification,
            reason="Default fallback — using vector retrieval.",
        )
