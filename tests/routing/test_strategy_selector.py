"""Tests for strategy selector."""

from unittest.mock import MagicMock

from active_rag.routing.query_classifier import (
    ClassificationResult,
    QueryComplexity,
    QueryIntent,
)
from active_rag.routing.strategy_selector import (
    RetrievalStrategy,
    StrategySelector,
)


def _make_classification(
    intent: QueryIntent = QueryIntent.SEMANTIC,
    complexity: QueryComplexity = QueryComplexity.SIMPLE,
) -> ClassificationResult:
    return ClassificationResult(
        intent=intent,
        complexity=complexity,
        detected_entities=[],
        relational_signals=[],
        confidence=0.8,
    )


class TestStrategySelector:
    """Test suite for StrategySelector."""

    def _make_selector(self, graph_enabled: bool = True) -> StrategySelector:
        config = MagicMock()
        config.enable_graph_features = graph_enabled
        return StrategySelector(config)

    # --- Graph enabled ---

    def test_semantic_simple_routes_to_vector(self):
        selector = self._make_selector(graph_enabled=True)
        decision = selector.select(_make_classification(QueryIntent.SEMANTIC, QueryComplexity.SIMPLE))
        assert decision.strategy == RetrievalStrategy.VECTOR

    def test_relational_simple_routes_to_graph(self):
        selector = self._make_selector(graph_enabled=True)
        decision = selector.select(_make_classification(QueryIntent.RELATIONAL, QueryComplexity.SIMPLE))
        assert decision.strategy == RetrievalStrategy.GRAPH

    def test_hybrid_intent_routes_to_hybrid(self):
        selector = self._make_selector(graph_enabled=True)
        decision = selector.select(_make_classification(QueryIntent.HYBRID, QueryComplexity.SIMPLE))
        assert decision.strategy == RetrievalStrategy.HYBRID

    def test_multi_hop_routes_to_hybrid(self):
        selector = self._make_selector(graph_enabled=True)
        decision = selector.select(_make_classification(QueryIntent.SEMANTIC, QueryComplexity.MULTI_HOP))
        assert decision.strategy == RetrievalStrategy.HYBRID

    # --- Graph disabled --- (fallback to VECTOR for all)

    def test_graph_disabled_always_vector(self):
        selector = self._make_selector(graph_enabled=False)
        for intent in QueryIntent:
            for complexity in QueryComplexity:
                decision = selector.select(_make_classification(intent, complexity))
                assert decision.strategy == RetrievalStrategy.VECTOR

    def test_decision_has_reason(self):
        selector = self._make_selector(graph_enabled=True)
        decision = selector.select(_make_classification())
        assert len(decision.reason) > 0
