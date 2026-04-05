"""Tests for the explainability system."""

from unittest.mock import MagicMock
import pytest

from active_rag.reasoning.explainability import (
    ExplainabilityFormatter,
    ExplanationResult,
)
from active_rag.reasoning.reasoning_engine import (
    ReasoningResult,
    ReasoningPath,
    Subgraph,
)
from active_rag.routing.result_combiner import CombinedResult


class TestExplainabilityFormatter:
    """Tests for explanation formatting."""

    def setup_method(self):
        self.formatter = ExplainabilityFormatter()

    def test_format_vector_only(self):
        """Vector-only answers should explain no graph reasoning was used."""
        combined = CombinedResult(
            chunks=[],
            strategy_used="vector",
            vector_count=3,
            graph_count=0,
        )
        result = self.formatter.format_reasoning(
            reasoning=None, combined=combined, strategy="vector"
        )
        assert isinstance(result, ExplanationResult)
        assert "semantic similarity" in result.reasoning_text.lower()
        assert result.strategy_used == "vector"
        assert result.source_breakdown["vector"] == 3
        assert result.source_breakdown["graph"] == 0

    def test_format_with_reasoning_paths(self):
        """Reasoning paths should appear in the explanation."""
        reasoning = ReasoningResult(
            query="test",
            seed_entities=[{"properties": {"id": "p1", "name": "Alice"}}],
            subgraph=Subgraph(nodes=[], edges=[], seed_entity_ids=["p1"]),
            ranked_paths=[
                ReasoningPath(
                    nodes=[{"name": "Alice"}, {"name": "MIT"}],
                    relationships=["AFFILIATED_WITH"],
                    reasoning_text="Alice → MIT",
                    score=0.85,
                    length=1,
                ),
            ],
            supporting_entities=[{"name": "Bob"}],
            reasoning_summary="Found 1 path",
            confidence=0.7,
        )
        combined = CombinedResult(
            chunks=[], strategy_used="hybrid",
            vector_count=2, graph_count=1,
        )
        result = self.formatter.format_reasoning(
            reasoning=reasoning, combined=combined, strategy="hybrid"
        )
        assert "Alice → MIT" in result.reasoning_text
        assert "85%" in result.reasoning_text
        assert "Bob" in result.reasoning_text
        assert len(result.top_paths) == 1

    def test_confidence_explanation_strong(self):
        """High confidence should say 'Strong graph support'."""
        reasoning = ReasoningResult(
            query="test",
            seed_entities=[],
            subgraph=Subgraph(nodes=[], edges=[], seed_entity_ids=[]),
            ranked_paths=[],
            supporting_entities=[],
            reasoning_summary="",
            confidence=0.8,
        )
        result = self.formatter.format_reasoning(
            reasoning=reasoning, combined=None, strategy="graph"
        )
        assert "Strong" in result.confidence_explanation

    def test_confidence_explanation_weak(self):
        """Low confidence should say 'Weak graph support'."""
        reasoning = ReasoningResult(
            query="test",
            seed_entities=[],
            subgraph=Subgraph(nodes=[], edges=[], seed_entity_ids=[]),
            ranked_paths=[],
            supporting_entities=[],
            reasoning_summary="",
            confidence=0.2,
        )
        result = self.formatter.format_reasoning(
            reasoning=reasoning, combined=None, strategy="graph"
        )
        assert "Weak" in result.confidence_explanation

    def test_path_visualization(self):
        """Path diagram should render node names and relationships."""
        reasoning = ReasoningResult(
            query="test",
            seed_entities=[],
            subgraph=Subgraph(nodes=[], edges=[], seed_entity_ids=[]),
            ranked_paths=[
                ReasoningPath(
                    nodes=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
                    relationships=["REL1", "REL2"],
                    reasoning_text="A → B → C",
                    score=0.9,
                    length=2,
                ),
            ],
            supporting_entities=[],
            reasoning_summary="",
            confidence=0.6,
        )
        result = self.formatter.format_reasoning(reasoning=reasoning, strategy="hybrid")
        assert "[A]" in result.path_visualization
        assert "REL1" in result.path_visualization
        assert "[C]" in result.path_visualization

    def test_empty_reasoning_no_paths(self):
        """No reasoning → path visualization says 'No graph paths'."""
        result = self.formatter.format_reasoning(reasoning=None, strategy="vector")
        assert "No graph" in result.path_visualization
