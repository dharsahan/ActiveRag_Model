"""Tests for result combiner."""

from active_rag.routing.result_combiner import (
    CombinedResult,
    ResultCombiner,
    SourcedChunk,
)


def _chunk(content: str, source: str = "vector", score: float = 0.8) -> SourcedChunk:
    return SourcedChunk(content=content, source=source, score=score)


class TestResultCombiner:
    """Test suite for ResultCombiner."""

    def setup_method(self):
        self.combiner = ResultCombiner(vector_weight=0.6, graph_weight=0.4)

    def test_vector_only(self):
        """Vector-only results set strategy to 'vector'."""
        result = self.combiner.combine(
            vector_results=[_chunk("doc A", "vector", 0.9)],
        )
        assert result.strategy_used == "vector"
        assert result.vector_count == 1
        assert result.graph_count == 0
        assert len(result.chunks) == 1

    def test_graph_only(self):
        """Graph-only results set strategy to 'graph'."""
        result = self.combiner.combine(
            graph_results=[_chunk("entity path", "graph", 0.85)],
        )
        assert result.strategy_used == "graph"
        assert result.graph_count == 1
        assert result.vector_count == 0

    def test_hybrid_results(self):
        """Mixed results set strategy to 'hybrid'."""
        result = self.combiner.combine(
            vector_results=[_chunk("vec doc", "vector", 0.9)],
            graph_results=[_chunk("graph doc", "graph", 0.8)],
        )
        assert result.strategy_used == "hybrid"
        assert result.vector_count == 1
        assert result.graph_count == 1

    def test_deduplication(self):
        """Duplicate content is deduplicated."""
        same = "The exact same content repeated in both sources."
        result = self.combiner.combine(
            vector_results=[_chunk(same, "vector", 0.9)],
            graph_results=[_chunk(same, "graph", 0.8)],
        )
        assert len(result.chunks) == 1

    def test_ranking_by_weighted_score(self):
        """Chunks are ranked by weighted score, descending."""
        result = self.combiner.combine(
            vector_results=[
                _chunk("low vec", "vector", 0.3),
                _chunk("high vec", "vector", 0.95),
            ],
            graph_results=[
                _chunk("high graph", "graph", 1.0),
            ],
        )
        scores = [c.score for c in result.chunks]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limit(self):
        """Only top_k results are returned."""
        vector_results = [_chunk(f"doc {i}", "vector", 0.5) for i in range(10)]
        result = self.combiner.combine(vector_results=vector_results, top_k=3)
        assert len(result.chunks) <= 3

    def test_context_text_generation(self):
        """context_text concatenates chunks with source headers."""
        result = self.combiner.combine(
            vector_results=[_chunk("Hello world", "vector", 0.9)],
        )
        text = result.context_text
        assert "Source 1: vector" in text
        assert "Hello world" in text

    def test_empty_inputs(self):
        """Empty inputs return empty result."""
        result = self.combiner.combine()
        assert len(result.chunks) == 0
        assert result.strategy_used == "vector"
