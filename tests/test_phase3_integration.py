"""Integration tests for Phase 3 features."""

from unittest.mock import MagicMock, patch
import pytest

from active_rag.routing.result_combiner import CombinedResult
from active_rag.reasoning.reasoning_engine import ReasoningEngine, ReasoningResult, Subgraph
from active_rag.reasoning.explainability import ExplainabilityFormatter
from active_rag.knowledge_graph.graph_cache import GraphCache
from active_rag.knowledge_graph.query_monitor import QueryMonitor


class TestReasoningExplainabilityIntegration:
    """Test reasoning engine + explainability working together."""

    def test_explain_empty_reasoning(self):
        """Explainability works even when reasoning returns empty."""
        engine = ReasoningEngine(graph_ops=None)
        result = engine.reason("What is Python?")

        formatter = ExplainabilityFormatter()
        explanation = formatter.format_reasoning(
            reasoning=result,
            combined=CombinedResult(
                chunks=[], strategy_used="vector",
                vector_count=2, graph_count=0,
            ),
            strategy="vector",
        )
        assert "semantic similarity" in explanation.reasoning_text.lower()
        assert explanation.source_breakdown["vector"] == 2

    def test_explain_with_graph_reasoning(self):
        """Full pipeline: extract → reason → explain."""
        extractor = MagicMock()
        extractor.extract_entities.return_value = [
            {"label": "Person", "properties": {"id": "p1", "name": "Alice"}},
        ]

        graph_ops = MagicMock()
        graph_ops.get_entity_neighborhood.return_value = [
            {"id": "p1", "name": "Alice", "labels": ["Person"]},
            {"id": "o1", "name": "MIT", "labels": ["Organization"]},
        ]
        graph_ops.find_related_entities.return_value = [
            {"id": "o1", "relationship_type": "WORKS_FOR"},
        ]
        graph_ops.multi_hop_query.return_value = {
            "entities": [],
            "paths": [{
                "nodes": [{"name": "Alice"}, {"name": "MIT"}],
                "relationship_types": ["AFFILIATED_WITH"],
                "length": 1,
                "reasoning_path": "Alice → MIT",
            }],
        }

        engine = ReasoningEngine(graph_ops, extractor)
        result = engine.reason("Where does Alice work?")

        formatter = ExplainabilityFormatter()
        explanation = formatter.format_reasoning(
            reasoning=result,
            combined=CombinedResult(
                chunks=[], strategy_used="hybrid",
                vector_count=1, graph_count=1,
            ),
            strategy="hybrid",
        )

        assert "Alice → MIT" in explanation.reasoning_text
        assert explanation.strategy_used == "hybrid"


class TestCacheMonitorIntegration:
    """Test cache and monitor working together."""

    def test_cached_query_with_monitoring(self):
        cache = GraphCache()
        monitor = QueryMonitor()

        # Cache miss → record
        with monitor.track("neighborhood", cache_hit=False) as metric:
            result = cache.get("neighborhood", entity_id="e1")
            assert result is None

        # Put in cache
        cache.put("neighborhood", {"id": "e1", "name": "Alice"}, entity_id="e1")

        # Cache hit → record
        with monitor.track("neighborhood", cache_hit=True) as metric:
            result = cache.get("neighborhood", entity_id="e1")
            assert result is not None
            metric.result_count = 1

        report = monitor.get_performance_report()
        assert report["total_queries"] == 2
        assert "neighborhood" in report["query_types"]

    def test_monitor_performance_report(self):
        monitor = QueryMonitor()
        monitor.record("multi_hop", duration_ms=150, cache_hit=False, graph_hops=2, result_count=5)
        monitor.record("multi_hop", duration_ms=50, cache_hit=True, graph_hops=2, result_count=3)
        monitor.record("find_paths", duration_ms=200, cache_hit=False, graph_hops=3, result_count=1)

        report = monitor.get_performance_report()
        assert report["total_queries"] == 3
        assert "multi_hop" in report["query_types"]
        assert "find_paths" in report["query_types"]
        assert report["query_types"]["multi_hop"]["count"] == 2

    def test_monitor_clear(self):
        monitor = QueryMonitor()
        monitor.record("test", duration_ms=10)
        monitor.clear()
        report = monitor.get_performance_report()
        assert report["total_queries"] == 0


class TestHybridPipelineExplain:
    """Test hybrid pipeline with explain=True."""

    @patch("active_rag.hybrid_pipeline.OpenAI")
    @patch("active_rag.hybrid_pipeline.VectorStore")
    def test_pipeline_explain_flag(self, mock_vs_cls, mock_openai_cls):
        """Pipeline returns diagnostics with explanation when explain=True."""
        from active_rag.hybrid_pipeline import HybridRAGPipeline

        config = MagicMock()
        config.enable_graph_features = False
        config.ollama_base_url = "http://localhost:11434/v1"
        config.api_key = "test"
        config.model_name = "test"
        config.chroma_persist_dir = "/tmp/test_chroma_explain"
        config.collection_name = "test"
        config.top_k = 3
        config.confidence_threshold = 0.7
        config.max_graph_hops = 2

        mock_vs = MagicMock()
        mock_search_result = MagicMock()
        mock_search_result.found = True
        mock_search_result.results = [
            MagicMock(content="Test content.", score=0.2, source_url="https://example.com"),
        ]
        mock_vs.search.return_value = mock_search_result
        mock_vs_cls.return_value = mock_vs

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test answer."
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        pipeline = HybridRAGPipeline(config)
        result = pipeline.run("What is Python?", explain=True)

        # Should contain explanation in diagnostics
        assert "explanation" in result.diagnostics
        exp = result.diagnostics["explanation"]
        assert "reasoning_text" in exp
        assert "confidence_explanation" in exp
        assert "source_breakdown" in exp

    @patch("active_rag.hybrid_pipeline.OpenAI")
    @patch("active_rag.hybrid_pipeline.VectorStore")
    def test_pipeline_no_explain(self, mock_vs_cls, mock_openai_cls):
        """Pipeline returns empty diagnostics when explain=False (default)."""
        from active_rag.hybrid_pipeline import HybridRAGPipeline

        config = MagicMock()
        config.enable_graph_features = False
        config.ollama_base_url = "http://localhost:11434/v1"
        config.api_key = "test"
        config.model_name = "test"
        config.chroma_persist_dir = "/tmp/test_chroma_noexplain"
        config.collection_name = "test"
        config.top_k = 3
        config.confidence_threshold = 0.7
        config.max_graph_hops = 2

        mock_vs = MagicMock()
        mock_search_result = MagicMock()
        mock_search_result.found = False
        mock_search_result.results = []
        mock_vs.search.return_value = mock_search_result
        mock_vs_cls.return_value = mock_vs

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Answer."
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        pipeline = HybridRAGPipeline(config)
        result = pipeline.run("What is Python?")

        # No explanation when explain=False
        assert result.diagnostics == {}
