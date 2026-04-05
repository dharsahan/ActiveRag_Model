"""Tests for the hybrid RAG pipeline."""

from unittest.mock import MagicMock, patch
import pytest

from active_rag.routing.query_classifier import QueryIntent, QueryComplexity
from active_rag.routing.strategy_selector import RetrievalStrategy


class TestHybridPipelineRouting:
    """Test hybrid pipeline routes queries correctly via the classifier + selector."""

    def test_semantic_query_routes_to_vector(self):
        """A simple semantic query should use VECTOR strategy."""
        from active_rag.routing.query_classifier import QueryClassifier
        from active_rag.routing.strategy_selector import StrategySelector

        config = MagicMock()
        config.enable_graph_features = True

        classifier = QueryClassifier()
        selector = StrategySelector(config)

        result = classifier.classify("What is quantum computing?")
        decision = selector.select(result)
        assert decision.strategy == RetrievalStrategy.VECTOR

    def test_relational_query_routes_to_graph(self):
        """A relational query should use GRAPH strategy."""
        from active_rag.routing.query_classifier import QueryClassifier
        from active_rag.routing.strategy_selector import StrategySelector

        config = MagicMock()
        config.enable_graph_features = True

        classifier = QueryClassifier()
        selector = StrategySelector(config)

        result = classifier.classify("Who manages the ML team?")
        decision = selector.select(result)
        assert decision.strategy in (RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID)

    def test_graph_disabled_always_vector(self):
        """With graph disabled, all queries route to VECTOR."""
        from active_rag.routing.query_classifier import QueryClassifier
        from active_rag.routing.strategy_selector import StrategySelector

        config = MagicMock()
        config.enable_graph_features = False

        classifier = QueryClassifier()
        selector = StrategySelector(config)

        for query in [
            "What is Python?",
            "Who manages the team?",
            "How is Einstein connected to Princeton?"
        ]:
            result = classifier.classify(query)
            decision = selector.select(result)
            assert decision.strategy == RetrievalStrategy.VECTOR


class TestHybridPipelineIntegration:
    """Test the full hybrid pipeline with mocked backends."""

    @patch("active_rag.hybrid_pipeline.OpenAI")
    @patch("active_rag.hybrid_pipeline.VectorStore")
    def test_pipeline_run_vector_only(self, mock_vs_cls, mock_openai_cls):
        """Pipeline runs with vector-only when graph is disabled."""
        from active_rag.hybrid_pipeline import HybridRAGPipeline

        config = MagicMock()
        config.enable_graph_features = False
        config.ollama_base_url = "http://localhost:11434/v1"
        config.api_key = "test"
        config.model_name = "test"
        config.chroma_persist_dir = "/tmp/test_chroma"
        config.collection_name = "test"
        config.top_k = 3
        config.confidence_threshold = 0.7
        config.max_graph_hops = 3

        # Mock vector store
        mock_vs = MagicMock()
        mock_search_result = MagicMock()
        mock_search_result.found = True
        mock_search_result.results = [
            MagicMock(content="Python is a programming language.", score=0.2, source_url="https://example.com"),
        ]
        mock_vs.search.return_value = mock_search_result
        mock_vs_cls.return_value = mock_vs

        # Mock LLM
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Python is a versatile programming language."
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_cls.return_value = mock_client

        pipeline = HybridRAGPipeline(config)
        result = pipeline.run("What is Python?")

        assert result.answer.text == "Python is a versatile programming language."
        assert "vector" in result.path

    @patch("active_rag.hybrid_pipeline.OpenAI")
    @patch("active_rag.hybrid_pipeline.VectorStore")
    def test_pipeline_clear_memory(self, mock_vs_cls, mock_openai_cls):
        """clear_memory doesn't crash."""
        from active_rag.hybrid_pipeline import HybridRAGPipeline
        config = MagicMock()
        config.enable_graph_features = False
        config.ollama_base_url = "http://localhost:11434/v1"
        config.api_key = "test"
        config.model_name = "test"
        config.chroma_persist_dir = "/tmp/test_chroma2"
        config.collection_name = "test"
        config.top_k = 3
        config.confidence_threshold = 0.7
        config.max_graph_hops = 3

        pipeline = HybridRAGPipeline(config)
        pipeline.clear_memory()  # Should not raise
