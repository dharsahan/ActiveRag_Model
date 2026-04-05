"""Tests for query classifier."""

from active_rag.routing.query_classifier import (
    QueryClassifier,
    QueryComplexity,
    QueryIntent,
)


class TestQueryClassifier:
    """Test suite for QueryClassifier."""

    def setup_method(self):
        self.classifier = QueryClassifier()

    # --- Intent detection ---

    def test_semantic_query(self):
        """Pure semantic query → SEMANTIC intent."""
        result = self.classifier.classify("What is quantum computing?")
        assert result.intent == QueryIntent.SEMANTIC

    def test_relational_query(self):
        """Relational query with 'who manages' → RELATIONAL intent."""
        result = self.classifier.classify("Who manages the ML team?")
        assert result.intent == QueryIntent.RELATIONAL

    def test_hybrid_query(self):
        """Query with both semantic and relational signals → HYBRID intent."""
        result = self.classifier.classify(
            "Explain the relationship between Albert Einstein and Princeton University"
        )
        assert result.intent == QueryIntent.HYBRID

    def test_relational_keywords_detected(self):
        """Relational keywords are captured in the result."""
        result = self.classifier.classify("Who authored this paper?")
        assert "who" in result.relational_signals or "authored" in result.relational_signals

    # --- Complexity detection ---

    def test_simple_complexity(self):
        """Single-hop query → SIMPLE."""
        result = self.classifier.classify("What is Python?")
        assert result.complexity == QueryComplexity.SIMPLE

    def test_multi_hop_complexity(self):
        """Multi-hop query with chained relationships → MULTI_HOP."""
        result = self.classifier.classify(
            "Who are the students of Einstein that collaborated with Bohr?"
        )
        assert result.complexity == QueryComplexity.MULTI_HOP

    # --- Entity detection ---

    def test_entity_detection(self):
        """Proper names are detected as entities."""
        result = self.classifier.classify("Albert Einstein worked at Princeton University")
        assert "Albert Einstein" in result.detected_entities

    def test_org_detection(self):
        """Known organization acronyms are detected."""
        result = self.classifier.classify("MIT researchers published a paper on AI")
        assert "MIT" in result.detected_entities

    # --- Confidence ---

    def test_confidence_range(self):
        """Confidence should be between 0 and 1."""
        result = self.classifier.classify("random question about stuff")
        assert 0.0 <= result.confidence <= 1.0

    def test_clear_signals_higher_confidence(self):
        """Clear signals should produce higher confidence."""
        vague = self.classifier.classify("stuff")
        clear = self.classifier.classify("What is the relationship between Alice and Bob?")
        assert clear.confidence >= vague.confidence
