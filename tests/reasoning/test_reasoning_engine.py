"""Tests for the multi-hop reasoning engine."""

from unittest.mock import MagicMock
import pytest

from active_rag.reasoning.reasoning_engine import (
    ReasoningEngine,
    ReasoningResult,
    ReasoningPath,
    PathRanker,
    SubgraphExtractor,
    Subgraph,
)


class TestPathRanker:
    """Tests for path scoring and ranking."""

    def setup_method(self):
        self.ranker = PathRanker()

    def test_shorter_paths_score_higher(self):
        paths = [
            {"nodes": [{"name": "A"}, {"name": "B"}, {"name": "C"}], "relationship_types": ["R1", "R2"], "length": 2},
            {"nodes": [{"name": "A"}, {"name": "B"}], "relationship_types": ["R1"], "length": 1},
        ]
        ranked = self.ranker.rank_paths(paths, [])
        assert ranked[0].length < ranked[1].length

    def test_entity_match_bonus(self):
        paths = [
            {"nodes": [{"name": "Alice"}, {"name": "Bob"}], "relationship_types": ["KNOWS"], "length": 1},
            {"nodes": [{"name": "X"}, {"name": "Y"}], "relationship_types": ["KNOWS"], "length": 1},
        ]
        ranked = self.ranker.rank_paths(paths, ["Alice"])
        # Path with Alice should score higher due to entity match bonus
        assert ranked[0].start_entity == "Alice"

    def test_reasoning_text_generation(self):
        paths = [
            {
                "nodes": [{"name": "Einstein"}, {"name": "Princeton"}],
                "relationship_types": ["AFFILIATED_WITH"],
                "length": 1,
                "reasoning_path": "",
            }
        ]
        ranked = self.ranker.rank_paths(paths, [])
        assert "Einstein" in ranked[0].reasoning_text
        assert "AFFILIATED_WITH" in ranked[0].reasoning_text
        assert "Princeton" in ranked[0].reasoning_text

    def test_score_clamped_to_0_1(self):
        paths = [
            {"nodes": [{"name": "A"}] * 20, "relationship_types": [], "length": 19},
        ]
        ranked = self.ranker.rank_paths(paths, [])
        assert 0.0 <= ranked[0].score <= 1.0

    def test_empty_paths(self):
        ranked = self.ranker.rank_paths([], ["Alice"])
        assert ranked == []


class TestSubgraphExtractor:
    """Tests for subgraph extraction."""

    def test_extract_builds_subgraph(self):
        graph_ops = MagicMock()
        graph_ops.get_entity_neighborhood.return_value = [
            {"id": "n1", "name": "Node1", "labels": ["Person"]},
            {"id": "n2", "name": "Node2", "labels": ["Organization"]},
        ]
        graph_ops.find_related_entities.return_value = [
            {"id": "n2", "name": "Node2", "relationship_type": "WORKS_FOR"},
        ]

        extractor = SubgraphExtractor()
        subgraph = extractor.extract(graph_ops, ["seed1"], max_radius=2)

        assert subgraph.node_count >= 2
        assert subgraph.edge_count >= 1
        assert "seed1" in subgraph.seed_entity_ids

    def test_extract_handles_errors(self):
        graph_ops = MagicMock()
        graph_ops.get_entity_neighborhood.side_effect = Exception("Connection failed")
        graph_ops.find_related_entities.side_effect = Exception("Connection failed")

        extractor = SubgraphExtractor()
        subgraph = extractor.extract(graph_ops, ["bad_id"])

        assert subgraph.node_count == 0
        assert subgraph.edge_count == 0


class TestReasoningEngine:
    """Tests for the full reasoning engine."""

    def test_empty_result_when_no_graph(self):
        engine = ReasoningEngine(graph_ops=None, entity_extractor=None)
        result = engine.reason("What is Python?")
        assert isinstance(result, ReasoningResult)
        assert not result.has_results
        assert result.confidence == 0.0

    def test_empty_result_when_no_entities_extracted(self):
        extractor = MagicMock()
        extractor.extract_entities.return_value = []
        engine = ReasoningEngine(graph_ops=MagicMock(), entity_extractor=extractor)
        result = engine.reason("Vague query")
        assert not result.has_results

    def test_reason_with_entities_and_graph(self):
        extractor = MagicMock()
        extractor.extract_entities.return_value = [
            {"label": "Person", "properties": {"id": "p1", "name": "Alice"}},
            {"label": "Organization", "properties": {"id": "o1", "name": "MIT"}},
        ]

        graph_ops = MagicMock()
        graph_ops.get_entity_neighborhood.return_value = [
            {"id": "p1", "name": "Alice", "labels": ["Person"]},
            {"id": "o1", "name": "MIT", "labels": ["Organization"]},
        ]
        graph_ops.find_related_entities.return_value = [
            {"id": "o1", "relationship_type": "WORKS_FOR"},
        ]
        graph_ops.find_paths.return_value = [
            {
                "nodes": [{"name": "Alice"}, {"name": "MIT"}],
                "relationship_types": ["AFFILIATED_WITH"],
                "length": 1,
                "reasoning_path": "Alice → MIT",
            }
        ]
        graph_ops.multi_hop_query.return_value = {"entities": [], "paths": []}

        engine = ReasoningEngine(graph_ops=graph_ops, entity_extractor=extractor)
        result = engine.reason("Where does Alice work?", max_hops=2)

        assert result.has_results
        assert result.confidence > 0.0
        assert len(result.ranked_paths) >= 1
        assert result.query == "Where does Alice work?"

    def test_reasoning_result_properties(self):
        result = ReasoningResult(
            query="test",
            seed_entities=[],
            subgraph=Subgraph(nodes=[], edges=[], seed_entity_ids=[]),
            ranked_paths=[],
            supporting_entities=[],
            reasoning_summary="No results",
            confidence=0.0,
        )
        assert not result.has_results

    def test_reasoning_path_start_end_entity(self):
        path = ReasoningPath(
            nodes=[{"name": "Start"}, {"name": "Middle"}, {"name": "End"}],
            relationships=["R1", "R2"],
            reasoning_text="test",
            score=0.5,
            length=2,
        )
        assert path.start_entity == "Start"
        assert path.end_entity == "End"
