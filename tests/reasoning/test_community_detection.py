"""Tests for community detection."""

from unittest.mock import MagicMock
import pytest

from active_rag.reasoning.community_detection import CommunityDetector, Community


class TestCommunityDetector:
    """Tests for label propagation community detection."""

    def test_label_propagation_simple(self):
        """Two disconnected groups should form two communities."""
        adjacency = {
            "a": {"b"},
            "b": {"a", "c"},
            "c": {"b"},
            "x": {"y"},
            "y": {"x"},
        }
        labels = CommunityDetector._label_propagation(adjacency, max_iterations=10)
        # a, b, c should share a label
        assert labels["a"] == labels["b"] == labels["c"]
        # x, y should share a label
        assert labels["x"] == labels["y"]
        # The two groups should have different labels
        assert labels["a"] != labels["x"]

    def test_single_node_community(self):
        adjacency = {"lone": set()}
        labels = CommunityDetector._label_propagation(adjacency)
        assert "lone" in labels

    def test_detect_communities_with_mock_graph(self):
        graph_ops = MagicMock()
        graph_ops.search_entities_by_name.return_value = [
            {"id": "e1", "name": "Alice", "labels": ["Person"]},
            {"id": "e2", "name": "Bob", "labels": ["Person"]},
            {"id": "e3", "name": "MIT", "labels": ["Organization"]},
        ]
        graph_ops.find_related_entities.side_effect = lambda eid, *a, **kw: {
            "e1": [{"id": "e2", "labels": ["Person"]}],
            "e2": [{"id": "e1", "labels": ["Person"]}],
            "e3": [],
        }.get(eid, [])

        detector = CommunityDetector()
        communities = detector.detect_communities(graph_ops)

        assert len(communities) >= 1
        assert all(isinstance(c, Community) for c in communities)
        # At least one community should have > 1 member
        sizes = [c.size for c in communities]
        assert max(sizes) >= 2

    def test_empty_graph(self):
        graph_ops = MagicMock()
        graph_ops.search_entities_by_name.return_value = []

        detector = CommunityDetector()
        communities = detector.detect_communities(graph_ops)
        assert communities == []

    def test_community_properties(self):
        community = Community(
            community_id=1,
            entities=[
                {"name": "Alice", "labels": ["Person"]},
                {"name": "Bob", "labels": ["Person"]},
            ],
            dominant_label="Person",
            size=2,
        )
        assert community.entity_names == ["Alice", "Bob"]
        assert community.dominant_label == "Person"
