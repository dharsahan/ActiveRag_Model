import pytest
from active_rag.knowledge_graph.graph_operations import GraphOperations
from active_rag.knowledge_graph.neo4j_client import Neo4jClient
from active_rag.config import Config

@pytest.fixture
def graph_ops():
    """Fixture to create GraphOperations instance for tests"""
    config = Config()
    client = Neo4jClient(config.neo4j_uri, config.neo4j_username, config.neo4j_password)
    return GraphOperations(client)

def test_find_related_entities(graph_ops):
    """Test finding entities related to a starting entity"""
    # Test with a non-existent entity - should return empty list
    related = graph_ops.find_related_entities("person_einstein", ["AFFILIATED_WITH"], depth=1)
    assert isinstance(related, list)

    # Test with different depths
    related_multi = graph_ops.find_related_entities("person_einstein", ["AFFILIATED_WITH"], depth=2)
    assert isinstance(related_multi, list)

def test_find_path_between_entities(graph_ops):
    """Test pathfinding between two entities"""
    # Test pathfinding with non-existent entities - should return empty list
    paths = graph_ops.find_paths("person_einstein", "concept_quantum", max_depth=3)
    assert isinstance(paths, list)

def test_multi_hop_query(graph_ops):
    """Test multi-hop reasoning query"""
    # Test multi-hop reasoning query
    results = graph_ops.multi_hop_query("Who collaborated with Einstein?", max_hops=2)
    assert "entities" in results
    assert "paths" in results
    assert "reasoning" in results
    assert isinstance(results["entities"], list)
    assert isinstance(results["paths"], list)