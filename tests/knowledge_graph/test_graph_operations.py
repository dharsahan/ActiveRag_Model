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

def test_entity_search(graph_ops):
    """Test searching entities by name pattern"""
    # Search for entities containing "Einstein" - may be empty but should work
    entities = graph_ops.search_entities_by_name("Einstein", ["Person"])
    assert isinstance(entities, list)

    # Test search without entity type filter
    all_entities = graph_ops.search_entities_by_name("test")
    assert isinstance(all_entities, list)

def test_graph_statistics(graph_ops):
    """Test graph statistics functionality"""
    stats = graph_ops.get_graph_stats()
    assert "total_nodes" in stats
    assert "node_types" in stats
    assert "total_relationships" in stats
    assert "relationship_types" in stats
    assert isinstance(stats["total_nodes"], int)
    assert isinstance(stats["node_types"], dict)
    assert isinstance(stats["total_relationships"], int)
    assert isinstance(stats["relationship_types"], dict)

def test_entity_neighborhood(graph_ops):
    """Test entity neighborhood exploration"""
    # Get neighborhood of an entity - should return empty list for non-existent entity
    neighborhood = graph_ops.get_entity_neighborhood("person_einstein", radius=2)
    assert isinstance(neighborhood, list)

def test_document_entities(graph_ops):
    """Test getting entities from a document"""
    # Test getting entities from a document - should return empty list for non-existent doc
    entities = graph_ops.get_document_entities("doc_1")
    assert isinstance(entities, list)

def test_concept_relationships(graph_ops):
    """Test exploring concept relationships"""
    # Test concept relationship exploration - should return empty list for non-existent concept
    concepts = graph_ops.explore_concept_relationships("concept_quantum", depth=2)
    assert isinstance(concepts, list)

def test_query_builder_functionality():
    """Test CypherQueryBuilder methods directly"""
    from active_rag.knowledge_graph.query_builder import CypherQueryBuilder

    # Test query building methods
    query = CypherQueryBuilder.find_related_entities("test_id", ["RELATED_TO"], depth=1)
    assert isinstance(query, str)
    assert "MATCH" in query
    assert "test_id" not in query  # Should use parameter

    path_query = CypherQueryBuilder.find_paths("start", "end", max_depth=2)
    assert isinstance(path_query, str)
    assert "shortestPath" in path_query

def test_reasoning_path_creation(graph_ops):
    """Test reasoning path creation helper"""
    nodes = [
        {"id": "node1", "name": "Node 1"},
        {"id": "node2", "name": "Node 2"},
        {"id": "node3", "name": "Node 3"}
    ]
    relationships = ["RELATED_TO", "CONNECTED_WITH"]

    path = graph_ops._create_reasoning_path(nodes, relationships)
    assert isinstance(path, str)
    assert "Node 1" in path
    assert "Node 2" in path
    assert "Node 3" in path

def test_entity_relevance_filtering(graph_ops):
    """Test entity relevance filtering"""
    entities = [
        {"id": "e1", "name": "Einstein", "distance": 1},
        {"id": "e2", "name": "Quantum Physics", "distance": 2},
        {"id": "e3", "name": "Unrelated", "distance": 3}
    ]

    query_text = "Einstein quantum research"
    target_entities = [{"properties": {"name": "Einstein"}}]

    relevant = graph_ops._filter_relevant_entities(entities, query_text, target_entities)
    assert isinstance(relevant, list)
    # Should prioritize Einstein (exact match) and Quantum (query match)
    if relevant:
        assert relevant[0]["id"] == "e1"  # Einstein should be first

def test_graph_operations_initialization():
    """Test that GraphOperations initializes correctly"""
    config = Config()
    client = Neo4jClient(config.neo4j_uri, config.neo4j_username, config.neo4j_password)
    graph_ops = GraphOperations(client)

    assert graph_ops.client is not None
    assert graph_ops.query_builder is not None
    assert graph_ops.entity_extractor is not None