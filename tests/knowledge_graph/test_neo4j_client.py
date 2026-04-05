import pytest
import sys
import os

# Add the project root to the path to import the client directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from active_rag.knowledge_graph.neo4j_client import Neo4jClient

@pytest.fixture
def neo4j_credentials():
    """Provide Neo4j connection credentials from environment variables."""
    return {
        'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        'username': os.getenv('NEO4J_USERNAME', 'neo4j'),
        'password': os.getenv('NEO4J_PASSWORD', 'activerag123')
    }

@pytest.fixture
def neo4j_client(neo4j_credentials):
    """Create a Neo4j client instance for testing."""
    client = Neo4jClient(**neo4j_credentials)
    yield client
    client.close()

def test_neo4j_connection(neo4j_client):
    """Test that we can connect to Neo4j database."""
    # Skip test if Neo4j is not available
    if not neo4j_client.is_connected():
        pytest.skip("Neo4j is not running or not accessible")

    assert neo4j_client.is_connected() == True

def test_create_entity(neo4j_client):
    """Test creating an entity in Neo4j with proper cleanup."""
    # Skip test if Neo4j is not available
    if not neo4j_client.is_connected():
        pytest.skip("Neo4j is not running or not accessible")

    test_id = "test_create_entity_1"

    try:
        # Create test entity
        result = neo4j_client.create_entity("TestPerson", {"name": "Test Person", "id": test_id})
        assert result["id"] == test_id
        assert result["name"] == "Test Person"

    finally:
        # Cleanup - remove test entity regardless of test outcome
        try:
            with neo4j_client._driver.session() as session:
                session.run("MATCH (n:TestPerson {id: $test_id}) DELETE n", test_id=test_id)
        except Exception:
            # If cleanup fails, log it but don't fail the test
            pass

def test_create_entity_invalid_label(neo4j_client):
    """Test that invalid label names raise ValueError."""
    # Skip test if Neo4j is not available
    if not neo4j_client.is_connected():
        pytest.skip("Neo4j is not running or not accessible")

    # Test various invalid label names
    invalid_labels = [
        "",  # empty
        "123Invalid",  # starts with number
        "Invalid-Label",  # contains dash
        "Invalid Label",  # contains space
        "Invalid.Label",  # contains dot
        None,  # None value
    ]

    for invalid_label in invalid_labels:
        with pytest.raises(ValueError):
            neo4j_client.create_entity(invalid_label, {"test": "value"})

def test_create_entity_invalid_properties(neo4j_client):
    """Test that invalid properties raise ValueError."""
    # Skip test if Neo4j is not available
    if not neo4j_client.is_connected():
        pytest.skip("Neo4j is not running or not accessible")

    # Test invalid properties (not a dict)
    with pytest.raises(ValueError):
        neo4j_client.create_entity("ValidLabel", "invalid_properties")

    with pytest.raises(ValueError):
        neo4j_client.create_entity("ValidLabel", None)