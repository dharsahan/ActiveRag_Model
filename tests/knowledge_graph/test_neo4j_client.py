import pytest
import sys
import os

# Add the project root to the path to import the client directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from active_rag.knowledge_graph.neo4j_client import Neo4jClient

def test_neo4j_connection():
    client = Neo4jClient("bolt://localhost:7687", "neo4j", "activerag123")
    assert client.is_connected() == True

def test_create_entity():
    client = Neo4jClient("bolt://localhost:7687", "neo4j", "activerag123")
    result = client.create_entity("Person", {"name": "Test Person", "id": "test_1"})
    assert result["id"] == "test_1"