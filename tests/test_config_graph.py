# tests/test_config_graph.py
import pytest
import os
from active_rag.config import Config

def test_neo4j_config():
    """Test that Neo4j configuration attributes exist with correct defaults"""
    config = Config()
    assert hasattr(config, 'neo4j_uri')
    assert hasattr(config, 'neo4j_username')
    assert hasattr(config, 'neo4j_password')
    assert config.neo4j_uri == "bolt://localhost:7687"
    assert config.neo4j_username == "neo4j"
    assert config.neo4j_password == "activerag123"

def test_graph_features_enabled():
    """Test that graph feature configuration attributes exist with correct defaults"""
    config = Config()
    assert hasattr(config, 'enable_graph_features')
    assert hasattr(config, 'max_graph_hops')
    assert config.enable_graph_features == True
    assert config.max_graph_hops == 3

def test_nlp_pipeline_config():
    """Test that NLP pipeline configuration attributes exist with correct defaults"""
    config = Config()
    assert hasattr(config, 'spacy_model')
    assert hasattr(config, 'enable_relation_extraction')
    assert config.spacy_model == "en_core_web_sm"
    assert config.enable_relation_extraction == True

def test_neo4j_config_from_env(monkeypatch):
    """Test Neo4j configuration from environment variables"""
    # Set environment variables
    monkeypatch.setenv('NEO4J_URI', 'bolt://test:7687')
    monkeypatch.setenv('NEO4J_USERNAME', 'testuser')
    monkeypatch.setenv('NEO4J_PASSWORD', 'testpass')

    config = Config()
    assert config.neo4j_uri == 'bolt://test:7687'
    assert config.neo4j_username == 'testuser'
    assert config.neo4j_password == 'testpass'

def test_graph_features_from_env(monkeypatch):
    """Test graph feature toggles from environment"""
    # Test disabled graph features
    monkeypatch.setenv('ENABLE_GRAPH_FEATURES', 'false')
    monkeypatch.setenv('MAX_GRAPH_HOPS', '5')

    config = Config()
    assert config.enable_graph_features == False
    assert config.max_graph_hops == 5

    # Test enabled graph features with different case
    monkeypatch.setenv('ENABLE_GRAPH_FEATURES', 'TRUE')
    config2 = Config()
    assert config2.enable_graph_features == True

def test_nlp_config_from_env(monkeypatch):
    """Test NLP pipeline configuration from environment"""
    monkeypatch.setenv('SPACY_MODEL', 'en_core_web_md')
    monkeypatch.setenv('ENABLE_RELATION_EXTRACTION', 'false')

    config = Config()
    assert config.spacy_model == 'en_core_web_md'
    assert config.enable_relation_extraction == False

def test_boolean_env_parsing(monkeypatch):
    """Test that boolean environment variables are parsed correctly"""
    # Test various true values
    for true_val in ['true', 'True', 'TRUE', '1', 'yes', 'Yes']:
        monkeypatch.setenv('ENABLE_GRAPH_FEATURES', true_val)
        config = Config()
        # Only 'true' (lowercase) should evaluate to True
        if true_val.lower() == 'true':
            assert config.enable_graph_features == True
        else:
            assert config.enable_graph_features == False

    # Test various false values
    for false_val in ['false', 'False', 'FALSE', '0', 'no', 'No']:
        monkeypatch.setenv('ENABLE_GRAPH_FEATURES', false_val)
        config = Config()
        assert config.enable_graph_features == False

def test_integer_env_parsing(monkeypatch):
    """Test that integer environment variables are parsed correctly"""
    monkeypatch.setenv('MAX_GRAPH_HOPS', '10')
    config = Config()
    assert config.max_graph_hops == 10
    assert isinstance(config.max_graph_hops, int)