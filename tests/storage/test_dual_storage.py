"""Comprehensive tests for dual storage integration."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import hashlib
from pathlib import Path

from active_rag.config import Config
from active_rag.schemas.entities import ContentDomain


class TestDualStorageManager:
    """Test dual storage manager functionality"""

    def test_dual_storage_initialization(self):
        """Test that dual storage manager initializes correctly"""
        config = Config()

        # Import should work when class exists
        from active_rag.storage.dual_storage_manager import DualStorageManager

        storage = DualStorageManager(config)
        assert storage.chroma_client is not None
        assert hasattr(storage, 'neo4j_client')
        assert hasattr(storage, 'entity_extractor')
        assert hasattr(storage, 'document_classifier')

    def test_dual_storage_with_graph_disabled(self):
        """Test dual storage when graph features are disabled"""
        config = Config()
        config.enable_graph_features = False

        from active_rag.storage.dual_storage_manager import DualStorageManager

        storage = DualStorageManager(config)
        assert storage.chroma_client is not None
        assert storage.neo4j_client is None
        assert storage.schema_manager is None

    @patch('active_rag.storage.dual_storage_manager.chromadb')
    @patch('active_rag.knowledge_graph.neo4j_client.Neo4jClient')
    @patch('active_rag.knowledge_graph.schema_manager.SchemaManager')
    def test_store_document_with_entities(self, mock_schema_manager, mock_neo4j, mock_chromadb):
        """Test storing document with entity extraction"""
        # Setup mocks
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value.get_or_create_collection.return_value = mock_collection

        mock_neo4j_instance = Mock()
        mock_neo4j.return_value = mock_neo4j_instance

        mock_schema_instance = Mock()
        mock_schema_manager.return_value = mock_schema_instance
        mock_schema_instance.validate_entity.return_value = True

        config = Config()
        config.enable_graph_features = True

        from active_rag.storage.dual_storage_manager import DualStorageManager

        storage = DualStorageManager(config)

        doc_data = {
            "title": "Einstein Research",
            "content": "Albert Einstein worked at Princeton University on quantum mechanics.",
            "url": "http://test.com/einstein",
            "domain": ContentDomain.RESEARCH
        }

        # Mock entity extraction
        mock_entities = [
            {
                "label": "Person",
                "properties": {"id": "albert_einstein", "name": "Albert Einstein"}
            },
            {
                "label": "Organization",
                "properties": {"id": "princeton_university", "name": "Princeton University"}
            }
        ]

        with patch.object(storage.entity_extractor, 'extract_entities', return_value=mock_entities):
            with patch.object(storage, '_create_document_node', return_value={"id": "doc123"}):
                with patch.object(storage, '_create_or_merge_entity', return_value={"id": "entity123"}):
                    with patch.object(storage, '_create_document_entity_relationship', return_value=True):
                        result = storage.store_document(doc_data)

        assert result["chroma_stored"] == True
        assert result["graph_stored"] == True
        assert len(result["entities_extracted"]) == 2
        assert len(result["relationships_created"]) == 2

        # Verify ChromaDB was called
        mock_collection.add.assert_called_once()

    def test_generate_doc_id_with_url(self):
        """Test document ID generation with URL"""
        config = Config()
        from active_rag.storage.dual_storage_manager import DualStorageManager

        storage = DualStorageManager(config)

        doc_data = {"url": "http://example.com/doc1", "content": "test"}
        doc_id = storage._generate_doc_id(doc_data)

        expected_id = hashlib.md5("http://example.com/doc1".encode()).hexdigest()
        assert doc_id == expected_id

    def test_generate_doc_id_without_url(self):
        """Test document ID generation without URL"""
        config = Config()
        from active_rag.storage.dual_storage_manager import DualStorageManager

        storage = DualStorageManager(config)

        doc_data = {"title": "Test Doc", "content": "This is test content"}
        doc_id = storage._generate_doc_id(doc_data)

        combined = "Test Doc_This is test content"
        expected_id = hashlib.md5(combined.encode()).hexdigest()
        assert doc_id == expected_id

    @patch('active_rag.storage.dual_storage_manager.chromadb')
    @patch('active_rag.knowledge_graph.neo4j_client.Neo4jClient')
    @patch('active_rag.knowledge_graph.schema_manager.SchemaManager')
    def test_get_document_entities(self, mock_schema_manager, mock_neo4j, mock_chromadb):
        """Test retrieving entities for a document"""
        # Setup mocks
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value.get_or_create_collection.return_value = mock_collection

        mock_neo4j_instance = Mock()
        mock_neo4j.return_value = mock_neo4j_instance

        mock_schema_instance = Mock()
        mock_schema_manager.return_value = mock_schema_instance

        # Mock session and result
        mock_session = Mock()
        mock_result = Mock()
        mock_record = Mock()
        mock_record.__getitem__ = Mock(side_effect=lambda k: {
            "e": {"id": "einstein", "name": "Albert Einstein"},
            "entity_labels": ["Person"]
        }[k])
        mock_result.__iter__ = Mock(return_value=iter([mock_record]))
        mock_session.run.return_value = mock_result
        mock_neo4j_instance._driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_neo4j_instance._driver.session.return_value.__exit__ = Mock(return_value=None)

        config = Config()
        config.enable_graph_features = True

        from active_rag.storage.dual_storage_manager import DualStorageManager

        storage = DualStorageManager(config)

        entities = storage.get_document_entities("test_doc_id")

        assert len(entities) == 1
        assert entities[0]["id"] == "einstein"
        assert entities[0]["name"] == "Albert Einstein"
        assert entities[0]["labels"] == ["Person"]


class TestDualStorageIntegration:
    """Integration tests for dual storage with realistic scenarios"""

    @patch('active_rag.storage.dual_storage_manager.chromadb')
    @patch('active_rag.knowledge_graph.neo4j_client.Neo4jClient')
    @patch('active_rag.knowledge_graph.schema_manager.SchemaManager')
    def test_research_document_full_pipeline(self, mock_schema_manager, mock_neo4j, mock_chromadb):
        """Test complete pipeline for research document"""
        # Setup comprehensive mocks
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value.get_or_create_collection.return_value = mock_collection

        mock_neo4j_instance = Mock()
        mock_neo4j.return_value = mock_neo4j_instance

        mock_schema_instance = Mock()
        mock_schema_manager.return_value = mock_schema_instance
        mock_schema_instance.validate_entity.return_value = True

        config = Config()
        config.enable_graph_features = True

        from active_rag.storage.dual_storage_manager import DualStorageManager

        storage = DualStorageManager(config)

        research_doc = {
            "title": "Quantum Computing Research",
            "content": "Marie Curie researched radioactivity at the Radium Institute in Paris. Her work contributed to quantum mechanics.",
            "url": "http://example.com/curie-research",
            "domain": ContentDomain.RESEARCH
        }

        # Mock realistic entity extraction
        expected_entities = [
            {
                "label": "Person",
                "properties": {"id": "marie_curie", "name": "Marie Curie"}
            },
            {
                "label": "Organization",
                "properties": {"id": "radium_institute", "name": "Radium Institute", "location": "Paris"}
            },
            {
                "label": "Concept",
                "properties": {"id": "radioactivity", "name": "radioactivity"}
            }
        ]

        with patch.object(storage.entity_extractor, 'extract_entities', return_value=expected_entities):
            with patch.object(storage, '_create_document_node', return_value={"id": "research_doc_123"}):
                with patch.object(storage, '_create_or_merge_entity', return_value={"id": "entity_456"}):
                    with patch.object(storage, '_create_document_entity_relationship', return_value=True):
                        result = storage.store_document(research_doc)

        # Verify comprehensive storage
        assert result["chroma_stored"] == True
        assert result["graph_stored"] == True
        assert len(result["entities_extracted"]) == 3
        assert len(result["relationships_created"]) == 3

        # Verify ChromaDB call with correct metadata
        call_args = mock_collection.add.call_args
        assert call_args[1]["metadatas"][0]["title"] == "Quantum Computing Research"
        assert call_args[1]["metadatas"][0]["domain"] == "research"

    @patch('active_rag.storage.dual_storage_manager.chromadb')
    def test_chroma_only_fallback(self, mock_chromadb):
        """Test graceful fallback when Neo4j is unavailable"""
        mock_collection = Mock()
        mock_chromadb.PersistentClient.return_value.get_or_create_collection.return_value = mock_collection

        config = Config()
        config.enable_graph_features = False  # Disable graph features

        from active_rag.storage.dual_storage_manager import DualStorageManager

        storage = DualStorageManager(config)

        doc_data = {
            "title": "Simple Document",
            "content": "This is a simple document without graph processing.",
            "domain": ContentDomain.MIXED_WEB
        }

        result = storage.store_document(doc_data)

        # Should store in ChromaDB only
        assert result["chroma_stored"] == True
        assert result["graph_stored"] == False
        assert len(result["entities_extracted"]) == 0
        assert len(result["relationships_created"]) == 0

        # Verify ChromaDB was called
        mock_collection.add.assert_called_once()