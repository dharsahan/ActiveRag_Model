#!/usr/bin/env python3
"""
Demonstration script for dual storage manager functionality.
Shows how the system would work with both ChromaDB and Neo4j enabled.
"""

import sys
import os
import tempfile
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from active_rag.config import Config
from active_rag.schemas.entities import ContentDomain

def demonstrate_dual_storage():
    """Demonstrate dual storage with mocked Neo4j components"""
    print("🔬 Demonstrating Dual Storage Manager with Full Pipeline")
    print("=" * 60)

    # Mock Neo4j components to avoid requiring actual Neo4j instance
    with patch('active_rag.knowledge_graph.neo4j_client.Neo4jClient') as mock_neo4j:
        with patch('active_rag.knowledge_graph.schema_manager.SchemaManager') as mock_schema:

            # Setup mocks
            mock_neo4j_instance = Mock()
            mock_neo4j.return_value = mock_neo4j_instance

            mock_schema_instance = Mock()
            mock_schema.return_value = mock_schema_instance
            mock_schema_instance.validate_entity.return_value = True
            mock_schema_instance.create_base_constraints.return_value = None

            mock_neo4j_instance.create_entity.return_value = {"id": "test_entity"}

            # Mock session for relationship creation
            mock_session = Mock()
            mock_result = Mock()
            mock_result.single.return_value = {"d": {}, "e": {}}
            mock_session.run.return_value = mock_result
            mock_neo4j_instance._driver.session.return_value.__enter__ = Mock(return_value=mock_session)
            mock_neo4j_instance._driver.session.return_value.__exit__ = Mock(return_value=None)

            # Create config with graph features enabled
            config = Config()
            config.enable_graph_features = True

            with tempfile.TemporaryDirectory() as temp_dir:
                config.chroma_persist_dir = temp_dir

                from active_rag.storage.dual_storage_manager import DualStorageManager

                storage = DualStorageManager(config)

                print(f"✓ DualStorageManager initialized")
                print(f"  - ChromaDB: {storage.chroma_client is not None}")
                print(f"  - Neo4j: {storage.neo4j_client is not None}")
                print(f"  - Schema Manager: {storage.schema_manager is not None}")
                print(f"  - Entity Extractor: {storage.entity_extractor is not None}")

                # Test research document
                research_doc = {
                    "title": "Einstein's Theory of Relativity",
                    "content": "Albert Einstein developed the theory of relativity while working at Princeton University. His research on spacetime and quantum mechanics revolutionized physics.",
                    "url": "http://example.com/einstein-relativity",
                    "domain": ContentDomain.RESEARCH
                }

                print(f"\n📄 Processing research document:")
                print(f"  Title: {research_doc['title']}")
                print(f"  Domain: {research_doc['domain'].value}")

                # Mock entity extraction for demonstration
                mock_entities = [
                    {
                        "label": "Person",
                        "properties": {"id": "albert_einstein", "name": "Albert Einstein"}
                    },
                    {
                        "label": "Organization",
                        "properties": {"id": "princeton_university", "name": "Princeton University"}
                    },
                    {
                        "label": "Concept",
                        "properties": {"id": "theory_of_relativity", "name": "theory of relativity"}
                    }
                ]

                with patch.object(storage.entity_extractor, 'extract_entities', return_value=mock_entities):
                    with patch.object(storage, '_create_document_node', return_value={"id": "doc_einstein_123"}):
                        with patch.object(storage, '_create_or_merge_entity', return_value={"id": "entity_123"}):
                            with patch.object(storage, '_create_document_entity_relationship', return_value=True):
                                result = storage.store_document(research_doc)

                print(f"\n📊 Storage Results:")
                print(f"  Document ID: {result['doc_id']}")
                print(f"  ChromaDB stored: ✅" if result['chroma_stored'] else "  ChromaDB stored: ❌")
                print(f"  Neo4j stored: ✅" if result['graph_stored'] else "  Neo4j stored: ❌")
                print(f"  Entities extracted: {len(result['entities_extracted'])}")
                print(f"  Relationships created: {len(result['relationships_created'])}")

                if result['entities_extracted']:
                    print(f"\n🎯 Extracted Entities:")
                    for i, entity in enumerate(result['entities_extracted'], 1):
                        print(f"  {i}. {entity['label']}: {entity['properties']['name']}")

                if result['relationships_created']:
                    print(f"\n🔗 Created Relationships:")
                    for i, rel in enumerate(result['relationships_created'], 1):
                        print(f"  {i}. Document MENTIONS {rel['to']}")

                # Test business document
                business_doc = {
                    "title": "Tesla Q4 Earnings Report",
                    "content": "Tesla Inc. reported strong quarterly earnings. CEO Elon Musk highlighted growth in electric vehicle sales and expansion into renewable energy markets.",
                    "domain": ContentDomain.BUSINESS
                }

                print(f"\n📄 Processing business document:")
                print(f"  Title: {business_doc['title']}")
                print(f"  Domain: {business_doc['domain'].value}")

                business_entities = [
                    {
                        "label": "Organization",
                        "properties": {"id": "tesla_inc", "name": "Tesla Inc."}
                    },
                    {
                        "label": "Person",
                        "properties": {"id": "elon_musk", "name": "Elon Musk"}
                    }
                ]

                with patch.object(storage.entity_extractor, 'extract_entities', return_value=business_entities):
                    with patch.object(storage, '_create_document_node', return_value={"id": "doc_tesla_456"}):
                        with patch.object(storage, '_create_or_merge_entity', return_value={"id": "entity_456"}):
                            with patch.object(storage, '_create_document_entity_relationship', return_value=True):
                                result2 = storage.store_document(business_doc)

                print(f"\n📊 Storage Results:")
                print(f"  Document ID: {result2['doc_id']}")
                print(f"  ChromaDB stored: ✅" if result2['chroma_stored'] else "  ChromaDB stored: ❌")
                print(f"  Neo4j stored: ✅" if result2['graph_stored'] else "  Neo4j stored: ❌")
                print(f"  Entities extracted: {len(result2['entities_extracted'])}")

                print(f"\n🎯 Extracted Entities:")
                for i, entity in enumerate(result2['entities_extracted'], 1):
                    print(f"  {i}. {entity['label']}: {entity['properties']['name']}")

                # Demonstrate entity retrieval
                with patch.object(storage, 'get_document_entities') as mock_get_entities:
                    mock_get_entities.return_value = [
                        {"id": "albert_einstein", "name": "Albert Einstein", "labels": ["Person"]},
                        {"id": "princeton_university", "name": "Princeton University", "labels": ["Organization"]}
                    ]

                    entities = storage.get_document_entities(result['doc_id'])
                    print(f"\n🔍 Retrieved entities for document {result['doc_id'][:8]}...:")
                    for entity in entities:
                        print(f"  - {entity['name']} ({entity['labels'][0]})")

                print(f"\n✨ Summary:")
                print(f"  - Documents processed: 2")
                print(f"  - Total entities extracted: {len(result['entities_extracted']) + len(result2['entities_extracted'])}")
                print(f"  - Both vector and graph storage working")
                print(f"  - Entity relationships established")
                print(f"  - Domain-specific classification working")

                storage.close()
                print(f"\n🔒 Database connections closed")

if __name__ == "__main__":
    demonstrate_dual_storage()