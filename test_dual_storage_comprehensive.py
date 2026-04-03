#!/usr/bin/env python3
"""
Comprehensive integration test for dual storage functionality with realistic scenarios.
This test focuses on end-to-end functionality rather than unit testing individual methods.
"""

import sys
import os
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from active_rag.schemas.entities import ContentDomain

def test_dual_storage_with_graph_disabled():
    """Test dual storage when graph features are disabled - this should work without Neo4j"""
    print("🧪 Testing dual storage with graph features disabled...")

    from active_rag.config import Config
    from active_rag.storage.dual_storage_manager import DualStorageManager

    # Create config with graph features disabled
    config = Config()
    config.enable_graph_features = False

    # Use temporary directory for ChromaDB
    with tempfile.TemporaryDirectory() as temp_dir:
        config.chroma_persist_dir = temp_dir

        storage = DualStorageManager(config)

        # Verify correct initialization
        assert storage.chroma_client is not None
        assert storage.neo4j_client is None
        assert storage.schema_manager is None
        print("  ✓ DualStorageManager correctly initialized with graph features disabled")

        # Test document storage
        doc_data = {
            "title": "Test Document",
            "content": "This is a test document about artificial intelligence and machine learning.",
            "url": "http://example.com/test-doc",
            "domain": ContentDomain.RESEARCH
        }

        result = storage.store_document(doc_data)

        # Verify results
        assert result["chroma_stored"] == True
        assert result["graph_stored"] == False
        assert len(result["entities_extracted"]) == 0
        assert len(result["relationships_created"]) == 0
        assert "doc_id" in result

        print(f"  ✓ Document stored successfully: {result['doc_id']}")
        print("  ✓ ChromaDB storage working")
        print("  ✓ Graph storage correctly skipped")

        return True

def test_document_id_generation():
    """Test document ID generation consistency"""
    print("\n🧪 Testing document ID generation...")

    from active_rag.config import Config
    from active_rag.storage.dual_storage_manager import DualStorageManager

    config = Config()
    config.enable_graph_features = False

    with tempfile.TemporaryDirectory() as temp_dir:
        config.chroma_persist_dir = temp_dir
        storage = DualStorageManager(config)

        # Test with URL
        doc1 = {"url": "http://example.com/doc1", "content": "test"}
        doc2 = {"url": "http://example.com/doc1", "content": "different content"}

        id1 = storage._generate_doc_id(doc1)
        id2 = storage._generate_doc_id(doc2)

        assert id1 == id2  # Same URL should generate same ID
        expected_url_id = hashlib.md5("http://example.com/doc1".encode()).hexdigest()
        assert id1 == expected_url_id
        print("  ✓ URL-based ID generation working")

        # Test without URL
        doc3 = {"title": "Test", "content": "Content here"}
        doc4 = {"title": "Test", "content": "Content here"}

        id3 = storage._generate_doc_id(doc3)
        id4 = storage._generate_doc_id(doc4)

        assert id3 == id4  # Same title+content should generate same ID
        expected_content_id = hashlib.md5("Test_Content here".encode()).hexdigest()
        assert id3 == expected_content_id
        print("  ✓ Content-based ID generation working")

        return True

def test_document_loader_integration():
    """Test DocumentLoader integration with dual storage"""
    print("\n🧪 Testing DocumentLoader integration...")

    from active_rag.config import Config
    from active_rag.document_loader import DocumentLoader

    # Create test document
    test_content = """
    Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity.
    She worked at the University of Paris and won Nobel Prizes in Physics and Chemistry.
    Her discoveries laid the groundwork for modern atomic theory.
    """

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content.strip())
        test_file_path = f.name

    try:
        # Test with graph features disabled
        config = Config()
        config.enable_graph_features = False

        with tempfile.TemporaryDirectory() as temp_dir:
            config.chroma_persist_dir = temp_dir

            loader = DocumentLoader(config)
            assert loader.dual_storage is None  # Should be None when graph disabled
            print("  ✓ DocumentLoader correctly handles disabled graph features")

            # Test document loading
            documents = loader.load(test_file_path)
            assert len(documents) == 1
            doc = documents[0]
            assert "Marie Curie" in doc.content
            assert "radioactivity" in doc.content
            print("  ✓ Document loading working correctly")

            # Test load_and_store with disabled graph
            result = loader.load_and_store(test_file_path, ContentDomain.RESEARCH)
            assert result["documents_processed"] == 1
            assert len(result["storage_results"]) == 0  # No storage when dual_storage is None
            print("  ✓ load_and_store handles disabled graph features correctly")

    finally:
        # Cleanup
        os.unlink(test_file_path)

    return True

def test_entity_extraction_components():
    """Test that NLP components can be imported and basic functionality works"""
    print("\n🧪 Testing NLP components basic functionality...")

    from active_rag.nlp_pipeline.entity_extractor import EntityExtractor
    from active_rag.nlp_pipeline.document_classifier import DocumentClassifier

    extractor = EntityExtractor()
    classifier = DocumentClassifier()

    print("  ✓ EntityExtractor and DocumentClassifier imported successfully")

    # Test basic classification
    test_text = "Einstein's theory of relativity revolutionized physics."
    domain = classifier.classify_document(test_text)
    assert domain in [ContentDomain.RESEARCH, ContentDomain.MIXED_WEB]
    print(f"  ✓ Document classification working: {domain}")

    # Test entity extraction
    entities = extractor.extract_entities(test_text, ContentDomain.RESEARCH)
    assert isinstance(entities, list)
    print(f"  ✓ Entity extraction working: found {len(entities)} entities")

    if entities:
        for entity in entities:
            assert "label" in entity
            assert "properties" in entity
            assert "id" in entity["properties"]
        print("  ✓ Entity structure validation passed")

    return True

def run_integration_tests():
    """Run all integration tests"""
    print("🚀 Running Dual Storage Integration Tests...\n")

    tests = [
        test_dual_storage_with_graph_disabled,
        test_document_id_generation,
        test_document_loader_integration,
        test_entity_extraction_components
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ Test {test_func.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n📊 Integration Test Results:")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📈 Success Rate: {passed}/{passed+failed} ({100*passed/(passed+failed):.1f}%)")

    if failed == 0:
        print("\n🎉 All integration tests passed!")
        print("✨ Dual storage system is working correctly")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)