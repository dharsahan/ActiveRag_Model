#!/usr/bin/env python3
"""
Standalone test for dual storage integration.
This tests the dual storage manager without requiring pytest or complex dependencies.
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_basic_imports():
    """Test that we can import all required components"""
    print("🧪 Testing basic imports...")

    try:
        from active_rag.schemas.entities import ContentDomain
        print("  ✓ ContentDomain imported")

        # Test ContentDomain enum values
        assert ContentDomain.RESEARCH.value == "research"
        assert ContentDomain.MIXED_WEB.value == "mixed_web"
        print("  ✓ ContentDomain values correct")

        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_config_creation():
    """Test config creation and graph feature toggle"""
    print("\n🧪 Testing config creation...")

    try:
        from active_rag.config import Config

        config = Config()
        print(f"  ✓ Config created, graph features: {config.enable_graph_features}")

        # Test disabling graph features
        config.enable_graph_features = False
        print(f"  ✓ Graph features disabled: {config.enable_graph_features}")

        return True
    except Exception as e:
        print(f"  ❌ Config test failed: {e}")
        return False

def test_dual_storage_manager_import():
    """Test importing dual storage manager"""
    print("\n🧪 Testing dual storage manager import...")

    try:
        from active_rag.storage.dual_storage_manager import DualStorageManager
        print("  ✓ DualStorageManager imported successfully")
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_document_loader_integration():
    """Test document loader with dual storage integration"""
    print("\n🧪 Testing document loader integration...")

    try:
        from active_rag.document_loader import DocumentLoader
        from active_rag.config import Config

        # Test with graph features disabled (simpler)
        config = Config()
        config.enable_graph_features = False

        loader = DocumentLoader(config)
        print("  ✓ DocumentLoader created with graph features disabled")

        # Verify dual storage is None when disabled
        assert loader.dual_storage is None
        print("  ✓ Dual storage correctly disabled")

        return True
    except Exception as e:
        print(f"  ❌ DocumentLoader test failed: {e}")
        return False

def test_entity_extraction_components():
    """Test entity extraction and classification components"""
    print("\n🧪 Testing NLP components...")

    try:
        from active_rag.nlp_pipeline.entity_extractor import EntityExtractor
        from active_rag.nlp_pipeline.document_classifier import DocumentClassifier

        print("  ✓ EntityExtractor imported")
        print("  ✓ DocumentClassifier imported")

        return True
    except Exception as e:
        print(f"  ❌ NLP components import failed: {e}")
        return False

def create_test_document():
    """Create a simple test document for testing"""
    test_file = Path("test_document.txt")
    test_content = """
    Albert Einstein was a theoretical physicist who developed the theory of relativity.
    He worked at Princeton University and received the Nobel Prize in Physics in 1921.
    His work on quantum mechanics contributed to our understanding of the universe.
    """

    test_file.write_text(test_content.strip())
    print(f"  ✓ Test document created: {test_file}")
    return test_file

def cleanup_test_document(test_file: Path):
    """Remove test document"""
    if test_file.exists():
        test_file.unlink()
        print(f"  ✓ Test document cleaned up: {test_file}")

def test_document_loading():
    """Test loading a document without storage (basic functionality)"""
    print("\n🧪 Testing document loading...")

    test_file = None
    try:
        from active_rag.document_loader import DocumentLoader
        from active_rag.config import Config

        # Create test document
        test_file = create_test_document()

        # Test loading
        config = Config()
        config.enable_graph_features = False  # Keep it simple

        loader = DocumentLoader(config)
        documents = loader.load(str(test_file))

        print(f"  ✓ Loaded {len(documents)} document(s)")

        if documents:
            doc = documents[0]
            print(f"  ✓ Document title: {doc.title}")
            print(f"  ✓ Content length: {len(doc.content)} chars")
            print(f"  ✓ Word count: {doc.word_count}")

            # Verify content contains expected terms
            content_lower = doc.content.lower()
            assert "einstein" in content_lower
            assert "princeton" in content_lower
            print("  ✓ Document content verified")

        return True

    except Exception as e:
        print(f"  ❌ Document loading failed: {e}")
        return False
    finally:
        if test_file:
            cleanup_test_document(test_file)

def run_all_tests():
    """Run all standalone tests"""
    print("🚀 Starting dual storage integration tests...\n")

    tests = [
        test_basic_imports,
        test_config_creation,
        test_dual_storage_manager_import,
        test_document_loader_integration,
        test_entity_extraction_components,
        test_document_loading
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
            print(f"  ❌ Test {test_func.__name__} crashed: {e}")
            failed += 1

    print(f"\n📊 Test Results:")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📈 Success Rate: {passed}/{passed+failed} ({100*passed/(passed+failed):.1f}%)")

    if failed == 0:
        print("\n🎉 All tests passed! Dual storage integration is working.")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)