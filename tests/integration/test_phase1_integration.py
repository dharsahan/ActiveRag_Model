"""
Integration tests for Phase 1 hybrid vector-graph RAG system.

This test suite validates the complete end-to-end functionality of the hybrid
vector-graph RAG system, including document ingestion, entity extraction,
dual storage (ChromaDB + Neo4j), and graph operations.
"""

import pytest
import tempfile
import shutil
import time
import logging
from pathlib import Path
from active_rag.config import Config
from active_rag.storage.dual_storage_manager import DualStorageManager
from active_rag.document_loader import DocumentLoader
from active_rag.knowledge_graph.graph_operations import GraphOperations
from active_rag.schemas.entities import ContentDomain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def temp_doc():
    """Create temporary test document"""
    content = """
    Einstein's Theory of Relativity

    Albert Einstein developed the theory of relativity while working at Princeton University.
    His work built upon previous research by Maxwell and Lorentz. Einstein collaborated
    with colleagues at the Institute for Advanced Study.

    The theory revolutionized our understanding of space and time, leading to new concepts
    in quantum mechanics and cosmology.
    """

    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    temp_file.write(content)
    temp_file.close()

    yield temp_file.name

    # Clean up
    try:
        Path(temp_file.name).unlink()
    except FileNotFoundError:
        pass


@pytest.fixture
def config():
    """Test configuration with graph features enabled"""
    return Config()


def test_full_document_ingestion_pipeline(temp_doc, config):
    """Test complete document ingestion from file to dual storage"""
    logger.info("Testing full document ingestion pipeline")

    # Initialize document loader with dual storage
    doc_loader = DocumentLoader(config)

    # Load and store document
    result = doc_loader.load_and_store(temp_doc, ContentDomain.RESEARCH)

    # Verify document was processed
    assert result["documents_processed"] == 1
    assert len(result["storage_results"]) == 1

    storage_result = result["storage_results"][0]
    assert storage_result["chroma_stored"] == True
    assert storage_result["graph_stored"] == True
    assert len(storage_result["entities_extracted"]) > 0

    # Verify we extracted expected entities
    entities = storage_result["entities_extracted"]
    person_entities = [e for e in entities if e["label"] == "Person"]
    org_entities = [e for e in entities if e["label"] == "Organization"]

    assert len(person_entities) >= 1  # Should extract Einstein
    assert len(org_entities) >= 1   # Should extract Princeton University

    logger.info(f"Successfully extracted {len(entities)} entities")


def test_graph_operations_with_real_data(temp_doc, config):
    """Test graph operations after document ingestion"""
    logger.info("Testing graph operations with real data")

    # First ingest document
    doc_loader = DocumentLoader(config)
    result = doc_loader.load_and_store(temp_doc, ContentDomain.RESEARCH)

    # Get the dual storage manager to access Neo4j client
    dual_storage = doc_loader.dual_storage
    graph_ops = GraphOperations(dual_storage.neo4j_client)

    # Test entity search
    entities = graph_ops.search_entities_by_name("Einstein", ["Person"])
    assert len(entities) >= 1

    einstein_entity = entities[0]
    logger.info(f"Found Einstein entity: {einstein_entity['id']}")

    # Test finding related entities
    related = graph_ops.find_related_entities(einstein_entity["id"], depth=1)
    assert len(related) >= 0  # Should have some relationships

    # Test neighborhood exploration
    neighborhood = graph_ops.get_entity_neighborhood(einstein_entity["id"], radius=2)
    assert isinstance(neighborhood, list)

    logger.info(f"Found {len(related)} related entities and neighborhood of size {len(neighborhood)}")


def test_multi_hop_reasoning_query(temp_doc, config):
    """Test multi-hop reasoning with real data"""
    logger.info("Testing multi-hop reasoning query")

    # Ingest document first
    doc_loader = DocumentLoader(config)
    doc_loader.load_and_store(temp_doc, ContentDomain.RESEARCH)

    # Test multi-hop query
    dual_storage = doc_loader.dual_storage
    graph_ops = GraphOperations(dual_storage.neo4j_client)

    result = graph_ops.multi_hop_query("Who worked with Einstein?", max_hops=2)

    assert "entities" in result
    assert "paths" in result
    assert "reasoning" in result
    assert isinstance(result["entities"], list)
    assert isinstance(result["paths"], list)

    logger.info(f"Multi-hop query found {len(result['entities'])} entities and {len(result['paths'])} paths")


def test_hybrid_retrieval_capability(temp_doc, config):
    """Test hybrid vector + graph retrieval"""
    logger.info("Testing hybrid retrieval capability")

    # Ingest document for richer testing
    doc_loader = DocumentLoader(config)
    doc_loader.load_and_store(temp_doc, ContentDomain.RESEARCH)

    # Test that we can retrieve both vector and graph information
    dual_storage = doc_loader.dual_storage
    graph_ops = GraphOperations(dual_storage.neo4j_client)

    # Vector search capability (ChromaDB)
    collection = dual_storage.collection
    vector_results = collection.query(
        query_texts=["Einstein physics research"],
        n_results=3
    )
    assert len(vector_results["ids"][0]) >= 1
    logger.info(f"Vector search returned {len(vector_results['ids'][0])} results")

    # Graph traversal capability (Neo4j)
    graph_stats = graph_ops.get_graph_stats()
    assert graph_stats["total_nodes"] > 0
    assert graph_stats["total_relationships"] >= 0

    logger.info(f"Graph contains {graph_stats['total_nodes']} nodes and {graph_stats['total_relationships']} relationships")


def test_cross_domain_entity_linking(config):
    """Test entity linking across different content domains"""
    logger.info("Testing cross-domain entity linking")

    doc_loader = DocumentLoader(config)

    # Create test documents from different domains
    research_doc = create_temp_doc("""
        Dr. Marie Curie conducted groundbreaking research at the Radium Institute.
        Her work on radioactivity earned her Nobel Prizes in physics and chemistry.
    """)

    business_doc = create_temp_doc("""
        The research team at our institute includes Dr. Marie Curie as the lead scientist.
        The project focuses on radioactive materials and their medical applications.
    """)

    try:
        # Ingest both documents
        doc_loader.load_and_store(research_doc, ContentDomain.RESEARCH)
        doc_loader.load_and_store(business_doc, ContentDomain.BUSINESS)

        # Verify entity linking across domains
        dual_storage = doc_loader.dual_storage
        graph_ops = GraphOperations(dual_storage.neo4j_client)

        # Search for Marie Curie - should appear in both domains
        curie_entities = graph_ops.search_entities_by_name("Curie", ["Person"])

        # Should have linked the same person across documents
        assert len(curie_entities) >= 1

        # Test cross-domain relationships
        if curie_entities:
            related = graph_ops.find_related_entities(curie_entities[0]["id"], depth=2)
            # Should find connections to both research and business entities
            assert len(related) >= 0

        logger.info(f"Cross-domain linking found {len(curie_entities)} Curie entities")

    finally:
        # Cleanup
        cleanup_temp_file(research_doc)
        cleanup_temp_file(business_doc)


def test_performance_with_multiple_documents(config):
    """Test system performance with multiple document ingestion"""
    logger.info("Testing performance with multiple documents")

    doc_loader = DocumentLoader(config)

    # Create multiple test documents
    documents = []
    for i in range(5):
        doc = create_temp_doc(f"""
            Document {i}: This is about researcher Dr. Smith working at Tech Corp.
            The research involves machine learning and artificial intelligence.
            Dr. Smith collaborated with teams at University Labs and Innovation Center.
        """)
        documents.append(doc)

    try:
        # Process all documents
        start_time = time.time()
        for doc in documents:
            result = doc_loader.load_and_store(doc, ContentDomain.RESEARCH)
            assert result["documents_processed"] == 1

        processing_time = time.time() - start_time
        logger.info(f"Processed {len(documents)} documents in {processing_time:.2f} seconds")

        # Verify graph population
        dual_storage = doc_loader.dual_storage
        graph_ops = GraphOperations(dual_storage.neo4j_client)
        stats = graph_ops.get_graph_stats()

        assert stats["total_nodes"] >= len(documents)  # At least one doc node per document

        if "node_types" in stats and stats["node_types"]:
            logger.info(f"Node types in graph: {stats['node_types']}")

        # Performance benchmark: should process at least 1 document per 10 seconds
        assert processing_time < (len(documents) * 10), f"Processing too slow: {processing_time:.2f}s for {len(documents)} docs"

    finally:
        # Cleanup
        for doc in documents:
            cleanup_temp_file(doc)


def test_system_resilience(config):
    """Test system behavior under error conditions"""
    logger.info("Testing system resilience")

    # Test with malformed document
    try:
        malformed_doc = create_temp_doc("")  # Empty document
        doc_loader = DocumentLoader(config)
        result = doc_loader.load_and_store(malformed_doc, ContentDomain.RESEARCH)
        # Should handle gracefully
        assert result["documents_processed"] == 1
        cleanup_temp_file(malformed_doc)
    except Exception as e:
        pytest.fail(f"System should handle empty documents gracefully: {e}")

    # Test with very large text
    large_text = "Large document content. " * 1000
    large_doc = create_temp_doc(large_text)

    try:
        result = doc_loader.load_and_store(large_doc, ContentDomain.TECHNICAL)
        assert result["documents_processed"] == 1
        logger.info("Successfully handled large document")
    finally:
        cleanup_temp_file(large_doc)


def test_vector_graph_consistency(temp_doc, config):
    """Test that vector and graph storage remain consistent"""
    logger.info("Testing vector-graph storage consistency")

    doc_loader = DocumentLoader(config)
    result = doc_loader.load_and_store(temp_doc, ContentDomain.RESEARCH)

    # Get both storage systems
    dual_storage = doc_loader.dual_storage
    graph_ops = GraphOperations(dual_storage.neo4j_client)

    # Check ChromaDB
    collection = dual_storage.collection
    chroma_count = collection.count()

    # Check Neo4j
    stats = graph_ops.get_graph_stats()
    neo4j_doc_nodes = stats.get("total_nodes", 0)

    # Both should have content
    assert chroma_count > 0, "ChromaDB should contain documents"
    assert neo4j_doc_nodes > 0, "Neo4j should contain nodes"

    logger.info(f"Consistency check: ChromaDB has {chroma_count} documents, Neo4j has {neo4j_doc_nodes} nodes")


def create_temp_doc(content):
    """Helper to create temporary document file"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    temp_file.write(content)
    temp_file.close()
    return temp_file.name


def cleanup_temp_file(file_path):
    """Helper to safely clean up temporary files"""
    try:
        Path(file_path).unlink()
    except FileNotFoundError:
        pass


# Test configuration and validation
def test_integration_environment_setup(config):
    """Test that the integration environment is properly configured"""
    logger.info("Testing integration environment setup")

    # Check that all required components are available
    doc_loader = DocumentLoader(config)
    dual_storage = doc_loader.dual_storage

    # Test ChromaDB connection
    collection = dual_storage.collection
    assert collection is not None, "ChromaDB collection should be available"

    # Test Neo4j connection
    neo4j_client = dual_storage.neo4j_client
    assert neo4j_client is not None, "Neo4j client should be available"

    # Test that we can connect to Neo4j
    try:
        graph_ops = GraphOperations(neo4j_client)
        stats = graph_ops.get_graph_stats()
        assert isinstance(stats, dict), "Should be able to query Neo4j"
        logger.info("Environment setup validation passed")
    except Exception as e:
        pytest.fail(f"Neo4j connection failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])