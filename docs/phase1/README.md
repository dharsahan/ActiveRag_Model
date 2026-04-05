# Phase 1: Hybrid Vector-Graph RAG Foundation

This directory contains documentation for Phase 1 of the Hybrid Vector-Graph RAG system implementation. Phase 1 establishes the foundational infrastructure for dual storage (ChromaDB + Neo4j) and basic entity extraction capabilities.

## What's Been Implemented

### ✅ Core Infrastructure
- **Neo4j Integration**: Community edition setup with Docker, connection management, and schema constraints
- **Dual Storage Architecture**: Seamless writes to both ChromaDB (vector) and Neo4j (graph) storage
- **Configuration Management**: Environment variables and feature toggles for graph functionality
- **Docker Setup**: Containerized Neo4j with APOC plugins and proper networking

### ✅ NLP Pipeline Enhancement
- **Document Classification**: Automatic categorization into research, technical, business, or mixed web content
- **Entity Extraction**: Domain-specific extraction using spaCy with custom heuristics for each content type
- **Schema Validation**: Comprehensive validation for entities and relationships across all domains
- **Normalization**: Consistent entity IDs and deduplication logic

### ✅ Knowledge Graph Operations  
- **Basic CRUD**: Create, read, update operations for entities and relationships
- **Graph Queries**: Pathfinding, neighborhood exploration, and entity search
- **Multi-hop Reasoning**: Foundation for complex reasoning queries (basic implementation)
- **Statistics**: Graph metrics and health monitoring

### ✅ Integration & Testing
- **Comprehensive Test Suite**: Unit tests for all components plus integration tests
- **Environment Setup**: Automated setup script for development environment
- **Documentation**: Architecture diagrams and API reference

## Quick Start

### Prerequisites
- Python 3.8+
- Docker and Docker Compose
- 4GB+ RAM available for Neo4j

### 1. Environment Setup
```bash
# Clone the repository and navigate to it
git clone <repository-url>
cd active-rag

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run automated setup (installs dependencies, starts Neo4j, runs tests)
python scripts/setup_phase1_environment.py
```

The setup script will:
- Install Python dependencies from `requirements.txt` and `requirements_nlp.txt`
- Download the spaCy English model (`en_core_web_sm`)
- Start a Neo4j Docker container with APOC plugins
- Create a `.env` file with default configuration
- Run integration tests to validate the setup

### 2. Verify Installation
```bash
# Run the integration test suite
pytest tests/integration/test_phase1_integration.py -v

# Check that Neo4j is accessible
curl http://localhost:7474
# Should return Neo4j browser interface
```

### 3. Process Your First Document
```python
from active_rag.document_loader import DocumentLoader
from active_rag.schemas.entities import ContentDomain

# Initialize with graph features enabled
loader = DocumentLoader()

# Process a research document
result = loader.load_and_store('path/to/research_paper.txt', ContentDomain.RESEARCH)

print(f"Documents processed: {result['documents_processed']}")
print(f"Entities extracted: {len(result['storage_results'][0]['entities_extracted'])}")
print(f"ChromaDB stored: {result['storage_results'][0]['chroma_stored']}")
print(f"Graph stored: {result['storage_results'][0]['graph_stored']}")
```

## System Architecture

```
┌─────────────────────┐    ┌─────────────────────┐
│   Document Input    │───▶│  Document Loader    │
│   (.txt, .md,       │    │                     │
│    .pdf, .docx)     │    │                     │
└─────────────────────┘    └─────────────────────┘
                                       │
                                       ▼
                           ┌─────────────────────┐
                           │ Document Classifier │
                           │   (Research/Tech/   │
                           │   Business/Mixed)   │
                           └─────────────────────┘
                                       │
                                       ▼
                           ┌─────────────────────┐
                           │  Entity Extractor   │
                           │  (Domain-Specific)  │
                           └─────────────────────┘
                                       │
                                       ▼
                           ┌─────────────────────┐
                           │ Dual Storage Writer │
                           │  + Validation       │
                           └─────────────────────┘
                                  │         │
                                  ▼         ▼
                    ┌─────────────────┐  ┌─────────────────┐
                    │    ChromaDB     │  │     Neo4j       │
                    │   (Vectors)     │  │   (Graph)       │
                    │                 │  │                 │
                    │ • Embeddings    │  │ • Entities      │
                    │ • Similarity    │  │ • Relationships │
                    │ • Full-text     │  │ • Graph queries │
                    └─────────────────┘  └─────────────────┘
```

## Content Domain Support

Phase 1 supports four content domains with specialized entity extraction:

| Domain | Entity Types | Relationship Types | Example Use Cases |
|--------|-------------|-------------------|-------------------|
| **Research** | Person, Organization, Concept, Document | AUTHORED, AFFILIATED_WITH, CITES, BUILDS_ON | "Who collaborated with Einstein on quantum mechanics?" |
| **Technical** | Component, API, Service, Configuration, Document | DEPENDS_ON, CONFIGURED_BY, IMPLEMENTS | "Which services depend on the authentication API?" |
| **Business** | Person, Team, Process, Product, Document | MANAGES, WORKS_FOR, RESPONSIBLE_FOR | "Who manages the ML platform development team?" |
| **Mixed Web** | General entities from web content, Document | MENTIONS, RELATED_TO | "What organizations are mentioned with climate research?" |

### Entity Schemas

Each domain uses specific entity types with required and optional properties:

```python
# Research Domain
PERSON_SCHEMA = EntitySchema(
    label="Person",
    required_properties=["name", "id"],
    optional_properties=["affiliation", "email", "orcid"],
    domain=ContentDomain.RESEARCH
)

# Technical Domain  
COMPONENT_SCHEMA = EntitySchema(
    label="Component",
    required_properties=["name", "id"],
    optional_properties=["version", "type", "description"],
    domain=ContentDomain.TECHNICAL
)
```

## Configuration

The system uses environment variables for configuration. Create a `.env` file:

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=activerag123

# Feature Toggles
ENABLE_GRAPH_FEATURES=true
ENABLE_RELATION_EXTRACTION=true

# ChromaDB Configuration
CHROMA_PERSIST_DIR=./chroma_db
COLLECTION_NAME=active_rag

# NLP Configuration
SPACY_MODEL=en_core_web_sm

# Performance Settings
TOP_K=3
MAX_GRAPH_HOPS=3
```

## Example Workflows

### Processing Different Document Types

```python
from active_rag.document_loader import DocumentLoader
from active_rag.schemas.entities import ContentDomain

loader = DocumentLoader()

# Research paper
research_result = loader.load_and_store(
    'research/quantum_computing.pdf', 
    ContentDomain.RESEARCH
)

# Technical documentation
tech_result = loader.load_and_store(
    'docs/api_reference.md',
    ContentDomain.TECHNICAL
)

# Business document
business_result = loader.load_and_store(
    'processes/deployment_workflow.docx',
    ContentDomain.BUSINESS
)
```

### Graph Queries

```python
from active_rag.knowledge_graph.graph_operations import GraphOperations
from active_rag.knowledge_graph.neo4j_client import Neo4jClient

# Initialize graph operations
neo4j_client = Neo4jClient("bolt://localhost:7687", "neo4j", "activerag123")
graph_ops = GraphOperations(neo4j_client)

# Search for entities
entities = graph_ops.search_entities_by_name("Einstein", ["Person"])

# Find related entities (1-hop)
related = graph_ops.find_related_entities(entities[0]["id"], depth=1)

# Multi-hop reasoning query
result = graph_ops.multi_hop_query(
    "Who worked with Einstein on relativity?", 
    max_hops=2
)

# Get graph statistics
stats = graph_ops.get_graph_stats()
print(f"Graph has {stats['total_nodes']} nodes and {stats['total_relationships']} relationships")
```

### Hybrid Retrieval

```python
from active_rag.storage.dual_storage_manager import DualStorageManager
from active_rag.config import Config

config = Config()
dual_storage = DualStorageManager(config)

# Process document with dual storage
doc_data = {
    "title": "Quantum Computing Research",
    "content": "Research on quantum computing algorithms...",
    "url": "https://example.com/research.pdf"
}

result = dual_storage.store_document(doc_data)

# Vector search in ChromaDB
vector_results = dual_storage.collection.query(
    query_texts=["quantum algorithms"],
    n_results=5
)

# Graph search in Neo4j
graph_ops = GraphOperations(dual_storage.neo4j_client)
graph_entities = graph_ops.search_entities_by_name("quantum", ["Concept"])
```

## Testing

### Running Tests

```bash
# All integration tests
pytest tests/integration/ -v

# Specific test categories
pytest tests/integration/test_phase1_integration.py::test_full_document_ingestion_pipeline -v
pytest tests/integration/test_phase1_integration.py::test_graph_operations_with_real_data -v
pytest tests/integration/test_phase1_integration.py::test_hybrid_retrieval_capability -v

# Performance tests
pytest tests/integration/test_phase1_integration.py::test_performance_with_multiple_documents -v
```

### Test Coverage

The integration test suite covers:
- ✅ **Full Document Pipeline**: File loading → classification → entity extraction → dual storage
- ✅ **Graph Operations**: Entity search, relationship traversal, neighborhood exploration
- ✅ **Multi-hop Reasoning**: Complex query processing with path finding
- ✅ **Hybrid Retrieval**: Vector similarity + graph traversal
- ✅ **Cross-domain Linking**: Entity linking across different content domains
- ✅ **Performance**: Multi-document ingestion and query performance
- ✅ **Resilience**: Error handling and edge cases
- ✅ **Consistency**: Vector-graph storage synchronization

## Monitoring and Management

### Neo4j Browser

Access the Neo4j browser at `http://localhost:7474`:
- Username: `neo4j`
- Password: `activerag123`

Useful Cypher queries:
```cypher
// Count entities by type
MATCH (n) RETURN labels(n) as entity_type, count(*) as count

// Find all documents and their entities
MATCH (d:Document)-[:MENTIONS]->(e) 
RETURN d.title, labels(e), e.name LIMIT 10

// Find entity relationships
MATCH (a)-[r]->(b) 
RETURN labels(a), type(r), labels(b), count(*) as relationship_count

// Search for specific entities
MATCH (p:Person {name: "Einstein"}) 
RETURN p
```

### ChromaDB

ChromaDB data is persisted in `./chroma_db/` directory. The collection stores:
- Document content as embeddings
- Metadata: title, URL, domain classification
- Automatic similarity indexing

## Limitations and Known Issues

### Phase 1 Limitations
1. **Basic Entity Extraction**: Uses rule-based spaCy patterns, not advanced NER models
2. **Simple Relationships**: Limited to basic relationship types (MENTIONS, etc.)
3. **No Query Routing**: Doesn't automatically choose between vector vs graph retrieval
4. **Basic Multi-hop**: Simple pathfinding without sophisticated reasoning
5. **Domain-Specific**: Entity extraction optimized for specific domains only

### Known Issues
1. **Memory Usage**: Large documents may consume significant memory during processing
2. **Neo4j Startup**: First-time Neo4j container startup can take 2-3 minutes
3. **spaCy Model**: English model required - may not work well with other languages
4. **Concurrent Access**: Not optimized for high-concurrency scenarios

## What's Next (Phase 2)

Phase 1 provides the foundation. Phase 2 will add:

### 🔄 **Advanced Relation Extraction**
- Integration with OpenNRE for sophisticated relationship detection
- Training domain-specific relation extraction models
- Improved relationship confidence scoring

### 🧠 **Intelligent Query Routing**
- Auto-detect whether to use vector, graph, or hybrid retrieval
- Query complexity analysis and optimization
- Smart fallback strategies

### 📊 **Enhanced Confidence Scoring**
- Incorporate graph signals into answer confidence
- Multi-signal confidence aggregation
- Uncertainty quantification

### 🔍 **Multi-hop Query Processing**
- Advanced pathfinding algorithms
- Reasoning chain construction and validation
- Explanation generation for complex queries

### ⚡ **Performance Optimization**
- Caching strategies for frequent queries
- Batch processing capabilities
- Distributed processing support

## Getting Help

- **Architecture Details**: See `docs/phase1/architecture.md`
- **API Reference**: See `docs/phase1/api-reference.md`
- **Troubleshooting**: See `docs/phase1/troubleshooting.md`
- **Integration Tests**: Run `pytest tests/integration/ -v` for examples
- **Neo4j Browser**: http://localhost:7474 for graph exploration
- **Issues**: Check logs in the application output for debugging

## Contributing

When contributing to Phase 1:

1. **Run Tests**: Always run the full integration test suite
2. **Check Compatibility**: Ensure changes work with existing entity schemas
3. **Document Changes**: Update relevant documentation files
4. **Performance**: Consider impact on document processing speed
5. **Memory**: Be mindful of memory usage with large documents

The Phase 1 foundation is designed to be stable and extensible for Phase 2 enhancements.