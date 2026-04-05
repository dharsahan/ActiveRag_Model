# Phase 1 Architecture Documentation

This document provides a detailed technical overview of the hybrid vector-graph RAG system architecture implemented in Phase 1.

## Table of Contents
- [System Overview](#system-overview)
- [Core Architecture](#core-architecture)
- [Component Descriptions](#component-descriptions)
- [Data Flow](#data-flow)
- [Storage Architecture](#storage-architecture)
- [NLP Pipeline](#nlp-pipeline)
- [Graph Schema Design](#graph-schema-design)
- [Performance Characteristics](#performance-characteristics)
- [Security Considerations](#security-considerations)
- [Extension Points](#extension-points)

## System Overview

The Phase 1 hybrid vector-graph RAG system combines traditional vector-based retrieval with knowledge graph reasoning to enable more sophisticated question answering and information retrieval capabilities.

### Design Principles

1. **Dual Storage Strategy**: Maintain both vector and graph representations simultaneously
2. **Domain-Aware Processing**: Tailor entity extraction to specific content domains
3. **Extensible Architecture**: Design for easy addition of new capabilities in Phase 2
4. **Configuration-Driven**: Use environment variables and feature toggles for flexibility
5. **Comprehensive Testing**: Integration tests validate end-to-end functionality

### Key Innovations

- **Synchronized Dual Storage**: Automatic writes to both ChromaDB and Neo4j with consistency guarantees
- **Domain-Specific NLP**: Different entity extraction strategies based on document classification
- **Schema-Validated Graph**: Enforced entity and relationship schemas prevent data corruption
- **Hybrid Query Potential**: Foundation for intelligent query routing (Phase 2)

## Core Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Active RAG System                              │
│                                 (Phase 1)                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────────────────────────────────────────┐
│                 │    │                    Document Processing              │
│   Input Layer   │    │                                                     │
│                 │    │  ┌─────────────────┐   ┌─────────────────────────┐  │
│ • File Upload   │────▶│  │ Document Loader │──▶│  Document Classifier     │  │
│ • Text Input    │    │  │ • TXT/MD/PDF/   │   │ • Research              │  │
│ • URL Ingestion │    │  │   DOCX Support  │   │ • Technical             │  │
│                 │    │  │ • Encoding      │   │ • Business              │  │
│                 │    │  │   Handling      │   │ • Mixed Web             │  │
│                 │    │  └─────────────────┘   └─────────────────────────┘  │
└─────────────────┘    └─────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             NLP Processing Pipeline                         │
│                                                                             │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────────┐ │
│  │      Entity Extractor       │    │         Schema Manager              │ │
│  │                             │    │                                     │ │
│  │ Domain-Specific Extraction: │────▶│ • Entity Schema Validation          │ │
│  │ • spaCy NER + Custom Rules  │    │ • Relationship Schema Enforcement   │ │
│  │ • Research: Person/Org/     │    │ • Property Type Checking           │ │
│  │   Concept/Document          │    │ • Constraint Creation              │ │
│  │ • Technical: Component/API/ │    │                                     │ │
│  │   Service/Document          │    │                                     │ │
│  │ • Business: Process/Team/   │    │                                     │ │
│  │   Person/Document           │    │                                     │ │
│  │ • Mixed: General entities   │    │                                     │ │
│  └─────────────────────────────┘    └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Dual Storage Architecture                         │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     Dual Storage Manager                             │  │
│  │                                                                      │  │
│  │ • Coordinated writes to both storage systems                        │  │
│  │ • Document ID generation and consistency                             │  │
│  │ • Entity-document relationship creation                              │  │
│  │ • Error handling and rollback capabilities                          │  │
│  │ • Storage result aggregation and reporting                          │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                    ┌───────────────┴───────────────┐                        │
│                    ▼                               ▼                        │
│  ┌─────────────────────────────┐     ┌─────────────────────────────────────┐ │
│  │      ChromaDB Storage       │     │         Neo4j Storage               │ │
│  │      (Vector Retrieval)     │     │       (Graph Retrieval)             │ │
│  │                             │     │                                     │ │
│  │ • Document embeddings       │     │ • Entity nodes with properties      │ │
│  │ • Similarity search         │     │ • Typed relationships               │ │
│  │ • Metadata filtering        │     │ • Graph traversal queries           │ │
│  │ • Collection management     │     │ • Multi-hop reasoning               │ │
│  │ • Persistent storage        │     │ • Cypher query interface            │ │
│  │                             │     │ • APOC plugin support               │ │
│  └─────────────────────────────┘     └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Query & Reasoning Layer                          │
│                                                                             │
│  ┌─────────────────────────┐     ┌─────────────────────────────────────────┐ │
│  │    Graph Operations     │     │         Query Builder                   │ │
│  │                         │     │                                         │ │
│  │ • Entity search         │─────▶ • Safe Cypher generation                │ │
│  │ • Relationship finding  │     │ • Parameterized queries                 │ │
│  │ • Path discovery        │     │ • Injection prevention                  │ │
│  │ • Neighborhood analysis │     │ • Query optimization                    │ │
│  │ • Multi-hop reasoning   │     │                                         │ │
│  │ • Graph statistics      │     │                                         │ │
│  └─────────────────────────┘     └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Descriptions

### Document Loader
**File**: `active_rag/document_loader.py`

The document loader handles file ingestion and coordinates with the dual storage system.

**Key Responsibilities**:
- Multi-format document parsing (TXT, MD, PDF, DOCX)
- Encoding detection and normalization
- Integration with dual storage manager
- Error handling for unsupported formats

**Design Patterns**:
- Factory pattern for format-specific loaders
- Optional dual storage initialization
- Graceful degradation when graph features disabled

### Document Classifier
**File**: `active_rag/nlp_pipeline/document_classifier.py`

Classifies documents into content domains to enable domain-specific processing.

**Classification Logic**:
```python
# Research indicators
research_indicators = ['research', 'study', 'paper', 'journal', 'university', 'professor']

# Technical indicators  
technical_indicators = ['api', 'function', 'class', 'method', 'database', 'server']

# Business indicators
business_indicators = ['process', 'workflow', 'team', 'department', 'manager', 'policy']
```

**Features**:
- Keyword-based classification with weighted scoring
- Fallback to MIXED_WEB domain for ambiguous content
- Configurable classification thresholds

### Entity Extractor
**File**: `active_rag/nlp_pipeline/entity_extractor.py`

Performs domain-aware entity extraction using spaCy NLP and custom rules.

**Domain Strategies**:

1. **Research Domain**:
   - Extracts persons (researchers, authors)
   - Identifies academic organizations (universities, institutes)
   - Finds concepts (theories, methods, technologies)
   - Creates Document entities for papers/articles

2. **Technical Domain**:
   - Detects software components and APIs
   - Identifies services and microservices
   - Finds configuration entities
   - Recognizes file names and modules

3. **Business Domain**:
   - Extracts business processes and workflows
   - Identifies teams and departments
   - Finds people in business contexts
   - Detects products and projects

4. **Mixed Web Domain**:
   - General named entity recognition
   - Flexible entity classification
   - Broader relationship patterns

**Entity Deduplication**:
- Normalized entity IDs based on name hashing
- Cross-reference checking to prevent duplicates
- Consistent property formatting

### Dual Storage Manager
**File**: `active_rag/storage/dual_storage_manager.py`

Coordinates writes between ChromaDB and Neo4j with consistency guarantees.

**Key Features**:
- **Atomic Operations**: Ensures both systems updated or neither
- **Error Recovery**: Handles partial failures gracefully
- **Document ID Generation**: Consistent hashing for reliable identification
- **Relationship Creation**: Automatically creates MENTIONS relationships

**Storage Flow**:
1. Generate consistent document ID
2. Classify document domain
3. Extract entities using domain-specific rules
4. Validate entities against schema
5. Write to ChromaDB (vector storage)
6. Create document node in Neo4j
7. Create/merge entity nodes in Neo4j
8. Establish document-entity relationships
9. Return comprehensive storage results

### Graph Operations
**File**: `active_rag/knowledge_graph/graph_operations.py`

High-level interface for graph queries and reasoning operations.

**Core Operations**:

1. **Entity Search**: Find entities by name pattern across types
2. **Relationship Discovery**: Find entities related to a starting entity
3. **Path Finding**: Discover paths between two entities
4. **Neighborhood Analysis**: Explore local graph structure
5. **Multi-hop Reasoning**: Complex query processing with NLP
6. **Graph Statistics**: Health monitoring and metrics

**Query Safety**:
- Parameterized queries to prevent Cypher injection
- Input validation for entity types and IDs
- Rate limiting on complex traversals
- Configurable query timeouts

### Neo4j Client
**File**: `active_rag/knowledge_graph/neo4j_client.py`

Low-level Neo4j database interface with connection management.

**Features**:
- Connection pooling and health checks
- Transaction management
- Error handling with retries
- Schema constraint management
- Bulk operation support

### Schema Manager
**File**: `active_rag/knowledge_graph/schema_manager.py`

Enforces graph schema constraints and validates entities.

**Schema Validation**:
- Required property checking
- Property type validation  
- Relationship constraint enforcement
- Entity label consistency

**Constraint Management**:
```cypher
-- Example constraints created
CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT org_id IF NOT EXISTS FOR (o:Organization) REQUIRE o.id IS UNIQUE;
CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE;
```

## Data Flow

### Document Ingestion Flow

```
[File Input] → [Document Loader] → [Document Classifier] 
     ↓
[Entity Extractor] → [Schema Validation] → [Dual Storage Manager]
     ↓                                            ↓
[ChromaDB Storage] ← ← ← ← ← ← ← ← ← ← ← → [Neo4j Storage]
     ↓                                            ↓
[Vector Embeddings]                    [Graph Nodes & Relationships]
     ↓                                            ↓
[Similarity Search]                    [Graph Traversal & Reasoning]
```

### Query Processing Flow

```
[User Query] → [Query Analysis] → [Retrieval Strategy]
                                        ↓
                              [Vector Search] + [Graph Traversal]
                                        ↓
                              [Result Consolidation] → [Response]
```

## Storage Architecture

### ChromaDB Schema

**Collections Structure**:
```python
collection_metadata = {
    "name": "active_rag",
    "metadata": {
        "title": str,        # Document title
        "url": str,          # Source URL or file path
        "domain": str,       # Content domain (research/technical/business/mixed_web)
        "doc_id": str        # Consistent document identifier
    }
}
```

**Storage Benefits**:
- Fast similarity search with embeddings
- Metadata filtering capabilities
- Persistent storage with SQLite backend
- No external dependencies for basic vector operations

### Neo4j Graph Schema

**Node Types**:

```cypher
-- Research Domain
(:Person {id, name, affiliation?, email?, orcid?})
(:Organization {id, name, type?, location?, website?})
(:Concept {id, name, definition?, domain?, aliases?})

-- Technical Domain  
(:Component {id, name, version?, type?, description?})

-- Business Domain
(:Process {id, name, description?, owner?, status?})

-- Universal
(:Document {id, title, url?, domain, content_hash, created_at?})
```

**Relationship Types**:

```cypher
-- Research Relationships
(:Person)-[:AUTHORED]->(:Document)
(:Person)-[:AFFILIATED_WITH]->(:Organization)
(:Document)-[:CITES]->(:Document)
(:Concept)-[:BUILDS_ON]->(:Concept)

-- Technical Relationships
(:Component)-[:DEPENDS_ON]->(:Component)
(:Component)-[:CONFIGURED_BY]->(:Component)
(:Component)-[:IMPLEMENTS]->(:Component)

-- Business Relationships
(:Person)-[:MANAGES]->(:Process)
(:Person)-[:WORKS_FOR]->(:Organization)
(:Process)-[:RESPONSIBLE_FOR]->(:Component)

-- Universal Relationships
(:Document)-[:MENTIONS]->(Entity)
(Entity)-[:RELATED_TO]->(Entity)
```

**Index Strategy**:
```cypher
-- Primary indexes on ID fields
CREATE INDEX entity_id FOR (n) ON (n.id);
CREATE INDEX entity_name FOR (n) ON (n.name);

-- Composite indexes for common queries
CREATE INDEX document_domain FOR (d:Document) ON (d.domain);
CREATE INDEX person_org FOR (p:Person) ON (p.affiliation);
```

## NLP Pipeline

### spaCy Integration

**Model Requirements**:
- `en_core_web_sm`: English language model for basic NER
- Custom rule-based patterns for domain-specific entities
- Extensible for future model upgrades (Phase 2)

**Processing Pipeline**:
1. **Tokenization**: spaCy tokenizer with custom rules
2. **Named Entity Recognition**: Built-in NER + custom patterns
3. **Context Analysis**: Surrounding word analysis for disambiguation
4. **Entity Linking**: Cross-document entity resolution
5. **Validation**: Schema compliance checking

### Domain-Specific Patterns

**Research Entities**:
```python
# Academic titles and indicators
researcher_patterns = [
    "Dr.", "Prof.", "Professor", "Research", "University", "Institute",
    "Laboratory", "Department", "Faculty", "Scholar"
]

# Publication indicators
publication_patterns = [
    "paper", "journal", "conference", "proceedings", "article",
    "study", "research", "analysis", "investigation"
]
```

**Technical Entities**:
```python
# API and service patterns
api_patterns = [
    r'([A-Z][a-zA-Z]*API)',
    r'([a-z]+(?:_[a-z]+)*\.(js|py|java|cpp))',
    r'(microservice|service|component|module)'
]

# Technology stack patterns
tech_patterns = [
    "Redis", "MongoDB", "PostgreSQL", "Kafka", "Docker",
    "Kubernetes", "React", "Angular", "Spring", "Django"
]
```

## Performance Characteristics

### Throughput Metrics

**Document Processing**:
- Small documents (< 10KB): ~2-5 seconds per document
- Medium documents (10-100KB): ~5-15 seconds per document  
- Large documents (> 100KB): ~15-45 seconds per document

**Factors Affecting Performance**:
- Entity extraction complexity (domain-dependent)
- Neo4j connection latency
- ChromaDB embedding generation
- Document size and complexity
- Available system resources

### Storage Efficiency

**ChromaDB Storage**:
- Approximate 1MB per 1000 documents (embeddings only)
- Metadata adds ~10% overhead
- SQLite backend provides good compression

**Neo4j Storage**:
- Node overhead: ~100-200 bytes per entity
- Relationship overhead: ~50-100 bytes per relationship
- Index overhead: ~20% of total graph size
- Property storage varies by data type and length

### Query Performance

**Vector Search (ChromaDB)**:
- Similarity queries: < 100ms for collections up to 100K documents
- Metadata filtering adds ~10-20ms overhead
- Performance degrades linearly with collection size

**Graph Traversal (Neo4j)**:
- Single-hop queries: < 10ms for graphs up to 1M nodes
- Multi-hop queries: 50ms-2s depending on complexity
- Path queries scale with graph density

## Security Considerations

### Input Validation

**Cypher Injection Prevention**:
```python
# Always use parameterized queries
query = "MATCH (n:Person {id: $person_id}) RETURN n"
session.run(query, person_id=user_input)

# Never string interpolation
# BAD: query = f"MATCH (n:Person {{id: '{user_input}'}}) RETURN n"
```

**File Upload Security**:
- Restricted file extensions (TXT, MD, PDF, DOCX only)
- File size limits (configurable, default 10MB)
- Content type validation
- Virus scanning integration point (Phase 2)

### Access Control

**Database Security**:
- Neo4j authentication with username/password
- Network isolation using Docker networking
- Configurable connection timeouts
- Read-only query modes for certain operations

**Data Privacy**:
- No sensitive data extraction by default
- Configurable entity filtering
- Option to disable certain entity types
- Audit logging capabilities (Phase 2)

### Configuration Security

**Environment Variables**:
```bash
# Strong password requirements
NEO4J_PASSWORD=<complex-password>

# Network binding restrictions  
NEO4J_URI=bolt://localhost:7687  # Local only by default

# Feature toggles for security
ENABLE_GRAPH_FEATURES=true
ENABLE_FILE_UPLOADS=false
```

## Extension Points

### Phase 2 Integration Points

**Advanced NLP Models**:
- OpenNRE integration for relation extraction
- Custom transformer models for domain-specific NER
- Multilingual support expansion
- Confidence scoring enhancements

**Query Router Architecture**:
```python
# Future query routing interface
class QueryRouter:
    def route_query(self, query: str) -> QueryStrategy:
        # Analyze query complexity and type
        # Return VECTOR, GRAPH, or HYBRID strategy
        pass
```

**Enhanced Reasoning**:
- Graph neural network integration
- Reasoning chain validation
- Explanation generation
- Confidence propagation through graph paths

### Monitoring Integration

**Health Check Endpoints**:
```python
# Future health monitoring
{
    "chromadb_status": "healthy",
    "neo4j_status": "healthy", 
    "graph_node_count": 15432,
    "graph_relationship_count": 8765,
    "last_document_processed": "2024-01-15T10:30:00Z"
}
```

**Performance Metrics**:
- Document processing latency distributions
- Query response time percentiles
- Storage utilization trends
- Error rate monitoring

### Custom Domain Support

**Domain Registration**:
```python
# Future extensible domain system
@register_domain("legal")
class LegalDomain:
    entity_types = ["Case", "Statute", "Court", "Judge"]
    extraction_patterns = [...]
    relationship_types = ["CITES", "OVERRULES", "DISTINGUISHES"]
```

This architecture provides a solid foundation for Phase 1 while maintaining clear extension points for Phase 2 enhancements.