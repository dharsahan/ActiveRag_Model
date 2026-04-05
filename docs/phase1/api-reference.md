# Phase 1 API Reference

This document provides comprehensive API documentation for all public classes and methods in the Phase 1 hybrid vector-graph RAG system.

## Table of Contents
- [Document Processing](#document-processing)
- [Storage Management](#storage-management)  
- [Knowledge Graph Operations](#knowledge-graph-operations)
- [NLP Pipeline](#nlp-pipeline)
- [Configuration](#configuration)
- [Schemas and Types](#schemas-and-types)
- [Error Handling](#error-handling)
- [Usage Examples](#usage-examples)

## Document Processing

### DocumentLoader

**File**: `active_rag/document_loader.py`

Main entry point for document ingestion with dual storage support.

#### Class: `DocumentLoader`

```python
class DocumentLoader:
    """Loads documents from local files for vector store and graph ingestion."""
    
    def __init__(self, config: Optional[Config] = None)
```

**Parameters**:
- `config` (Optional[Config]): System configuration. Uses default Config() if not provided.

**Attributes**:
- `config`: System configuration object
- `dual_storage`: DualStorageManager instance (None if graph features disabled)

#### Methods

##### `load(path: str) -> List[LoadedDocument]`

Load a document from a file path and return parsed content.

**Parameters**:
- `path` (str): Path to the document file

**Returns**:
- `List[LoadedDocument]`: List of parsed document objects

**Raises**:
- `FileNotFoundError`: If the specified file doesn't exist
- `ValueError`: If file type is not supported

**Supported formats**: `.txt`, `.md`, `.pdf`, `.docx`

**Example**:
```python
loader = DocumentLoader()
documents = loader.load('/path/to/document.txt')
for doc in documents:
    print(f"Title: {doc.title}")
    print(f"Content length: {len(doc.content)}")
    print(f"Word count: {doc.word_count}")
```

##### `load_and_store(path: str, domain: Optional[ContentDomain] = None) -> Dict[str, Any]`

Load document and store in both ChromaDB and Neo4j (if enabled).

**Parameters**:
- `path` (str): Path to the document file
- `domain` (Optional[ContentDomain]): Content domain for entity extraction. Auto-classified if not provided.

**Returns**:
- `Dict[str, Any]`: Storage results with the following structure:
  ```python
  {
      "documents_processed": int,
      "storage_results": [
          {
              "doc_id": str,
              "chroma_stored": bool,
              "graph_stored": bool, 
              "entities_extracted": List[Dict[str, Any]],
              "relationships_created": List[Dict[str, Any]]
          }
      ]
  }
  ```

**Example**:
```python
from active_rag.schemas.entities import ContentDomain

loader = DocumentLoader()
result = loader.load_and_store('research_paper.pdf', ContentDomain.RESEARCH)

print(f"Documents processed: {result['documents_processed']}")
storage_result = result['storage_results'][0]
print(f"Entities found: {len(storage_result['entities_extracted'])}")
print(f"Graph stored: {storage_result['graph_stored']}")
```

#### Class: `LoadedDocument`

```python
@dataclass
class LoadedDocument:
    """A document loaded from a local file."""
    content: str        # Full text content
    source: str         # Source file path or URL
    title: str = ""     # Document title (defaults to filename)
    word_count: int = 0 # Number of words in content
```

## Storage Management

### DualStorageManager

**File**: `active_rag/storage/dual_storage_manager.py`

Manages storage operations across both ChromaDB and Neo4j with entity extraction.

#### Class: `DualStorageManager`

```python
class DualStorageManager:
    """Manages storage operations across both ChromaDB and Neo4j with entity extraction"""
    
    def __init__(self, config: Config)
```

**Parameters**:
- `config` (Config): System configuration with database settings

**Attributes**:
- `config`: System configuration
- `chroma_client`: ChromaDB client instance
- `collection`: ChromaDB collection for documents
- `neo4j_client`: Neo4j client (None if graph features disabled)
- `schema_manager`: Graph schema manager
- `entity_extractor`: NLP entity extractor
- `document_classifier`: Document domain classifier

#### Methods

##### `store_document(doc_data: Dict[str, Any]) -> Dict[str, Any]`

Store document in both ChromaDB and Neo4j with entity extraction.

**Parameters**:
- `doc_data` (Dict[str, Any]): Document data with required keys:
  - `"title"` (str): Document title
  - `"content"` (str): Document content
  - `"url"` (Optional[str]): Source URL or file path
  - `"domain"` (Optional[ContentDomain]): Content domain (auto-classified if missing)

**Returns**:
- `Dict[str, Any]`: Detailed storage results:
  ```python
  {
      "doc_id": str,                    # Generated document ID
      "chroma_stored": bool,            # ChromaDB storage success
      "graph_stored": bool,             # Neo4j storage success  
      "entities_extracted": [           # Extracted entities
          {
              "label": str,             # Entity type (Person, Organization, etc.)
              "properties": {           # Entity properties
                  "id": str,            # Unique entity ID
                  "name": str,          # Entity name
                  # ... other domain-specific properties
              }
          }
      ],
      "relationships_created": [        # Created relationships
          {
              "from": str,              # Source entity/document ID
              "to": str,                # Target entity ID  
              "type": str               # Relationship type
          }
      ]
  }
  ```

**Example**:
```python
config = Config()
dual_storage = DualStorageManager(config)

doc_data = {
    "title": "AI Research Survey",
    "content": "This paper reviews recent advances in artificial intelligence...",
    "url": "https://example.com/ai-survey.pdf",
    "domain": ContentDomain.RESEARCH
}

result = dual_storage.store_document(doc_data)
print(f"Document ID: {result['doc_id']}")
print(f"Entities: {len(result['entities_extracted'])}")
```

##### `get_document_entities(doc_id: str) -> List[Dict[str, Any]]`

Get all entities mentioned in a specific document.

**Parameters**:
- `doc_id` (str): Document identifier

**Returns**:
- `List[Dict[str, Any]]`: List of entities with their properties and labels

**Example**:
```python
entities = dual_storage.get_document_entities("doc_123")
for entity in entities:
    print(f"Entity: {entity['name']} ({entity['labels'][0]})")
```

##### `close()`

Close database connections gracefully.

**Example**:
```python
dual_storage.close()
```

## Knowledge Graph Operations

### GraphOperations

**File**: `active_rag/knowledge_graph/graph_operations.py`

High-level graph operations for querying and reasoning.

#### Class: `GraphOperations`

```python
class GraphOperations:
    """High-level graph operations for querying and reasoning"""
    
    def __init__(self, client: Neo4jClient)
```

**Parameters**:
- `client` (Neo4jClient): Initialized Neo4j client

#### Methods

##### `search_entities_by_name(name_pattern: str, entity_types: Optional[List[str]] = None) -> List[Dict[str, Any]]`

Search for entities by name pattern.

**Parameters**:
- `name_pattern` (str): Name pattern to search for (case-insensitive, partial matching)
- `entity_types` (Optional[List[str]]): Filter by entity types (e.g., ["Person", "Organization"])

**Returns**:
- `List[Dict[str, Any]]`: Matching entities with properties and labels

**Example**:
```python
from active_rag.knowledge_graph.graph_operations import GraphOperations

graph_ops = GraphOperations(neo4j_client)

# Find all entities containing "Einstein"
entities = graph_ops.search_entities_by_name("Einstein")

# Find only Person entities containing "Marie"
persons = graph_ops.search_entities_by_name("Marie", ["Person"])
```

##### `find_related_entities(entity_id: str, relationship_types: Optional[List[str]] = None, depth: int = 1) -> List[Dict[str, Any]]`

Find entities related to a starting entity.

**Parameters**:
- `entity_id` (str): Starting entity ID
- `relationship_types` (Optional[List[str]]): Filter by relationship types
- `depth` (int): Traversal depth (1-10, default: 1)

**Returns**:
- `List[Dict[str, Any]]`: Related entities with relationship information

**Example**:
```python
# Find entities directly connected to Einstein
related = graph_ops.find_related_entities("person_einstein")

# Find entities connected through AUTHORED relationships within 2 hops
related = graph_ops.find_related_entities(
    "person_einstein", 
    relationship_types=["AUTHORED"], 
    depth=2
)
```

##### `find_paths(start_id: str, end_id: str, max_depth: int = 3) -> List[Dict[str, Any]]`

Find paths between two entities.

**Parameters**:
- `start_id` (str): Starting entity ID
- `end_id` (str): Target entity ID  
- `max_depth` (int): Maximum path length (1-10, default: 3)

**Returns**:
- `List[Dict[str, Any]]`: Paths with detailed route information:
  ```python
  [
      {
          "length": int,                    # Path length
          "relationship_types": List[str],  # Relationship types in path
          "nodes": List[Dict],              # Nodes in path order
          "reasoning_path": str             # Human-readable path description
      }
  ]
  ```

**Example**:
```python
paths = graph_ops.find_paths("person_einstein", "org_princeton", max_depth=2)
for path in paths:
    print(f"Path length: {path['length']}")
    print(f"Path: {path['reasoning_path']}")
```

##### `multi_hop_query(query_text: str, max_hops: int = 2) -> Dict[str, Any]`

Execute multi-hop reasoning query using NLP to extract entities.

**Parameters**:
- `query_text` (str): Natural language query
- `max_hops` (int): Maximum traversal depth (1-10, default: 2)

**Returns**:
- `Dict[str, Any]`: Query results with entities, paths, and reasoning:
  ```python
  {
      "entities": List[Dict[str, Any]],     # Relevant entities found
      "paths": List[Dict[str, Any]],        # Connection paths
      "reasoning": str                       # Explanation of results
  }
  ```

**Example**:
```python
result = graph_ops.multi_hop_query("Who worked with Einstein on relativity?")
print(f"Found {len(result['entities'])} relevant entities")
print(f"Reasoning: {result['reasoning']}")

for entity in result['entities']:
    print(f"- {entity['name']} (relevance: {entity.get('relevance_score', 0):.2f})")
```

##### `get_entity_neighborhood(entity_id: str, radius: int = 2) -> List[Dict[str, Any]]`

Get the neighborhood of entities around a given entity.

**Parameters**:
- `entity_id` (str): Center entity ID
- `radius` (int): Neighborhood radius (1-10, default: 2)

**Returns**:
- `List[Dict[str, Any]]`: Neighboring entities with distance information

**Example**:
```python
neighborhood = graph_ops.get_entity_neighborhood("person_einstein", radius=3)
for neighbor in neighborhood:
    print(f"{neighbor['name']} (distance: {neighbor['distance']})")
```

##### `get_graph_stats() -> Dict[str, Any]`

Get basic statistics about the graph.

**Returns**:
- `Dict[str, Any]`: Graph statistics:
  ```python
  {
      "total_nodes": int,                   # Total number of nodes
      "total_relationships": int,           # Total number of relationships
      "node_types": List[str],              # Available node types
      "relationship_types": List[str]       # Available relationship types
  }
  ```

**Example**:
```python
stats = graph_ops.get_graph_stats()
print(f"Graph contains {stats['total_nodes']} nodes and {stats['total_relationships']} relationships")
print(f"Node types: {', '.join(stats['node_types'])}")
```

### Neo4jClient

**File**: `active_rag/knowledge_graph/neo4j_client.py`

Low-level Neo4j database interface.

#### Class: `Neo4jClient`

```python
class Neo4jClient:
    """Neo4j database client with connection management"""
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j")
```

**Parameters**:
- `uri` (str): Neo4j connection URI (e.g., "bolt://localhost:7687")
- `username` (str): Neo4j username
- `password` (str): Neo4j password
- `database` (str): Database name (default: "neo4j")

#### Methods

##### `create_entity(label: str, properties: Dict[str, Any]) -> Dict[str, Any]`

Create a new entity in the graph.

**Parameters**:
- `label` (str): Entity type label
- `properties` (Dict[str, Any]): Entity properties

**Returns**:
- `Dict[str, Any]`: Created entity properties

**Example**:
```python
client = Neo4jClient("bolt://localhost:7687", "neo4j", "password")
entity = client.create_entity("Person", {
    "id": "person_123",
    "name": "Albert Einstein",
    "affiliation": "Princeton University"
})
```

##### `get_entity(label: str, entity_id: str) -> Optional[Dict[str, Any]]`

Retrieve an entity by ID.

**Parameters**:
- `label` (str): Entity type label  
- `entity_id` (str): Entity ID

**Returns**:
- `Optional[Dict[str, Any]]`: Entity properties or None if not found

##### `create_relationship(from_label: str, from_id: str, to_label: str, to_id: str, relationship_type: str, properties: Optional[Dict[str, Any]] = None) -> bool`

Create a relationship between two entities.

**Parameters**:
- `from_label` (str): Source entity label
- `from_id` (str): Source entity ID
- `to_label` (str): Target entity label  
- `to_id` (str): Target entity ID
- `relationship_type` (str): Relationship type
- `properties` (Optional[Dict[str, Any]]): Relationship properties

**Returns**:
- `bool`: Success status

##### `close()`

Close the database connection.

## NLP Pipeline

### DocumentClassifier

**File**: `active_rag/nlp_pipeline/document_classifier.py`

Classifies documents into content domains.

#### Class: `DocumentClassifier`

```python
class DocumentClassifier:
    """Classifies documents into content domains for entity extraction"""
    
    def __init__(self)
```

#### Methods

##### `classify_document(text: str) -> ContentDomain`

Classify document into one of the four content domains.

**Parameters**:
- `text` (str): Document text content

**Returns**:
- `ContentDomain`: Classified domain (RESEARCH, TECHNICAL, BUSINESS, or MIXED_WEB)

**Example**:
```python
from active_rag.nlp_pipeline.document_classifier import DocumentClassifier
from active_rag.schemas.entities import ContentDomain

classifier = DocumentClassifier()

# Research document
research_text = "This paper presents a novel approach to quantum computing..."
domain = classifier.classify_document(research_text)
assert domain == ContentDomain.RESEARCH

# Technical document  
tech_text = "The UserService API provides authentication endpoints..."
domain = classifier.classify_document(tech_text)
assert domain == ContentDomain.TECHNICAL
```

### EntityExtractor

**File**: `active_rag/nlp_pipeline/entity_extractor.py`

Extracts entities from text based on content domain.

#### Class: `EntityExtractor`

```python
class EntityExtractor:
    """Extracts entities from text based on content domain"""
    
    def __init__(self)
```

#### Methods

##### `extract_entities(text: str, domain: ContentDomain) -> List[Dict[str, Any]]`

Extract entities from text based on content domain.

**Parameters**:
- `text` (str): Text content to analyze
- `domain` (ContentDomain): Content domain for specialized extraction

**Returns**:
- `List[Dict[str, Any]]`: Extracted entities:
  ```python
  [
      {
          "label": str,                 # Entity type (Person, Organization, etc.)
          "properties": {               # Entity properties
              "id": str,                # Unique identifier  
              "name": str,              # Entity name
              # ... additional domain-specific properties
          }
      }
  ]
  ```

**Example**:
```python
from active_rag.nlp_pipeline.entity_extractor import EntityExtractor
from active_rag.schemas.entities import ContentDomain

extractor = EntityExtractor()

text = "Dr. Einstein worked at Princeton University on quantum mechanics."
entities = extractor.extract_entities(text, ContentDomain.RESEARCH)

for entity in entities:
    print(f"Found {entity['label']}: {entity['properties']['name']}")
    # Output: Found Person: Einstein
    #         Found Organization: Princeton University
    #         Found Concept: quantum mechanics
```

## Configuration

### Config

**File**: `active_rag/config.py`

Centralized configuration management.

#### Class: `Config`

```python
@dataclass
class Config:
    """Configuration for the Active RAG pipeline."""
```

#### Configuration Parameters

**LLM Settings**:
- `provider` (str): LLM provider ("local", "openai", etc.)
- `ollama_base_url` (str): Ollama server URL
- `model_name` (str): Model name to use
- `api_key` (str): API key for external providers

**Vector Store Settings**:
- `chroma_persist_dir` (str): ChromaDB persistence directory
- `collection_name` (str): ChromaDB collection name
- `top_k` (int): Number of top results to retrieve
- `confidence_threshold` (float): Confidence threshold for responses

**Neo4j Settings**:
- `neo4j_uri` (str): Neo4j connection URI
- `neo4j_username` (str): Neo4j username  
- `neo4j_password` (str): Neo4j password

**Feature Toggles**:
- `enable_graph_features` (bool): Enable graph storage and operations
- `enable_relation_extraction` (bool): Enable relationship extraction
- `max_graph_hops` (int): Maximum graph traversal depth

**NLP Settings**:
- `spacy_model` (str): spaCy model name
- `time_sensitive_max_age` (int): Time sensitivity for documents
- `max_search_results` (int): Maximum web search results

#### Environment Variables

All configuration can be set via environment variables:

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j  
NEO4J_PASSWORD=activerag123

# Feature Toggles
ENABLE_GRAPH_FEATURES=true
ENABLE_RELATION_EXTRACTION=true

# ChromaDB
CHROMA_PERSIST_DIR=./chroma_db
COLLECTION_NAME=active_rag

# Performance
TOP_K=5
MAX_GRAPH_HOPS=3
CONFIDENCE_THRESHOLD=0.7
```

## Schemas and Types

### ContentDomain

**File**: `active_rag/schemas/entities.py`

Content domain enumeration for document classification.

```python
class ContentDomain(Enum):
    """Content domains for entity classification"""
    RESEARCH = "research"      # Academic papers, research documents
    TECHNICAL = "technical"    # API docs, technical specifications  
    BUSINESS = "business"      # Business processes, workflows
    MIXED_WEB = "mixed_web"   # General web content
```

### EntitySchema

```python
@dataclass
class EntitySchema:
    """Schema definition for graph entities"""
    label: str                          # Entity type label
    required_properties: List[str]      # Required property names
    optional_properties: List[str]      # Optional property names  
    domain: ContentDomain              # Associated content domain
    description: str = ""              # Human-readable description
```

### Available Entity Schemas

```python
# Research Domain
PERSON_SCHEMA = EntitySchema(
    label="Person",
    required_properties=["name", "id"],
    optional_properties=["affiliation", "email", "orcid"],
    domain=ContentDomain.RESEARCH
)

ORGANIZATION_SCHEMA = EntitySchema(
    label="Organization", 
    required_properties=["name", "id"],
    optional_properties=["type", "location", "website"],
    domain=ContentDomain.RESEARCH
)

CONCEPT_SCHEMA = EntitySchema(
    label="Concept",
    required_properties=["name", "id"], 
    optional_properties=["definition", "domain", "aliases"],
    domain=ContentDomain.RESEARCH
)

# Technical Domain
COMPONENT_SCHEMA = EntitySchema(
    label="Component",
    required_properties=["name", "id"],
    optional_properties=["version", "type", "description"], 
    domain=ContentDomain.TECHNICAL
)

# Business Domain  
PROCESS_SCHEMA = EntitySchema(
    label="Process",
    required_properties=["name", "id"],
    optional_properties=["description", "owner", "status"],
    domain=ContentDomain.BUSINESS
)

# Universal
DOCUMENT_SCHEMA = EntitySchema(
    label="Document",
    required_properties=["title", "id", "content_hash"],
    optional_properties=["url", "type", "domain", "created_at"],
    domain=ContentDomain.MIXED_WEB
)
```

### Schema Utility Functions

```python
def get_entity_schema(label: str) -> Optional[EntitySchema]:
    """Get entity schema by label"""

def get_entities_by_domain(domain: ContentDomain) -> List[EntitySchema]:
    """Get all entity schemas for a specific domain"""
    
def list_entity_types() -> List[str]:
    """Get list of all available entity types"""
```

## Error Handling

### Common Exceptions

#### Document Processing Errors

```python
# File not found
try:
    documents = loader.load('/nonexistent/file.txt')
except FileNotFoundError as e:
    print(f"File not found: {e}")

# Unsupported file type
try:
    documents = loader.load('/path/to/file.xyz')
except ValueError as e:
    print(f"Unsupported file type: {e}")
```

#### Database Connection Errors

```python
# Neo4j connection failure
try:
    client = Neo4jClient("bolt://localhost:7687", "neo4j", "wrong_password")
except Exception as e:
    print(f"Neo4j connection failed: {e}")

# ChromaDB initialization failure
try:
    dual_storage = DualStorageManager(config)
except Exception as e:
    print(f"Storage initialization failed: {e}")
```

#### NLP Model Errors

```python
# Missing spaCy model
try:
    extractor = EntityExtractor()
except ValueError as e:
    print(f"spaCy model not found: {e}")
    print("Run: python -m spacy download en_core_web_sm")
```

#### Graph Query Errors

```python
# Invalid entity types in graph operations
try:
    entities = graph_ops.search_entities_by_name("test", ["InvalidType"])
except ValueError as e:
    print(f"Invalid entity type: {e}")

# Invalid traversal depth
try:
    related = graph_ops.find_related_entities("entity_123", depth=15)
except ValueError as e:
    print(f"Invalid depth parameter: {e}")
```

## Usage Examples

### Complete Document Processing Workflow

```python
from active_rag.document_loader import DocumentLoader
from active_rag.config import Config
from active_rag.schemas.entities import ContentDomain
from active_rag.knowledge_graph.graph_operations import GraphOperations

# Initialize system
config = Config()
loader = DocumentLoader(config)

# Process multiple documents
documents = [
    ("/docs/research_paper.pdf", ContentDomain.RESEARCH),
    ("/docs/api_reference.md", ContentDomain.TECHNICAL),
    ("/docs/business_process.docx", ContentDomain.BUSINESS)
]

for doc_path, domain in documents:
    try:
        result = loader.load_and_store(doc_path, domain)
        print(f"Processed {doc_path}:")
        print(f"  Entities: {len(result['storage_results'][0]['entities_extracted'])}")
        print(f"  Graph stored: {result['storage_results'][0]['graph_stored']}")
    except Exception as e:
        print(f"Error processing {doc_path}: {e}")

# Query the knowledge graph
if loader.dual_storage and loader.dual_storage.neo4j_client:
    graph_ops = GraphOperations(loader.dual_storage.neo4j_client)
    
    # Get graph statistics
    stats = graph_ops.get_graph_stats()
    print(f"\nGraph Statistics:")
    print(f"  Nodes: {stats['total_nodes']}")
    print(f"  Relationships: {stats['total_relationships']}")
    print(f"  Node types: {', '.join(stats['node_types'])}")
    
    # Search for entities
    people = graph_ops.search_entities_by_name("Einstein", ["Person"])
    for person in people:
        print(f"Found person: {person['name']}")
        
        # Find related entities
        related = graph_ops.find_related_entities(person['id'], depth=2)
        print(f"  Related entities: {len(related)}")

# Cleanup
loader.dual_storage.close()
```

### Custom Entity Extraction

```python
from active_rag.nlp_pipeline.entity_extractor import EntityExtractor
from active_rag.nlp_pipeline.document_classifier import DocumentClassifier
from active_rag.schemas.entities import ContentDomain

# Initialize components
classifier = DocumentClassifier()
extractor = EntityExtractor()

# Process custom text
text = """
Dr. Sarah Johnson from MIT published a paper on quantum computing 
algorithms in Nature journal. Her research builds upon work by 
Professor Wang at Stanford University.
"""

# Classify domain
domain = classifier.classify_document(text)
print(f"Classified domain: {domain.value}")

# Extract entities
entities = extractor.extract_entities(text, domain)

print("\nExtracted entities:")
for entity in entities:
    print(f"  {entity['label']}: {entity['properties']['name']}")
    if 'affiliation' in entity['properties']:
        print(f"    Affiliation: {entity['properties']['affiliation']}")
```

### Hybrid Query Processing

```python
from active_rag.storage.dual_storage_manager import DualStorageManager
from active_rag.knowledge_graph.graph_operations import GraphOperations
from active_rag.config import Config

# Initialize hybrid storage
config = Config()
dual_storage = DualStorageManager(config)
graph_ops = GraphOperations(dual_storage.neo4j_client)

# Vector similarity search
vector_query = "quantum computing research"
vector_results = dual_storage.collection.query(
    query_texts=[vector_query],
    n_results=5,
    where={"domain": "research"}  # Filter by domain
)

print("Vector search results:")
for i, doc_id in enumerate(vector_results['ids'][0]):
    metadata = vector_results['metadatas'][0][i]
    print(f"  Document: {metadata['title']}")
    
    # Get entities from this document
    entities = dual_storage.get_document_entities(doc_id)
    print(f"    Entities: {len(entities)}")

# Graph reasoning query  
reasoning_query = "Who worked on quantum computing?"
reasoning_result = graph_ops.multi_hop_query(reasoning_query, max_hops=2)

print(f"\nReasoning results:")
print(f"  Found {len(reasoning_result['entities'])} entities")
print(f"  Found {len(reasoning_result['paths'])} connection paths")
print(f"  Reasoning: {reasoning_result['reasoning']}")

# Display top entities by relevance
for entity in reasoning_result['entities'][:3]:
    relevance = entity.get('relevance_score', 0)
    print(f"  {entity['name']} (relevance: {relevance:.2f})")

dual_storage.close()
```

This API reference provides comprehensive documentation for integrating with and extending the Phase 1 hybrid vector-graph RAG system.