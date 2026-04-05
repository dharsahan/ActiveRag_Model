# Hybrid Vector-Graph RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform existing Active RAG system from ChromaDB-only to hybrid vector-graph architecture with Neo4j knowledge graph for multi-hop reasoning and explainable answers.

**Architecture:** Dual storage with intelligent routing between vector similarity, graph traversal, and hybrid retrieval. Enhanced NLP pipeline extracts entities/relationships for four content domains (research papers, technical docs, business knowledge, mixed web content).

**Tech Stack:** Neo4j Community, spaCy + Transformers, OpenNRE, ChromaDB (existing), NetworkX, Python 3.10+

---

## File Structure Overview

**New Files to Create:**
- `active_rag/knowledge_graph/` - Graph database interface and operations
- `active_rag/nlp_pipeline/` - Enhanced NLP processing components  
- `active_rag/routing/` - Intelligent query routing logic
- `active_rag/schemas/` - Entity and relationship definitions
- `tests/knowledge_graph/` - Graph-specific test modules
- `requirements_graph.txt` - New dependencies for graph functionality
- `docker-compose.neo4j.yml` - Neo4j container configuration

**Files to Modify:**
- `active_rag/agent.py:45-85` - Add graph routing integration
- `active_rag/document_loader.py:120-180` - Add dual storage writes
- `active_rag/config.py:15-35` - Add Neo4j configuration
- `requirements.txt` - Add core graph dependencies

---

## Phase 1: Foundation Infrastructure (Tasks 1-8)

### Task 1: Neo4j Database Setup

**Files:**
- Create: `docker-compose.neo4j.yml`
- Create: `active_rag/knowledge_graph/__init__.py`
- Create: `active_rag/knowledge_graph/neo4j_client.py`
- Create: `tests/knowledge_graph/test_neo4j_client.py`

- [ ] **Step 1: Write failing Neo4j client test**

```python
# tests/knowledge_graph/test_neo4j_client.py
import pytest
from active_rag.knowledge_graph.neo4j_client import Neo4jClient

def test_neo4j_connection():
    client = Neo4jClient("bolt://localhost:7687", "neo4j", "password")
    assert client.is_connected() == True
    
def test_create_entity():
    client = Neo4jClient("bolt://localhost:7687", "neo4j", "password")
    result = client.create_entity("Person", {"name": "Test Person", "id": "test_1"})
    assert result["id"] == "test_1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/knowledge_graph/test_neo4j_client.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'active_rag.knowledge_graph.neo4j_client'"

- [ ] **Step 3: Create Docker Compose for Neo4j**

```yaml
# docker-compose.neo4j.yml
version: '3.8'
services:
  neo4j:
    image: neo4j:5.15-community
    container_name: active_rag_neo4j
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=neo4j/activerag123
      - NEO4J_PLUGINS=["apoc"]
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_import:/var/lib/neo4j/import
      - neo4j_plugins:/plugins
    restart: unless-stopped

volumes:
  neo4j_data:
  neo4j_logs:
  neo4j_import:
  neo4j_plugins:
```

- [ ] **Step 4: Create Neo4j client class**

```python
# active_rag/knowledge_graph/neo4j_client.py
from neo4j import GraphDatabase
import logging
from typing import Dict, List, Any, Optional

class Neo4jClient:
    def __init__(self, uri: str, username: str, password: str):
        self.uri = uri
        self.username = username
        self.password = password
        self._driver = None
        self._connect()
    
    def _connect(self):
        try:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
            # Test connection
            with self._driver.session() as session:
                session.run("RETURN 1")
            logging.info("Connected to Neo4j database")
        except Exception as e:
            logging.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def is_connected(self) -> bool:
        if not self._driver:
            return False
        try:
            with self._driver.session() as session:
                session.run("RETURN 1")
            return True
        except:
            return False
    
    def create_entity(self, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        query = f"CREATE (n:{label} $props) RETURN n"
        with self._driver.session() as session:
            result = session.run(query, props=properties)
            record = result.single()
            return dict(record["n"])
    
    def close(self):
        if self._driver:
            self._driver.close()
```

- [ ] **Step 5: Create knowledge_graph module init**

```python
# active_rag/knowledge_graph/__init__.py
from .neo4j_client import Neo4jClient

__all__ = ['Neo4jClient']
```

- [ ] **Step 6: Start Neo4j container**

Run: `docker-compose -f docker-compose.neo4j.yml up -d`
Expected: Neo4j container running on ports 7474/7687

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/knowledge_graph/test_neo4j_client.py -v`
Expected: PASS (both connection and entity creation tests)

- [ ] **Step 8: Commit foundation**

```bash
git add docker-compose.neo4j.yml active_rag/knowledge_graph/ tests/knowledge_graph/
git commit -m "feat: add Neo4j foundation with Docker setup and client class"
```

### Task 2: Graph Schema Definition

**Files:**
- Create: `active_rag/schemas/__init__.py`
- Create: `active_rag/schemas/entities.py`
- Create: `active_rag/schemas/relationships.py`
- Create: `active_rag/knowledge_graph/schema_manager.py`
- Create: `tests/schemas/test_schema_manager.py`

- [ ] **Step 1: Write failing schema manager test**

```python
# tests/schemas/test_schema_manager.py
import pytest
from active_rag.knowledge_graph.schema_manager import SchemaManager
from active_rag.knowledge_graph.neo4j_client import Neo4jClient

def test_create_constraints():
    client = Neo4jClient("bolt://localhost:7687", "neo4j", "activerag123")
    schema = SchemaManager(client)
    result = schema.create_base_constraints()
    assert result == True

def test_entity_validation():
    client = Neo4jClient("bolt://localhost:7687", "neo4j", "activerag123")
    schema = SchemaManager(client)
    valid = schema.validate_entity("Person", {"name": "John", "id": "p1"})
    assert valid == True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/schemas/test_schema_manager.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'active_rag.knowledge_graph.schema_manager'"

- [ ] **Step 3: Define entity schemas**

```python
# active_rag/schemas/entities.py
from dataclasses import dataclass
from typing import Dict, Any, List
from enum import Enum

class ContentDomain(Enum):
    RESEARCH = "research"
    TECHNICAL = "technical"
    BUSINESS = "business"
    MIXED_WEB = "mixed_web"

@dataclass
class EntitySchema:
    label: str
    required_properties: List[str]
    optional_properties: List[str]
    domain: ContentDomain

# Research domain entities
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

# Technical domain entities
COMPONENT_SCHEMA = EntitySchema(
    label="Component",
    required_properties=["name", "id"],
    optional_properties=["version", "type", "description"],
    domain=ContentDomain.TECHNICAL
)

# Business domain entities  
PROCESS_SCHEMA = EntitySchema(
    label="Process",
    required_properties=["name", "id"],
    optional_properties=["description", "owner", "status"],
    domain=ContentDomain.BUSINESS
)

# Document entity (cross-domain)
DOCUMENT_SCHEMA = EntitySchema(
    label="Document",
    required_properties=["title", "id", "content_hash"],
    optional_properties=["url", "type", "domain", "created_at"],
    domain=ContentDomain.MIXED_WEB
)

ENTITY_SCHEMAS = {
    "Person": PERSON_SCHEMA,
    "Organization": ORGANIZATION_SCHEMA,
    "Concept": CONCEPT_SCHEMA,
    "Component": COMPONENT_SCHEMA,
    "Process": PROCESS_SCHEMA,
    "Document": DOCUMENT_SCHEMA
}
```

- [ ] **Step 4: Define relationship schemas**

```python
# active_rag/schemas/relationships.py
from dataclasses import dataclass
from typing import List
from .entities import ContentDomain

@dataclass
class RelationshipSchema:
    type: str
    from_labels: List[str]
    to_labels: List[str]
    required_properties: List[str]
    optional_properties: List[str]
    domain: ContentDomain

# Research domain relationships
AUTHORED_REL = RelationshipSchema(
    type="AUTHORED",
    from_labels=["Person"],
    to_labels=["Document"],
    required_properties=[],
    optional_properties=["year", "role"],
    domain=ContentDomain.RESEARCH
)

AFFILIATED_WITH_REL = RelationshipSchema(
    type="AFFILIATED_WITH",
    from_labels=["Person"],
    to_labels=["Organization"],
    required_properties=[],
    optional_properties=["start_year", "end_year", "role"],
    domain=ContentDomain.RESEARCH
)

# Technical domain relationships
DEPENDS_ON_REL = RelationshipSchema(
    type="DEPENDS_ON",
    from_labels=["Component"],
    to_labels=["Component"],
    required_properties=[],
    optional_properties=["version_constraint", "dependency_type"],
    domain=ContentDomain.TECHNICAL
)

# Business domain relationships
MANAGES_REL = RelationshipSchema(
    type="MANAGES",
    from_labels=["Person"],
    to_labels=["Person", "Process"],
    required_properties=[],
    optional_properties=["since", "responsibility_level"],
    domain=ContentDomain.BUSINESS
)

# Cross-domain relationships
MENTIONS_REL = RelationshipSchema(
    type="MENTIONS",
    from_labels=["Document"],
    to_labels=["Person", "Organization", "Concept", "Component"],
    required_properties=[],
    optional_properties=["context", "sentiment", "confidence"],
    domain=ContentDomain.MIXED_WEB
)

RELATIONSHIP_SCHEMAS = {
    "AUTHORED": AUTHORED_REL,
    "AFFILIATED_WITH": AFFILIATED_WITH_REL,
    "DEPENDS_ON": DEPENDS_ON_REL,
    "MANAGES": MANAGES_REL,
    "MENTIONS": MENTIONS_REL
}
```

- [ ] **Step 5: Create schema manager**

```python
# active_rag/knowledge_graph/schema_manager.py
from typing import Dict, Any, List
import logging
from ..schemas.entities import ENTITY_SCHEMAS, EntitySchema
from ..schemas.relationships import RELATIONSHIP_SCHEMAS, RelationshipSchema
from .neo4j_client import Neo4jClient

class SchemaManager:
    def __init__(self, client: Neo4jClient):
        self.client = client
        
    def create_base_constraints(self) -> bool:
        """Create unique constraints and indexes for entity IDs"""
        constraints = [
            "CREATE CONSTRAINT entity_id FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT person_id FOR (p:Person) REQUIRE p.id IS UNIQUE", 
            "CREATE CONSTRAINT org_id FOR (o:Organization) REQUIRE o.id IS UNIQUE",
            "CREATE CONSTRAINT concept_id FOR (c:Concept) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT component_id FOR (c:Component) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT process_id FOR (p:Process) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT document_id FOR (d:Document) REQUIRE d.id IS UNIQUE"
        ]
        
        try:
            with self.client._driver.session() as session:
                for constraint in constraints:
                    try:
                        session.run(constraint)
                        logging.info(f"Created constraint: {constraint}")
                    except Exception as e:
                        # Constraint may already exist
                        if "already exists" not in str(e):
                            logging.warning(f"Failed to create constraint: {e}")
            return True
        except Exception as e:
            logging.error(f"Failed to create constraints: {e}")
            return False
    
    def validate_entity(self, label: str, properties: Dict[str, Any]) -> bool:
        """Validate entity properties against schema"""
        if label not in ENTITY_SCHEMAS:
            return False
            
        schema = ENTITY_SCHEMAS[label]
        
        # Check required properties
        for prop in schema.required_properties:
            if prop not in properties:
                return False
                
        return True
    
    def validate_relationship(self, rel_type: str, from_label: str, to_label: str, properties: Dict[str, Any] = None) -> bool:
        """Validate relationship against schema"""
        if rel_type not in RELATIONSHIP_SCHEMAS:
            return False
            
        schema = RELATIONSHIP_SCHEMAS[rel_type]
        
        # Check valid label combinations
        if from_label not in schema.from_labels or to_label not in schema.to_labels:
            return False
            
        if properties:
            # Check required properties
            for prop in schema.required_properties:
                if prop not in properties:
                    return False
                    
        return True
```

- [ ] **Step 6: Create schemas module init**

```python
# active_rag/schemas/__init__.py
from .entities import ENTITY_SCHEMAS, EntitySchema, ContentDomain
from .relationships import RELATIONSHIP_SCHEMAS, RelationshipSchema

__all__ = ['ENTITY_SCHEMAS', 'EntitySchema', 'ContentDomain', 'RELATIONSHIP_SCHEMAS', 'RelationshipSchema']
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/schemas/test_schema_manager.py -v`
Expected: PASS (constraints created and entity validation working)

- [ ] **Step 8: Commit schema foundation**

```bash
git add active_rag/schemas/ active_rag/knowledge_graph/schema_manager.py tests/schemas/
git commit -m "feat: add graph schema definitions and validation for all content domains"
```

### Task 3: Enhanced NLP Pipeline Foundation

**Files:**
- Create: `active_rag/nlp_pipeline/__init__.py`
- Create: `active_rag/nlp_pipeline/entity_extractor.py`
- Create: `active_rag/nlp_pipeline/document_classifier.py`
- Create: `tests/nlp_pipeline/test_entity_extractor.py`
- Create: `requirements_nlp.txt`

- [ ] **Step 1: Write failing entity extraction test**

```python
# tests/nlp_pipeline/test_entity_extractor.py
import pytest
from active_rag.nlp_pipeline.entity_extractor import EntityExtractor
from active_rag.schemas.entities import ContentDomain

def test_research_entity_extraction():
    extractor = EntityExtractor()
    text = "Einstein published his theory of relativity while working at Princeton University."
    entities = extractor.extract_entities(text, ContentDomain.RESEARCH)
    
    person_entities = [e for e in entities if e["label"] == "Person"]
    org_entities = [e for e in entities if e["label"] == "Organization"]
    
    assert len(person_entities) >= 1
    assert person_entities[0]["properties"]["name"] == "Einstein"
    assert len(org_entities) >= 1
    assert "Princeton" in org_entities[0]["properties"]["name"]

def test_technical_entity_extraction():
    extractor = EntityExtractor()
    text = "The authentication API depends on the Redis cache component for session storage."
    entities = extractor.extract_entities(text, ContentDomain.TECHNICAL)
    
    component_entities = [e for e in entities if e["label"] == "Component"]
    assert len(component_entities) >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/nlp_pipeline/test_entity_extractor.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'active_rag.nlp_pipeline.entity_extractor'"

- [ ] **Step 3: Create NLP requirements file**

```txt
# requirements_nlp.txt
spacy>=3.7.0
transformers>=4.30.0
torch>=2.0.0
scikit-learn>=1.3.0
nltk>=3.8.0
```

- [ ] **Step 4: Create document classifier**

```python
# active_rag/nlp_pipeline/document_classifier.py
import spacy
from typing import Dict, List, Any
import re
from ..schemas.entities import ContentDomain

class DocumentClassifier:
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise ValueError("spaCy English model not found. Run: python -m spacy download en_core_web_sm")
        
        # Keywords for each domain
        self.domain_keywords = {
            ContentDomain.RESEARCH: {
                "keywords": ["research", "study", "paper", "journal", "publication", "experiment", 
                           "hypothesis", "methodology", "analysis", "findings", "abstract", "citation"],
                "patterns": [r"\b(doi:|arxiv:|pubmed:)", r"\b(university|institute|lab)\b", 
                           r"\b\d{4}[a-zA-Z]?\b"]  # Years
            },
            ContentDomain.TECHNICAL: {
                "keywords": ["API", "component", "service", "database", "server", "configuration",
                           "deployment", "code", "function", "class", "library", "framework"],
                "patterns": [r"\b[A-Z][a-zA-Z]*[A-Z][a-zA-Z]*\b",  # CamelCase
                           r"\b(http|https|ftp)://", r"\.(js|py|java|cpp|go)\b"]
            },
            ContentDomain.BUSINESS: {
                "keywords": ["team", "project", "manager", "process", "workflow", "department",
                           "business", "strategy", "goals", "metrics", "stakeholder", "client"],
                "patterns": [r"\b(Q[1-4]|quarter)\b", r"\b(CEO|CTO|VP|Director)\b"]
            }
        }
    
    def classify_document(self, text: str) -> ContentDomain:
        """Classify document into one of the four content domains"""
        if not text or len(text.strip()) < 10:
            return ContentDomain.MIXED_WEB
            
        text_lower = text.lower()
        doc = self.nlp(text[:1000])  # Analyze first 1000 chars for performance
        
        scores = {domain: 0 for domain in ContentDomain}
        
        for domain, criteria in self.domain_keywords.items():
            # Keyword matching
            keyword_score = sum(1 for keyword in criteria["keywords"] 
                              if keyword.lower() in text_lower)
            scores[domain] += keyword_score
            
            # Pattern matching
            pattern_score = sum(1 for pattern in criteria["patterns"]
                              if re.search(pattern, text, re.IGNORECASE))
            scores[domain] += pattern_score * 2  # Weight patterns higher
        
        # Additional heuristics based on named entities
        entities = [(ent.label_, ent.text) for ent in doc.ents]
        
        for label, text in entities:
            if label == "ORG":
                if any(word in text.lower() for word in ["university", "institute", "lab"]):
                    scores[ContentDomain.RESEARCH] += 2
                elif any(word in text.lower() for word in ["corp", "inc", "ltd", "company"]):
                    scores[ContentDomain.BUSINESS] += 1
                    
        # Return domain with highest score, default to MIXED_WEB
        max_domain = max(scores, key=scores.get)
        return max_domain if scores[max_domain] > 0 else ContentDomain.MIXED_WEB
```

- [ ] **Step 5: Create entity extractor**

```python
# active_rag/nlp_pipeline/entity_extractor.py
import spacy
import hashlib
import re
from typing import Dict, List, Any, Tuple
from ..schemas.entities import ContentDomain, ENTITY_SCHEMAS

class EntityExtractor:
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise ValueError("spaCy English model not found. Run: python -m spacy download en_core_web_sm")
    
    def extract_entities(self, text: str, domain: ContentDomain) -> List[Dict[str, Any]]:
        """Extract entities from text based on content domain"""
        if not text or len(text.strip()) < 5:
            return []
            
        doc = self.nlp(text)
        entities = []
        
        if domain == ContentDomain.RESEARCH:
            entities.extend(self._extract_research_entities(doc))
        elif domain == ContentDomain.TECHNICAL:
            entities.extend(self._extract_technical_entities(doc))
        elif domain == ContentDomain.BUSINESS:
            entities.extend(self._extract_business_entities(doc))
        else:  # MIXED_WEB
            entities.extend(self._extract_general_entities(doc))
            
        return self._deduplicate_entities(entities)
    
    def _extract_research_entities(self, doc) -> List[Dict[str, Any]]:
        """Extract research-specific entities"""
        entities = []
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Check if likely to be an author/researcher
                context = self._get_entity_context(ent, doc)
                if self._is_likely_researcher(context):
                    entities.append({
                        "label": "Person",
                        "properties": {
                            "name": ent.text.strip(),
                            "id": self._generate_id("person", ent.text)
                        }
                    })
                    
            elif ent.label_ == "ORG":
                # Check if academic/research organization
                if self._is_academic_org(ent.text):
                    entities.append({
                        "label": "Organization", 
                        "properties": {
                            "name": ent.text.strip(),
                            "id": self._generate_id("org", ent.text),
                            "type": "academic"
                        }
                    })
        
        # Extract concepts from noun phrases
        concepts = self._extract_concepts(doc)
        entities.extend(concepts)
        
        return entities
    
    def _extract_technical_entities(self, doc) -> List[Dict[str, Any]]:
        """Extract technical/API entities"""
        entities = []
        
        # Look for technical components in text
        component_patterns = [
            r'\b([A-Z][a-zA-Z]*(?:API|Service|Component|Module|Library))\b',
            r'\b([a-z]+(?:_[a-z]+)*\.(?:js|py|java|cpp|go))\b',
            r'\b([A-Z][a-zA-Z]*[A-Z][a-zA-Z]*)\b'  # CamelCase
        ]
        
        text = doc.text
        for pattern in component_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                component_name = match.group(1)
                entities.append({
                    "label": "Component",
                    "properties": {
                        "name": component_name,
                        "id": self._generate_id("component", component_name)
                    }
                })
        
        return entities
    
    def _extract_business_entities(self, doc) -> List[Dict[str, Any]]:
        """Extract business process and people entities"""
        entities = []
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Business people (managers, etc)
                context = self._get_entity_context(ent, doc)
                if self._is_business_person(context):
                    entities.append({
                        "label": "Person",
                        "properties": {
                            "name": ent.text.strip(),
                            "id": self._generate_id("person", ent.text)
                        }
                    })
                    
            elif ent.label_ == "ORG":
                # Business organizations
                entities.append({
                    "label": "Organization",
                    "properties": {
                        "name": ent.text.strip(),
                        "id": self._generate_id("org", ent.text),
                        "type": "business"
                    }
                })
        
        # Extract processes from text patterns
        process_patterns = [r'\b([A-Z][a-zA-Z\s]+ (?:process|workflow|procedure))\b']
        for pattern in process_patterns:
            matches = re.finditer(pattern, doc.text)
            for match in matches:
                process_name = match.group(1).strip()
                entities.append({
                    "label": "Process",
                    "properties": {
                        "name": process_name,
                        "id": self._generate_id("process", process_name)
                    }
                })
        
        return entities
    
    def _extract_general_entities(self, doc) -> List[Dict[str, Any]]:
        """Extract general entities from mixed web content"""
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG"]:
                label_map = {"PERSON": "Person", "ORG": "Organization"}
                entities.append({
                    "label": label_map[ent.label_],
                    "properties": {
                        "name": ent.text.strip(),
                        "id": self._generate_id(ent.label_.lower(), ent.text)
                    }
                })
        
        return entities
    
    def _generate_id(self, entity_type: str, name: str) -> str:
        """Generate consistent ID for entity"""
        normalized = name.lower().strip()
        hash_obj = hashlib.md5(f"{entity_type}_{normalized}".encode())
        return f"{entity_type}_{hash_obj.hexdigest()[:8]}"
    
    def _get_entity_context(self, ent, doc, window=50) -> str:
        """Get surrounding context for an entity"""
        start = max(0, ent.start - window)
        end = min(len(doc), ent.end + window)
        return doc[start:end].text
    
    def _is_likely_researcher(self, context: str) -> bool:
        """Check if person is likely a researcher based on context"""
        research_indicators = ["professor", "dr.", "phd", "researcher", "published", "university", "institute"]
        context_lower = context.lower()
        return any(indicator in context_lower for indicator in research_indicators)
    
    def _is_academic_org(self, org_name: str) -> bool:
        """Check if organization is academic/research"""
        academic_keywords = ["university", "institute", "college", "lab", "laboratory", "research"]
        org_lower = org_name.lower()
        return any(keyword in org_lower for keyword in academic_keywords)
    
    def _is_business_person(self, context: str) -> bool:
        """Check if person is in business context"""
        business_indicators = ["manager", "director", "ceo", "cto", "team", "leads", "manages"]
        context_lower = context.lower()
        return any(indicator in context_lower for indicator in business_indicators)
    
    def _extract_concepts(self, doc) -> List[Dict[str, Any]]:
        """Extract conceptual entities from noun phrases"""
        concepts = []
        
        # Look for noun phrases that might be concepts
        for chunk in doc.noun_chunks:
            # Skip short or common phrases
            if len(chunk.text.strip()) < 5 or chunk.text.lower() in ["the", "this", "that"]:
                continue
                
            # Check if it looks like a technical/research concept
            if len(chunk) >= 2 and not any(token.is_stop for token in chunk):
                concepts.append({
                    "label": "Concept",
                    "properties": {
                        "name": chunk.text.strip(),
                        "id": self._generate_id("concept", chunk.text)
                    }
                })
        
        return concepts[:5]  # Limit to top 5 concepts to avoid noise
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities based on ID"""
        seen_ids = set()
        deduped = []
        
        for entity in entities:
            entity_id = entity["properties"]["id"]
            if entity_id not in seen_ids:
                seen_ids.add(entity_id)
                deduped.append(entity)
                
        return deduped
```

- [ ] **Step 6: Create NLP pipeline init**

```python
# active_rag/nlp_pipeline/__init__.py
from .entity_extractor import EntityExtractor
from .document_classifier import DocumentClassifier

__all__ = ['EntityExtractor', 'DocumentClassifier']
```

- [ ] **Step 7: Install NLP dependencies**

Run: `pip install -r requirements_nlp.txt && python -m spacy download en_core_web_sm`
Expected: Dependencies installed and spaCy model downloaded

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/nlp_pipeline/test_entity_extractor.py -v`
Expected: PASS (entity extraction working for research and technical domains)

- [ ] **Step 9: Commit NLP foundation**

```bash
git add active_rag/nlp_pipeline/ tests/nlp_pipeline/ requirements_nlp.txt
git commit -m "feat: add enhanced NLP pipeline with domain-specific entity extraction"
```

### Task 4: Configuration Updates

**Files:**
- Modify: `active_rag/config.py`
- Modify: `requirements.txt`
- Create: `tests/test_config_graph.py`

- [ ] **Step 1: Write failing config test**

```python
# tests/test_config_graph.py
import pytest
from active_rag.config import Config

def test_neo4j_config():
    config = Config()
    assert hasattr(config, 'neo4j_uri')
    assert hasattr(config, 'neo4j_username') 
    assert hasattr(config, 'neo4j_password')
    assert config.neo4j_uri == "bolt://localhost:7687"

def test_graph_features_enabled():
    config = Config()
    assert hasattr(config, 'enable_graph_features')
    assert config.enable_graph_features == True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_graph.py -v`
Expected: FAIL with AttributeError on neo4j_uri

- [ ] **Step 3: Update config with graph settings**

<function_calls>
<invoke name="Read">
<parameter name="file_path">/home/dharshan/chatbot/active_rag/config.py