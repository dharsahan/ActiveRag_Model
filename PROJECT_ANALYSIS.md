# ActiveRAG Model - Project Analysis 🧠🕸️

**Last Updated:** March 31, 2026  
**Project Type:** Autonomous GraphRAG Agent  
**Repository:** https://github.com/dharsahan/ActiveRag_Model  
**Main Language:** Python 3.10+

---

## 📋 Executive Summary

**ActiveRAG** is an advanced **Autonomous GraphRAG Agent** that unifies vector similarity search and relational reasoning into a single **pure Neo4j architecture**. Unlike traditional RAG systems, it uses an **autonomous agent loop** (ReAct pattern) to dynamically decide whether to answer directly, search long-term memory, browse the live internet, or perform graph reasoning—continuously learning from every interaction.

### Key Innovation
The system has recently migrated from a dual-storage approach (ChromaDB + Neo4j) to a **unified Neo4j backend**, eliminating data silos and enabling seamless integration between vector search and graph reasoning.

---

## 🏗️ Architecture Overview

### High-Level Flow

```
User Query
    ↓
Agentic Orchestrator (ReAct Loop)
    ↓
    ├─→ Calculator Tool
    ├─→ Web Browser Tool (DuckDuckGo + Playwright)
    ├─→ Query Memory Tool (Vector Search in Neo4j)
    ├─→ Graph Query Tool (Multi-hop Traversal)
    ├─→ Store Memory Tool (Fact Injection)
    └─→ List Memory Tool (Knowledge Enumeration)
    ↓
Neo4j Unified Database
    ├─ Vector Index (Chunk Embeddings)
    └─ Knowledge Graph (Entities & Relations)
    ↓
Answer Generator + Citations
    ↓
User (with streaming output)
```

### Core Subsystems

#### 1. **Agentic Orchestrator** (`agent.py` - 585 lines)
- Implements ReAct (Reasoning + Acting) loop
- Maintains conversation memory for multi-turn dialogue
- Tool schema definitions and execution routing
- Supports both sync and async execution modes
- Exponential backoff retry logic

**Key Methods:**
- `_execute_tool()` / `_execute_tool_async()` - Tool invocation dispatcher
- `run()` / `run_async()` - Main reasoning loop with token streaming
- `_parse_assistant_response()` - Action/observation parsing

#### 2. **Neo4j Knowledge Graph** (`knowledge_graph/` - 1,276 lines)

**Components:**

| Module | Size | Purpose |
|--------|------|---------|
| `neo4j_client.py` | 167 L | Connection pooling, session management |
| `graph_operations.py` | 260 L | Entity/relation CRUD, multi-hop traversal |
| `schema_manager.py` | 366 L | Graph schema validation, index creation |
| `query_builder.py` | 80 L | Cypher query DSL construction |
| `graph_cache.py` | 173 L | LRU query result caching |
| `query_monitor.py` | 113 L | Performance metrics, slow query detection |
| `index_manager.py` | 112 L | Vector index and property index management |

**Key Features:**
- ✅ Lazy connection initialization
- ✅ Native vector index support (`vector-1.0` index type)
- ✅ Schema validation with Pydantic
- ✅ Query caching to reduce database load
- ✅ Performance monitoring and slow query logging

#### 3. **Vector Store** (`vector_store.py` - 262 lines)

**Responsibilities:**
- Manages text **chunking** (sliding window strategy)
- Handles **embedding generation** (via sentence-transformers)
- Upserts chunks into Neo4j vector index
- Similarity search with configurable `top_k`
- Chunk metadata (source, timestamp, domain)

**Key Methods:**
- `ingest_document()` - Chunk, embed, and store
- `query()` - Vector similarity search
- `get_stats()` - Knowledge base statistics

#### 4. **NLP Pipeline** (`nlp_pipeline/` - 539 lines)

| Module | Size | Purpose |
|--------|------|---------|
| `entity_extractor.py` | 261 L | Extract entities (Person, Organization, Location, etc.) |
| `relation_extractor.py` | 100 L | Identify relationships between entities |
| `document_classifier.py` | 156 L | Classify document type (news, research, wiki, etc.) |

**Extraction Methods:**
- Spacy-based NER for fast extraction
- Transformer-based refinement for accuracy
- Confidence scoring for quality filtering
- Relationship context analysis

#### 5. **Tools Suite** (`tools/` - 7 tools)

| Tool | Type | Purpose |
|------|------|---------|
| `calculator.py` | Utility | Arithmetic and mathematical expressions |
| `web_browser.py` | Integration | DuckDuckGo search + Playwright scraping |
| `vector_database.py` | Memory | Similarity search in Neo4j vector index |
| `graph_query.py` | Reasoning | Multi-hop graph traversal (up to 3 hops) |
| `store_memory.py` | Learning | Inject new facts/entities into knowledge graph |
| `list_memory.py` | Enumeration | List entities by type or category |
| `crawl.py` | Web Integration | (Deprecated/Legacy) |

#### 6. **Reasoning Engine** (`reasoning/` - 4 modules)

| Module | Purpose |
|--------|---------|
| `reasoning_engine.py` | Core inference with explanation generation |
| `community_detection.py` | Identify clusters of related entities |
| `cross_domain.py` | Bridge concepts across different domains |
| `explainability.py` | Generate human-readable reasoning paths |

---

## 📁 Project Structure

```
/home/dharshan/chatbot/
├── active_rag/                     # Main package
│   ├── agent.py                    # Agentic Orchestrator (ReAct)
│   ├── vector_store.py             # Neo4j Vector Store
│   ├── config.py                   # Centralized config (95 L)
│   ├── pipeline.py                 # Legacy pipeline (373 L)
│   ├── hybrid_pipeline.py          # Hybrid RAG pipeline (412 L)
│   ├── ultimate_pipeline.py        # Advanced pipeline (612 L)
│   ├── answer_generator.py         # Response synthesis (268 L)
│   ├── console.py                  # Rich UI helpers (143 L)
│   ├── api.py                      # FastAPI REST interface (148 L)
│   │
│   ├── knowledge_graph/            # Neo4j backend
│   │   ├── neo4j_client.py
│   │   ├── graph_operations.py
│   │   ├── schema_manager.py
│   │   ├── graph_cache.py
│   │   ├── query_monitor.py
│   │   ├── index_manager.py
│   │   └── query_builder.py
│   │
│   ├── nlp_pipeline/               # Entity/relation extraction
│   │   ├── entity_extractor.py
│   │   ├── relation_extractor.py
│   │   └── document_classifier.py
│   │
│   ├── tools/                      # Agent toolset
│   │   ├── calculator.py
│   │   ├── web_browser.py
│   │   ├── vector_database.py
│   │   ├── graph_query.py
│   │   ├── store_memory.py
│   │   ├── list_memory.py
│   │   └── crawl.py
│   │
│   ├── reasoning/                  # Advanced reasoning
│   │   ├── reasoning_engine.py
│   │   ├── explainability.py
│   │   ├── community_detection.py
│   │   └── cross_domain.py
│   │
│   ├── routing/                    # Query routing
│   │   ├── query_classifier.py
│   │   ├── strategy_selector.py
│   │   └── result_combiner.py
│   │
│   ├── schemas/                    # Pydantic models
│   │   ├── entities.py
│   │   └── relationships.py
│   │
│   ├── memory.py                   # Conversation memory (142 L)
│   ├── cache.py                    # Response cache (84 L)
│   ├── confidence_checker.py       # Confidence scoring (87 L)
│   ├── chunker.py                  # Text chunking (76 L)
│   ├── document_loader.py          # File ingestion (96 L)
│   └── ... (11 more utility modules)
│
├── main.py                         # CLI entry point (25 KB)
├── tests/                          # Comprehensive test suite
│   ├── test_*.py (10+ modules)
│   ├── knowledge_graph/
│   ├── nlp_pipeline/
│   └── schemas/
│
├── scripts/
│   └── health_check.py
│
├── README.md                       # User-facing docs
├── activerag_documentation.md      # Technical deep-dive
├── docker-compose.neo4j.yml        # Neo4j containerization
├── requirements.txt                # Dependencies (36 lines)
├── requirements_nlp.txt            # NLP extras
└── .env                            # Configuration (not in repo)
```

### Line Count Summary
```
Core Agent Logic:        585 lines
Knowledge Graph:       1,276 lines
NLP Pipeline:            539 lines
Tools Suite:             ~700 lines
Reasoning Engine:        ~600 lines
Utilities & Config:      ~700 lines
────────────────────────────────
Total Core Package:    ~3,947 lines
```

---

## 🔧 Key Technologies

### LLM & NLP
- **LLM Backends:** OpenAI, Local Ollama, GitHub Copilot API
- **Embeddings:** Sentence-Transformers (all-MiniLM-L6-v2)
- **NER/POS:** Spacy 3.7+
- **Advanced NLP:** HuggingFace Transformers

### Database & Storage
- **Primary DB:** Neo4j 5.15+ (Graph DB)
- **Vector Index:** Neo4j native vector index (cosine distance)
- **Caching:** DiskCache, LRU in-memory cache
- **Session Management:** Neo4j driver with connection pooling

### Web & Automation
- **Search:** DuckDuckGo (via `duckduckgo-search`)
- **Scraping:** Playwright (headless/headed browser)
- **HTTP:** httpx, requests
- **HTML Parsing:** BeautifulSoup4

### Web Framework & API
- **REST API:** FastAPI 0.100+
- **Server:** Uvicorn
- **WebSocket:** Async streaming support

### Development & Testing
- **Test Framework:** pytest
- **Mocking:** pytest fixtures
- **Code Quality:** Type hints (Python 3.10+)
- **Terminal UI:** Rich (colors, panels, progress bars)

### Supporting Libraries
- **Document Parsing:** PyPDF2, python-docx
- **Ranking:** rank_bm25, RapidFuzz
- **Async:** nest_asyncio, asyncio
- **Config:** python-dotenv
- **Graph Analysis:** NetworkX

---

## 🔄 Data Flow & Execution Patterns

### Query Processing (Complete Flow)

```python
1. User Input → main.py
   ↓
2. LLM Backend Health Check
   ↓
3. AgenticOrchestrator Initialization
   ├─ Load Config
   ├─ Initialize Neo4j Client
   ├─ Load Tool Schemas
   └─ Load Conversation Memory
   ↓
4. ReAct Loop (Agent Thinking)
   ├─ Send query + tool schemas to LLM
   ├─ LLM returns: Action (tool name + args) or Final Answer
   ├─ Repeat:
   │  ├─ Parse LLM output
   │  ├─ If Action → Execute Tool
   │  │  ├─ Tool processing
   │  │  └─ Return observation
   │  ├─ If Final Answer → Generate response
   │  └─ Send [Thought, Action, Observation] back to LLM
   └─ Exit when Final Answer generated (max 10 iterations)
   ↓
5. Answer Generation
   ├─ Synthesis from observations
   ├─ Citation extraction
   └─ Confidence scoring
   ↓
6. Rich Console Output (streaming)
   ├─ Panel with answer
   ├─ Sources/citations
   ├─ Execution stats
   └─ Token count
```

### Document Ingestion (Learning Loop)

```python
Web Content (or uploaded file)
   ↓
Text Extraction & Cleaning
   ↓
Chunking (Sliding Window: 512 tokens, 256 overlap)
   ↓
Embedding Generation (sentence-transformers)
   ↓
Vector Store Upsert (Neo4j)
   ↓
NLP Entity Extraction (Spacy + Transformers)
   ↓
Relationship Extraction (contextual analysis)
   ↓
Graph Schema Validation (Pydantic)
   ↓
Graph Operations (Neo4j Cypher)
   │
   ├─ MERGE nodes (idempotent)
   ├─ CREATE relationships
   └─ Index updates
   ↓
Conversation Memory Update
   ↓
✓ Knowledge base enriched
```

### Tool Execution Context

**Vector Search (query_memory):**
- Input: Query text
- Processing: Embed → Neo4j similarity search → Top-K chunks
- Output: Ranked results with similarity scores

**Graph Query (graph_query):**
- Input: Entity name, relation type, max hops
- Processing: Cypher multi-hop traversal
- Output: Connected entities and paths

**Web Browser (web_browser):**
- Input: Search query
- Processing: DuckDuckGo → URL ranking → Playwright scrape
- Output: Extracted text, source URL, timestamp
- Side Effect: Ingest into vector store + graph

**Store Memory (store_memory):**
- Input: Fact (entity type, properties, relations)
- Processing: NLP validation → Graph schema check
- Output: Updated knowledge graph
- Side Effect: Vector index updated

---

## 🎯 Configuration & Environment

### Environment Variables (`config.py`)

**LLM Settings:**
```
LLM_PROVIDER          → "local" | "openai" | "github" | "together"
OLLAMA_BASE_URL       → HTTP endpoint (default: http://localhost:4141/v1)
MODEL_NAME            → Model ID (e.g., "gpt-4o", "llama2")
LLM_API_KEY           → API key for external providers
```

**Neo4j Settings:**
```
NEO4J_URI             → bolt://localhost:7687
NEO4J_USERNAME        → neo4j
NEO4J_PASSWORD        → activerag123
VECTOR_INDEX_NAME     → active_rag
```

**RAG Settings:**
```
CONFIDENCE_THRESHOLD  → 0.7 (skip RAG if confidence ≥ threshold)
TOP_K                 → 3 (number of chunks to retrieve)
MAX_SEARCH_RESULTS    → 3 (web search result count)
TIME_SENSITIVE_MAX_AGE → 3600 (seconds, for recent event queries)
MAX_GRAPH_HOPS        → 3 (reasoning depth limit)
HEADLESS              → true (Playwright browser mode)
```

**NLP Settings:**
```
SPACY_MODEL           → en_core_web_sm
ENABLE_GRAPH_FEATURES → true
ENABLE_RELATION_EXTRACTION → true
```

### Config Class Hierarchy

```
Config (dataclass)
├─ Provider detection (__post_init__)
├─ LLM config loading
├─ Neo4j connection parameters
├─ Vector store settings
├─ Web search settings
├─ Graph feature toggles
└─ NLP pipeline configuration
```

---

## 🧪 Testing & Quality Assurance

### Test Suite Coverage

**Test Count:** 20+ test modules

| Category | Test Files | Coverage |
|----------|-----------|----------|
| **Core** | test_main.py, test_pipeline.py, test_vector_store.py | Agent loop, vector ops |
| **Configuration** | test_config.py, test_config_graph.py, test_providers.py | Config loading, providers |
| **Knowledge Graph** | test_neo4j_client.py, test_graph_operations.py, test_graph_cache.py | Graph CRUD, caching |
| **NLP** | test_entity_extractor.py, test_document_classifier.py | Entity extraction, classification |
| **Utilities** | test_chunker.py, test_document_loader.py, test_token_tracker.py | Text processing |
| **API** | test_api.py | REST endpoints |

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific category
pytest tests/knowledge_graph/

# Verbose output
pytest -v tests/

# With coverage
pytest --cov=active_rag tests/
```

---

## 🚀 Usage Patterns

### 1. Interactive CLI Mode (Default)

```bash
$ python main.py

Welcome to ActiveRAG Agent! 🧠🕸️
> Tell me about quantum computing

[Agent thinks and uses tools...]
Final Answer:
Quantum computing leverages superposition and entanglement...

/stats  → View knowledge base size
/health → Run system diagnostics
/reset  → Wipe knowledge base
```

### 2. Direct Query

```bash
$ python main.py "How many countries are in Europe?"

[Processes query directly without interactive mode]
```

### 3. REST API

```bash
$ python -m active_rag.api
# Starts FastAPI server on http://localhost:8000

# Query endpoint
POST /query
{
  "query": "Tell me about Python programming",
  "stream": true,
  "use_memory": true
}

# Memory operations
GET /memory/list?entity_type=Person
POST /memory/store
DELETE /memory/clear
```

### 4. Programmatic Usage

```python
from active_rag.agent import AgenticOrchestrator
from active_rag.config import Config

config = Config()
agent = AgenticOrchestrator(config)

# Sync execution
result = agent.run("Who founded OpenAI?")
print(result.answer)
print(result.sources)

# Async execution
import asyncio
result = asyncio.run(agent.run_async("What is AGI?"))
```

---

## 📊 Recent Development History

### Git Timeline (Last 5 Commits)

| Commit | Message | Impact |
|--------|---------|--------|
| `ddf6e86` | Merge branch 'main' | Integration checkpoint |
| `7f0cc2f` | Replace dual storage with direct extraction | Unified Neo4j architecture |
| `52b07c6` | Fix formatting in architecture diagram | Documentation |
| `a01f954` | Update README for Neo4j migration | User documentation |
| `ba9b1f9` | Remove DualStorageManager, migrate logic | Major refactoring |

### Migration Milestones

1. **Phase 0:** ChromaDB + Neo4j dual storage (legacy)
2. **Phase 1:** Unified Neo4j backend (current)
   - Direct entity/relation extraction in ingestion
   - Removed DualStorageManager class
   - Simplified API surface
   - Better data consistency

---

## 🔍 Key Metrics & Statistics

### Codebase
- **Total Lines:** ~3,947 (core package)
- **Python Modules:** 50+
- **Test Modules:** 20+
- **Documentation:** 4 markdown files

### Database
- **Primary Storage:** Neo4j 5.15+
- **Vector Index Type:** Neo4j native (cosine similarity)
- **Chunk Strategy:** Sliding window (512 tokens, 256 overlap)
- **Embedding Model:** all-MiniLM-L6-v2 (384 dimensions)

### Performance Targets
- **Query Latency:** <5s (with LLM)
- **Vector Search:** <100ms (for 10K chunks)
- **Graph Traversal:** <200ms (3-hop query)
- **Token Limit:** Configurable (default ~4K context)

---

## ⚠️ Known Limitations & Future Work

### Current Limitations
1. **Graph Size Scaling:** Not tested beyond 100K nodes
2. **Concurrent Requests:** Single-threaded agent loop (scalability concern)
3. **Vector Dimensionality:** Fixed 384 dimensions (all-MiniLM)
4. **Relationship Inference:** Limited to extraction-based, no learned relations
5. **Multi-Language:** English-optimized (Spacy model)
6. **Reasoning Depth:** Max 3 hops (configurable but may impact performance)

### Future Enhancements
1. **Distributed Reasoning:** Multi-agent collaboration
2. **Dynamic Fact Verification:** Fact-checking against trusted sources
3. **Continual Learning:** Online fine-tuning of extractors
4. **Multi-Modal:** Image + video understanding
5. **Explanability UI:** Interactive reasoning path visualization
6. **Monitoring Dashboard:** Real-time metrics and alerts

---

## 🎓 Architecture Decision Records (ADRs)

### ADR-001: Neo4j Over Vector-Only Search
**Decision:** Use unified Neo4j backend instead of separate vector + graph stores.

**Rationale:**
- Single source of truth for all knowledge
- Transactional consistency
- Reduced data synchronization overhead
- Native support for both embeddings and structured data

### ADR-002: ReAct Pattern for Agent Loop
**Decision:** Implement autonomous agent using ReAct (Reasoning + Acting) pattern.

**Rationale:**
- Transparent reasoning process
- Easy tool composition
- Works with any LLM API
- Supports multi-step reasoning

### ADR-003: Spacy + Transformers for NLP
**Decision:** Use hybrid approach: Spacy for speed, Transformers for accuracy.

**Rationale:**
- Spacy provides fast baseline
- Transformers improve precision when needed
- Confidence scoring allows filtering
- Flexible accuracy/speed tradeoff

---

## 📚 Documentation References

| Document | Purpose | Location |
|----------|---------|----------|
| **README.md** | User guide, setup, usage | Root |
| **activerag_documentation.md** | Technical deep-dive | Root |
| **API_PERFORMANCE_FIX.md** | Performance optimization notes | Root |
| **ULTIMATE_DEMO.md** | Advanced demo scenarios | Root |

---

## 💼 Deployment & Ops

### Docker Deployment
```bash
# Start Neo4j
docker-compose -f docker-compose.neo4j.yml up -d

# Verify connectivity
python -m active_rag.scripts.health_check
```

### Production Considerations
- ✅ Async-ready architecture
- ✅ Connection pooling
- ✅ Query caching
- ✅ Health check endpoints
- ⚠️ Monitoring dashboards (needed)
- ⚠️ Distributed caching (needed for scalability)
- ⚠️ Rate limiting (needed for API tier)

---

## 🎯 Project Health Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| **Architecture** | ✅ Stable | Recent migration to unified Neo4j complete |
| **Testing** | ✅ Good | 20+ test modules, comprehensive coverage |
| **Documentation** | ✅ Excellent | Multiple markdown guides, code comments |
| **Code Quality** | ✅ Good | Type hints, modular design, config management |
| **Performance** | ⚠️ Monitor | Not stress-tested at scale, single-threaded |
| **Maintainability** | ✅ Good | Clear separation of concerns, well-organized |
| **DevOps** | ⚠️ Partial | Docker support, but no CI/CD pipeline visible |

---

## 🤝 Getting Started for Contributors

### Setup Development Environment
```bash
# Clone repo
git clone https://github.com/dharsahan/ActiveRag_Model.git
cd chatbot

# Create venv
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_nlp.txt
python -m spacy download en_core_web_sm

# Setup Playwright
playwright install chromium

# Start Neo4j (if not already running)
docker-compose -f docker-compose.neo4j.yml up -d

# Run tests
pytest tests/ -v

# Start agent
python main.py
```

### Code Organization Best Practices
- **Agent logic:** Modify `agent.py` for tool execution
- **New tools:** Add to `tools/` directory with schema definition
- **NLP improvements:** Update `nlp_pipeline/` modules
- **Graph operations:** Extend `knowledge_graph/graph_operations.py`
- **Config changes:** Update `config.py` (not hardcoded values)

---

## 📖 Summary

**ActiveRAG** represents a sophisticated approach to knowledge-intensive question answering by combining:

1. **Autonomous reasoning** (ReAct agent loop)
2. **Unified knowledge storage** (Neo4j graph + vectors)
3. **Continuous learning** (from web and conversations)
4. **Multi-hop reasoning** (graph traversal)
5. **Tool composition** (calculator, web, memory, reasoning)

The system is **production-ready** for small to medium deployments and demonstrates clean architecture principles with excellent separation of concerns. Recent refactoring unified the storage model, reducing complexity while maintaining powerful capabilities.

---

**Analysis prepared by:** AI Assistant  
**Analysis date:** 2026-03-31  
**Project status:** Active Development ✅
