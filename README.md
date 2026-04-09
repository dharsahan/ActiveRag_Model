# Refined Active GraphRAG Agent 🧠🕸️

An advanced **Autonomous GraphRAG Agent** that unifies vector similarity and relational reasoning into a single **Pure Neo4j architecture**. The system intelligently decides whether to answer directly, search its long-term memory, or browse the live internet—continuously learning from every interaction.

> **API v2.0** — All features are now exposed as production-ready REST endpoints. Build FAQ bots, legal record systems, research assistants, and more.

## Key Features

*   **Pure GraphRAG:** ChromaDB has been replaced by **Neo4j**'s native vector index. Text chunks, their embeddings, and structured entities/relationships all live in a single, unified database.
*   **Autonomous Agent (ReAct):** Uses an Agentic Orchestrator that dynamically chooses from a suite of tools (`web_browser`, `query_memory`, `graph_query`, `calculator`) to answer complex queries.
*   **Continuous Learning Loop:** 
    *   **From Web:** Scraped content is automatically chunked, indexed, and enriched with extracted entities and relationships.
    *   **From Chat:** Every user query and assistant answer is automatically indexed into memory for future recall.
*   **Multi-Hop Reasoning:** Capable of traversing the knowledge graph to "connect the dots" across multiple entities and relationships.
*   **Full REST API:** 20+ endpoints covering queries, ingestion, knowledge graph, NLP, reasoning, evaluation, and system management.
*   **Beautiful Terminal UI:** Powered by `Rich` for a polished, interactive experience with streaming tokens and status indicators.

## Architecture

```mermaid
graph TD
    User([User Query]) --> Agent[Agentic Orchestrator\nReAct Reasoning Loop]

    subgraph Tools [Dynamic Toolset]
        Agent <--> Web[Web Browser\nDDG + Playwright]
        Agent <--> VectorTool[query_memory\nSimilarity Search]
        Agent <--> GraphTool[graph_query\nMulti-hop Logic]
        Agent --> Store[store_memory\nFact Injection]
        Agent --> Calc[calculator\nMath Engine]
    end

    subgraph Learning [Continuous Learning Loop]
        Web -- 1. Scrape --> Content[Raw Text]
        Content -- 2. Index --> Neo4jVector
        Content -- 3. NLP Entity Extraction --> Entities[Nodes and Relations]
        Entities -- 4. Upsert --> Neo4jGraph
    end

    subgraph Unified [Neo4j Graph Database]
        Neo4jVector[(Vector Index)]
        Neo4jGraph[(Knowledge Graph)]
    end

    subgraph API [REST API Layer]
        QueryAPI[/api/v1/query]
        GraphAPI[/api/v1/graph]
        NLPAPI[/api/v1/nlp]
        ReasonAPI[/api/v1/reasoning]
    end

    VectorTool <--> Neo4jVector
    GraphTool <--> Neo4jGraph
    Store --> Neo4jVector

    Agent --> Answer[Final Answer + Citations]
    Answer --> User

    API --> Agent
    API --> Neo4jVector
    API --> Neo4jGraph
```

## Project Structure

```
active_rag/
├── api.py                    # FastAPI app factory (mounts all routers)
├── dependencies.py           # Shared DI (sessions, resources, auth)
├── routers/                  # API endpoint modules
│   ├── query.py              # Query pipelines (agent/hybrid/ultimate)
│   ├── ingestion.py          # File upload, text, batch, URL ingestion
│   ├── knowledge_base.py     # KB stats, search, export, reset
│   ├── graph.py              # Knowledge graph operations
│   ├── nlp.py                # NLP pipeline (entities, relations, classify)
│   ├── reasoning.py          # Multi-hop reasoning & analytics
│   ├── evaluation.py         # Answer quality evaluation
│   └── system.py             # Health, sessions, performance, cache
├── agent.py                  # Agentic Orchestrator (ReAct loop)
├── vector_store.py           # Neo4j-backed Vector Store
├── knowledge_graph/          # Neo4j Client & Graph Operations
├── reasoning/                # Reasoning engine, community detection, cross-domain
├── nlp_pipeline/             # Entity & Relation Extraction
├── tools/                    # ReAct Agent Toolset
├── web_search.py             # Playwright-based Scraper
└── config.py                 # Centralized Configuration
main.py                       # CLI Interface
```

## Setup

### 1. Requirements
*   Python 3.10+
*   Docker (for Neo4j)
*   An LLM backend (Ollama, OpenAI, or GitHub Copilot API)

### 2. Installation
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
playwright install chromium
```

### 3. Start Neo4j
```bash
docker-compose -f docker-compose.neo4j.yml up -d
```

## Configuration

Settings are managed via a `.env` file:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MODEL_NAME` | `gpt-4o` | LLM model to use |
| `OLLAMA_BASE_URL` | `http://localhost:4141/v1` | API endpoint (e.g., local proxy or Ollama) |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j Bolt connection string |
| `VECTOR_INDEX_NAME` | `active_rag` | Name of the vector index in Neo4j |
| `HEADLESS` | `true` | Set to `false` to see the browser while scraping |
| `ACTIVE_RAG_API_KEY` | *(none)* | Optional API key for auth (set to enable) |

## Usage

### Interactive CLI (Recommended)
```bash
python main.py
```

### Direct Query
```bash
python main.py "How is A connected to C in my family tree?"
```

### CLI Commands
*   **`/stats`**: View knowledge base size (nodes, relations, chunks).
*   **`/health`**: Run full system diagnostics.
*   **`/reset`**: Wipe the entire knowledge base and start fresh.
*   **`/dump`**: See every raw text chunk learned by the agent.
*   **`/clear`**: Clear current conversation context.

---

## REST API

Start the API server:
```bash
python main.py --serve
# Server runs on http://localhost:8000
# Swagger docs: http://localhost:8000/docs
```

All endpoints are under `/api/v1/`. Optional API key auth via `X-API-Key` header (set `ACTIVE_RAG_API_KEY` env var to enable).

### Quick Start

```bash
# Health check
curl http://localhost:8000/api/v1/system/health

# Ask a question
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is quantum computing?", "session_id": "user-1"}'
```

---

### 🔍 Query — `/api/v1/query`

Execute queries against any of the 4 RAG pipelines.

#### `POST /api/v1/query`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | string | **required** | The question to ask |
| `session_id` | string | `"default"` | Session ID for conversation memory |
| `pipeline_type` | string | `"agent"` | Pipeline: `agent`, `hybrid`, `ultimate`, `legacy` |

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who founded OpenAI?",
    "session_id": "user-42",
    "pipeline_type": "agent"
  }'
```

**Response:**
```json
{
  "answer": "OpenAI was founded by Sam Altman, Elon Musk, ...",
  "citations": ["https://en.wikipedia.org/wiki/OpenAI"],
  "path": "agent",
  "confidence": 0.92,
  "reasoning": "Known fact with high web evidence",
  "session_id": "user-42"
}
```

#### `POST /api/v1/query/stream`

Same parameters as above. Returns an NDJSON stream for real-time token-by-token responses.

```bash
curl -X POST http://localhost:8000/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain transformers in ML"}'
```

---

### 📥 Ingestion — `/api/v1/ingest`

Load documents into the knowledge base.

#### `POST /api/v1/ingest/upload`

Upload a file (PDF, TXT, MD, DOCX) using multipart form data.

```bash
curl -X POST http://localhost:8000/api/v1/ingest/upload \
  -F "file=@/path/to/document.pdf"
```

#### `POST /api/v1/ingest/text`

Ingest raw text content.

```bash
curl -X POST http://localhost:8000/api/v1/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"content": "Active RAG is a GraphRAG system...", "source": "manual"}'
```

#### `POST /api/v1/ingest/batch`

Bulk-ingest multiple documents at once. Perfect for loading FAQ databases, legal records, or knowledge articles.

```bash
curl -X POST http://localhost:8000/api/v1/ingest/batch \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"content": "Q: How to reset my password? A: Go to Settings > Security...", "source": "faq"},
      {"content": "Q: What are your business hours? A: Mon-Fri, 9am-5pm.", "source": "faq"},
      {"content": "Q: How to contact support? A: Email support@company.com", "source": "faq"}
    ]
  }'
```

**Response:**
```json
{
  "status": "success",
  "total_documents": 3,
  "chunks_created": 3,
  "ids": ["chunk-abc123-1f2e3d4c", "chunk-def456-5a6b7c8d", "chunk-ghi789-9e0f1a2b"]
}
```

#### `POST /api/v1/ingest/url`

Scrape a URL and ingest its content automatically.

```bash
curl -X POST http://localhost:8000/api/v1/ingest/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://en.wikipedia.org/wiki/Knowledge_graph", "max_pages": 1}'
```

---

### 🕸️ Knowledge Graph — `/api/v1/graph`

Direct access to the Neo4j knowledge graph for entity exploration and path discovery.

#### `POST /api/v1/graph/entities/search`

Search for entities by name pattern, optionally filtered by type.

```bash
curl -X POST http://localhost:8000/api/v1/graph/entities/search \
  -H "Content-Type: application/json" \
  -d '{"name_pattern": "Einstein", "entity_types": ["Person"]}'
```

**Response:**
```json
{
  "count": 2,
  "entities": [
    {"id": "person_abc", "name": "Albert Einstein", "labels": ["Person"]},
    {"id": "person_def", "name": "Einstein (disambiguation)", "labels": ["Person"]}
  ]
}
```

#### `GET /api/v1/graph/entities/{entity_id}/neighborhood`

Explore all entities within a given radius of hops.

```bash
curl "http://localhost:8000/api/v1/graph/entities/person_abc/neighborhood?radius=2"
```

#### `GET /api/v1/graph/entities/{entity_id}/related`

Get directly related entities with relationship types.

```bash
curl "http://localhost:8000/api/v1/graph/entities/person_abc/related?depth=1"
```

#### `POST /api/v1/graph/paths`

Find paths connecting two entities — essential for legal case analysis, citation chains, and organizational mapping.

```bash
curl -X POST http://localhost:8000/api/v1/graph/paths \
  -H "Content-Type: application/json" \
  -d '{"start_id": "person_abc", "end_id": "org_xyz", "max_depth": 3}'
```

#### `POST /api/v1/graph/multi-hop`

NLP-powered multi-hop reasoning over the graph.

```bash
curl -X POST http://localhost:8000/api/v1/graph/multi-hop \
  -H "Content-Type: application/json" \
  -d '{"query": "Which researchers at MIT work on transformers?", "max_hops": 2}'
```

#### `GET /api/v1/graph/stats`

Get graph statistics: total nodes, relationships, type breakdowns.

```bash
curl http://localhost:8000/api/v1/graph/stats
```

---

### 🧬 NLP Pipeline — `/api/v1/nlp`

Extract structured information from unstructured text.

#### `POST /api/v1/nlp/entities/extract`

Extract named entities with domain-specific strategies.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | string | **required** | Text to analyze |
| `domain` | string | auto-detect | `research`, `technical`, `business`, or `mixed_web` |

```bash
curl -X POST http://localhost:8000/api/v1/nlp/entities/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Dr. Sarah Chen at Stanford published a paper on transformer architectures using PyTorch.",
    "domain": "research"
  }'
```

**Response:**
```json
{
  "domain": "research",
  "count": 3,
  "entities": [
    {"label": "Person", "properties": {"id": "person_sarah_chen", "name": "Sarah Chen"}},
    {"label": "Organization", "properties": {"id": "org_stanford", "name": "Stanford"}},
    {"label": "Concept", "properties": {"id": "concept_transformer", "name": "transformer architectures"}}
  ]
}
```

#### `POST /api/v1/nlp/relations/extract`

Extract subject-predicate-object relationships between entities.

```bash
curl -X POST http://localhost:8000/api/v1/nlp/relations/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "John manages the engineering team at Google."}'
```

#### `POST /api/v1/nlp/classify`

Classify text into content domains: `research`, `technical`, `business`, or `mixed_web`.

```bash
curl -X POST http://localhost:8000/api/v1/nlp/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Abstract: We present a novel approach to reinforcement learning..."}'
```

**Response:**
```json
{"domain": "research", "text_length": 64}
```

#### `POST /api/v1/nlp/sentiment`

Analyze text sentiment.

```bash
curl -X POST http://localhost:8000/api/v1/nlp/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is absolutely fantastic and exceeded all expectations!"}'
```

---

### 🧠 Reasoning & Analytics — `/api/v1/reasoning`

Advanced graph analysis with multi-hop reasoning, community detection, and cross-domain discovery.

#### `POST /api/v1/reasoning/reason`

Execute the full multi-hop reasoning pipeline: extract entities → expand neighborhoods → find paths → rank by relevance → return confidence.

```bash
curl -X POST http://localhost:8000/api/v1/reasoning/reason \
  -H "Content-Type: application/json" \
  -d '{"query": "How is Dr. Smith connected to the Stanford AI Lab?", "max_hops": 3}'
```

**Response:**
```json
{
  "query": "How is Dr. Smith connected to the Stanford AI Lab?",
  "confidence": 0.78,
  "reasoning_summary": "Identified entities: Dr. Smith, Stanford AI Lab. Found 3 reasoning path(s). Top path: Dr. Smith —[AFFILIATED_WITH]→ Stanford —[CONTAINS]→ AI Lab",
  "has_results": true,
  "ranked_paths": [
    {
      "reasoning_text": "Dr. Smith —[AFFILIATED_WITH]→ Stanford —[CONTAINS]→ AI Lab",
      "score": 0.85,
      "length": 2,
      "start_entity": "Dr. Smith",
      "end_entity": "AI Lab"
    }
  ],
  "subgraph": {"node_count": 12, "edge_count": 8}
}
```

#### `POST /api/v1/reasoning/communities`

Detect entity communities using label propagation clustering.

```bash
curl -X POST http://localhost:8000/api/v1/reasoning/communities \
  -H "Content-Type: application/json" \
  -d '{"entity_type": "Person", "max_entities": 100}'
```

**Response:**
```json
{
  "count": 3,
  "communities": [
    {"community_id": 0, "size": 8, "dominant_label": "Person", "entity_names": ["Alice", "Bob", "Carol"]},
    {"community_id": 1, "size": 5, "dominant_label": "Organization", "entity_names": ["MIT", "Stanford"]}
  ]
}
```

#### `POST /api/v1/reasoning/cross-domain`

Find how an entity connects across different content domains.

```bash
curl -X POST http://localhost:8000/api/v1/reasoning/cross-domain \
  -H "Content-Type: application/json" \
  -d '{"entity_id": "person_abc", "max_depth": 2}'
```

#### `GET /api/v1/reasoning/bridges`

Discover bridge entities that connect multiple domains.

```bash
curl "http://localhost:8000/api/v1/reasoning/bridges?max_entities=50"
```

---

### ✅ Evaluation — `/api/v1/evaluate`

Score answer quality using the LLM as a judge.

#### `POST /api/v1/evaluate`

```bash
curl -X POST http://localhost:8000/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "answer": "The capital of France is Paris."
  }'
```

**Response:**
```json
{
  "quality": 0.95,
  "is_acceptable": true,
  "issues": [],
  "suggestion": ""
}
```

---

### 📊 Knowledge Base — `/api/v1/kb`

Manage the vector knowledge base.

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/kb/stats` | GET | Vector chunk count + graph node/relation counts |
| `/api/v1/kb/search` | POST | Semantic search (`{"query": "...", "limit": 5}`) |
| `/api/v1/kb/export` | GET | Export all stored documents |
| `/api/v1/kb/reset` | DELETE | Wipe the entire knowledge base |

---

### ⚙️ System — `/api/v1/system`

Health checks, monitoring, and management.

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/system/health` | GET | Check LLM, Neo4j, and graph connectivity |
| `/api/v1/system/sessions` | GET | List all active conversation sessions |
| `/api/v1/system/memory/{session_id}` | GET | Dump memory for a session |
| `/api/v1/system/memory/{session_id}` | DELETE | Clear a session's memory |
| `/api/v1/system/performance` | GET | Query execution time stats and cache hit rates |
| `/api/v1/system/cache/stats` | GET | Graph cache metrics (hits, misses, evictions) |
| `/api/v1/system/cache/invalidate` | POST | Clear cache (`{"query_type": null}` for all) |
| `/api/v1/config` | GET | View current system configuration |
| `/api/v1/config` | PATCH | Update config (top_k, thresholds, etc.) |

---

## 🏗️ Building Apps — Examples

### Company FAQ Bot

```python
import requests

API = "http://localhost:8000/api/v1"

# Step 1: Bulk-load FAQ entries
faqs = [
    {"content": "Q: How to reset password? A: Go to Settings > Security > Reset", "source": "faq"},
    {"content": "Q: What are business hours? A: Mon-Fri, 9am-5pm EST", "source": "faq"},
    {"content": "Q: How to contact support? A: Email support@company.com", "source": "faq"},
]
requests.post(f"{API}/ingest/batch", json={"documents": faqs})

# Step 2: Query with session memory for follow-up questions
response = requests.post(f"{API}/query", json={
    "query": "How do I reset my password?",
    "session_id": "customer-123",
})
print(response.json()["answer"])

# Step 3: Evaluate answer quality
answer = response.json()["answer"]
eval_result = requests.post(f"{API}/evaluate", json={
    "query": "How do I reset my password?",
    "answer": answer,
})
print(f"Quality: {eval_result.json()['quality']}")
```

### Legal Record Analysis

```python
import requests

API = "http://localhost:8000/api/v1"

# Step 1: Ingest case documents
requests.post(f"{API}/ingest/upload", files={"file": open("case_file.pdf", "rb")})

# Step 2: Extract entities from text
entities = requests.post(f"{API}/nlp/entities/extract", json={
    "text": "Defendant John Doe was represented by Acme Legal Corp in Case #2024-001.",
    "domain": "business",
}).json()

# Step 3: Find how two entities are connected
paths = requests.post(f"{API}/graph/paths", json={
    "start_id": "person_john_doe",
    "end_id": "org_acme_legal",
    "max_depth": 3,
}).json()

# Step 4: Full reasoning query
reasoning = requests.post(f"{API}/reasoning/reason", json={
    "query": "How is John Doe connected to Case #2024-001?",
    "max_hops": 3,
}).json()
print(reasoning["reasoning_summary"])
```

### Research Assistant

```python
import requests

API = "http://localhost:8000/api/v1"

# Step 1: Ingest research papers via URL
requests.post(f"{API}/ingest/url", json={"url": "https://arxiv.org/abs/2301.00001"})

# Step 2: Classify document domains
result = requests.post(f"{API}/nlp/classify", json={
    "text": "We propose a novel attention mechanism for language models..."
}).json()
print(f"Domain: {result['domain']}")  # → "research"

# Step 3: Discover research communities
communities = requests.post(f"{API}/reasoning/communities", json={
    "entity_type": "Person",
    "max_entities": 200,
}).json()
for c in communities["communities"][:3]:
    print(f"Community {c['community_id']}: {', '.join(c['entity_names'][:5])}")

# Step 4: Find cross-domain bridges
bridges = requests.get(f"{API}/reasoning/bridges?max_entities=50").json()
for b in bridges["bridges"][:5]:
    print(f"Bridge: {b['entity']['name']} → Domains: {b['connected_domains']}")
```

---

## Testing

```bash
# All tests
python -m pytest tests/ -v

# API tests only
python -m pytest tests/test_api_v1.py -v
```

## API Documentation

Interactive Swagger UI is available at `http://localhost:8000/docs` when the server is running. All endpoints are grouped by category with full request/response schemas.
