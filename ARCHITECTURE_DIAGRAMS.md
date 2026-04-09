# ActiveRAG - Architecture Diagrams & Flowcharts

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          USER INTERACTION LAYER                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐  │
│  │   main.py (CLI)  │  │   FastAPI REST   │  │  Programmatic Import     │  │
│  │  - Interactive   │  │   - /query       │  │  - agent.run(query)      │  │
│  │  - Direct query  │  │   - /memory/*    │  │  - agent.run_async()     │  │
│  │  - Commands      │  │   - /stats       │  │                          │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────────────┘  │
│           │                     │                     │                    │
└───────────┼─────────────────────┼─────────────────────┼────────────────────┘
            │                     │                     │
            └─────────────────────┼─────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   Config Initialization   │
                    │  (LLM, Neo4j, NLP        │
                    │   Provider Detection)     │
                    └─────────────┬─────────────┘
                                  │
        ┌─────────────────────────▼─────────────────────────┐
        │     AGENTIC ORCHESTRATOR LAYER (ReAct Loop)       │
        ├──────────────────────────────────────────────────┤
        │                                                    │
        │  1. Load Conversation Memory                      │
        │  2. Format Tools Schema                           │
        │  3. Send Query + Schemas to LLM                   │
        │                                                    │
        │  ┌─ Thinking Loop (max 10 iterations) ──────────┐ │
        │  │                                               │ │
        │  │  LLM Response Parser                          │ │
        │  │    ├─ Thought: reasoning                      │ │
        │  │    ├─ Action: tool_name + arguments           │ │
        │  │    └─ Final Answer: synthesis                 │ │
        │  │                                               │ │
        │  └───────────────┬──────────────────────────────┘ │
        │                  │                                 │
        │    ┌─────────────▼──────────────┐                 │
        │    │  Tool Execution Dispatcher │                 │
        │    │  (_execute_tool handler)   │                 │
        │    └─────────────┬──────────────┘                 │
        │                  │                                 │
        └──────────────────┼─────────────────────────────────┘
                           │
        ┌──────────────────┴────────────────────────┐
        │         TOOL ECOSYSTEM LAYER              │
        ├───────────────────────────────────────────┤
        │                                            │
        │  ┌──────────────────────────────────────┐ │
        │  │  Utility Tools                       │ │
        │  ├──────────────────────────────────────┤ │
        │  │  • calculator.py (arithmetic)        │ │
        │  └──────────────────────────────────────┘ │
        │                                            │
        │  ┌──────────────────────────────────────┐ │
        │  │  Integration Tools                   │ │
        │  ├──────────────────────────────────────┤ │
        │  │  • web_browser (DuckDuckGo +        │ │
        │  │    Playwright scraping)              │ │
        │  └──────────────────────────────────────┘ │
        │                                            │
        │  ┌──────────────────────────────────────┐ │
        │  │  Memory & Search Tools               │ │
        │  ├──────────────────────────────────────┤ │
        │  │  • query_memory (vector search)      │ │
        │  │  • store_memory (fact injection)     │ │
        │  │  • list_memory (entity enumeration)  │ │
        │  └──────────────────────────────────────┘ │
        │                                            │
        │  ┌──────────────────────────────────────┐ │
        │  │  Reasoning Tools                     │ │
        │  ├──────────────────────────────────────┤ │
        │  │  • graph_query (multi-hop logic)     │ │
        │  │  • reasoning_engine (inference)      │ │
        │  └──────────────────────────────────────┘ │
        │                                            │
        └────┬─────────────────────────────────┬────┘
             │                                 │
        ┌────▼────────────────────────────────▼────┐
        │   NEO4J UNIFIED DATABASE LAYER           │
        ├─────────────────────────────────────────┤
        │                                           │
        │  ┌──────────────────────────────────┐   │
        │  │  Vector Index                    │   │
        │  │  (native cosine similarity)      │   │
        │  │                                  │   │
        │  │  • Text chunks (512 tokens)      │   │
        │  │  • Embeddings (384 dimensions)   │   │
        │  │  • Metadata (source, timestamp)  │   │
        │  └──────────────────────────────────┘   │
        │                                           │
        │  ┌──────────────────────────────────┐   │
        │  │  Knowledge Graph                 │   │
        │  │  (Cypher-queryable structure)    │   │
        │  │                                  │   │
        │  │  • Entity Nodes (Person, Org)    │   │
        │  │  • Relationship Edges            │   │
        │  │  • Property Indexes              │   │
        │  └──────────────────────────────────┘   │
        │                                           │
        │  ┌──────────────────────────────────┐   │
        │  │  Query Caching (LRU)             │   │
        │  │  Performance Monitoring          │   │
        │  └──────────────────────────────────┘   │
        │                                           │
        └─────────────────────────────────────────┘
                           │
        ┌──────────────────▼──────────────────┐
        │   NLP PROCESSING PIPELINE           │
        ├───────────────────────────────────┤
        │                                     │
        │  • Entity Extraction (Spacy NER)   │
        │  • Relation Extraction (Context)   │
        │  • Document Classification         │
        │  • Confidence Scoring              │
        │                                     │
        └─────────────────────────────────────┘
                           │
        ┌──────────────────▼──────────────────┐
        │   ANSWER GENERATION LAYER          │
        ├───────────────────────────────────┤
        │                                     │
        │  • Observation Synthesis           │
        │  • Source Citation Extraction      │
        │  • Confidence Assessment           │
        │  • Streaming Token Output          │
        │                                     │
        └─────────────────────────────────────┘
```

## Query Processing Flow (Detailed)

```
START: User Query
  │
  ├─→ [1] INITIALIZATION
  │   ├─→ Load Config (env vars)
  │   ├─→ Connect to Neo4j (pooled connection)
  │   ├─→ Load Conversation Memory
  │   ├─→ Initialize Tool Schemas
  │   └─→ Health Check LLM Backend
  │
  ├─→ [2] REACT LOOP (Max 10 iterations)
  │   │
  │   ├─→ Iteration N:
  │   │   ├─→ Format System Prompt
  │   │   ├─→ Append Conversation History
  │   │   ├─→ Append Tool Schemas
  │   │   ├─→ Append Query
  │   │   ├─→ Stream Response from LLM
  │   │   │
  │   │   ├─→ Parse LLM Output
  │   │   │   ├─→ If "Final Answer" → Go to [3]
  │   │   │   ├─→ If "Action" with tool_name + args → Continue
  │   │   │   └─→ If malformed → Return error
  │   │   │
  │   │   ├─→ Route to Tool Handler
  │   │   │   ├─→ Parse JSON args
  │   │   │   ├─→ Validate against schema
  │   │   │   └─→ Execute Tool
  │   │   │
  │   │   ├─→ Tool Execution (context-dependent):
  │   │   │   │
  │   │   │   ├─ [calculator]
  │   │   │   │  ├─ Parse expression
  │   │   │   │  └─ Return numeric result
  │   │   │   │
  │   │   │   ├─ [query_memory]
  │   │   │   │  ├─ Embed query
  │   │   │   │  ├─ Search Neo4j vector index
  │   │   │   │  ├─ Retrieve top-3 chunks
  │   │   │   │  └─ Return ranked results
  │   │   │   │
  │   │   │   ├─ [graph_query]
  │   │   │   │  ├─ Parse entity + relation + hops
  │   │   │   │  ├─ Build Cypher query
  │   │   │   │  ├─ Execute multi-hop traversal
  │   │   │   │  └─ Return connected entities
  │   │   │   │
  │   │   │   ├─ [web_browser]
  │   │   │   │  ├─ Search DuckDuckGo
  │   │   │   │  ├─ Rank results
  │   │   │   │  ├─ Launch Playwright
  │   │   │   │  ├─ Scrape content
  │   │   │   │  ├─ Extract text
  │   │   │   │  ├─ Ingest to Neo4j (async)
  │   │   │   │  └─ Return extracted content
  │   │   │   │
  │   │   │   ├─ [store_memory]
  │   │   │   │  ├─ Parse fact structure
  │   │   │   │  ├─ Validate against schema
  │   │   │   │  ├─ Create/update Neo4j nodes
  │   │   │   │  ├─ Create relationships
  │   │   │   │  └─ Update vector index
  │   │   │   │
  │   │   │   ├─ [list_memory]
  │   │   │   │  ├─ Query Neo4j by entity type
  │   │   │   │  └─ Return enumerated list
  │   │   │   │
  │   │   │   └─ [Observation returned to LLM]
  │   │   │
  │   │   └─→ Record in Conversation Memory
  │   │
  │   └─→ If iteration ≥ 10 → Force "Final Answer"
  │
  ├─→ [3] ANSWER GENERATION
  │   ├─→ Collect all observations
  │   ├─→ Synthesize final response
  │   ├─→ Extract sources & citations
  │   ├─→ Compute confidence score
  │   └─→ Update Memory Store
  │
  ├─→ [4] OUTPUT & DISPLAY
  │   ├─→ Stream tokens with Rich formatting
  │   ├─→ Display source citations
  │   ├─→ Show execution statistics
  │   │   ├─→ Total tokens used
  │   │   ├─→ Iterations taken
  │   │   ├─→ Tools invoked
  │   │   └─→ Execution time
  │   └─→ Return PipelineResult object
  │
  └─→ END: Return to User
```

## Document Ingestion Pipeline

```
INPUT: Raw Document (Text / PDF / URL / HTML)
  │
  ├─→ [1] DOCUMENT LOADING
  │   ├─→ Detect file type
  │   ├─→ Load content (PyPDF2, python-docx, etc.)
  │   ├─→ Extract text
  │   └─→ Clean text (whitespace, special chars)
  │
  ├─→ [2] CHUNKING
  │   ├─→ Split into tokens (using tiktoken/transformers)
  │   ├─→ Apply sliding window (512 tokens, 256 overlap)
  │   ├─→ Preserve chunk boundaries (sentences)
  │   └─→ Collect chunk metadata:
  │       ├─→ source_url
  │       ├─→ timestamp
  │       ├─→ domain
  │       └─→ chunk_index
  │
  ├─→ [3] EMBEDDING GENERATION
  │   ├─→ Load sentence-transformers model
  │   ├─→ Batch embed chunks (384 dimensions)
  │   └─→ Cache embeddings for reuse
  │
  ├─→ [4] VECTOR STORE UPSERT
  │   ├─→ Connect to Neo4j
  │   ├─→ Create/Update CHUNK nodes
  │   │   ├─→ Properties: text, source, timestamp, domain
  │   │   └─→ Set embedding property (vector)
  │   └─→ Update vector index (cosine similarity)
  │
  ├─→ [5] NLP EXTRACTION
  │   │
  │   ├─→ [Entity Extraction]
  │   │   ├─→ Fast: Spacy NER (Person, Org, Location, etc.)
  │   │   ├─→ Refine: Transformers for boundary detection
  │   │   ├─→ Confidence scoring
  │   │   └─→ Filter by threshold (>0.5)
  │   │
  │   ├─→ [Relation Extraction]
  │   │   ├─→ Contextual analysis (entities in same chunk)
  │   │   ├─→ Dependency parsing
  │   │   ├─→ Relationship type inference
  │   │   └─→ Confidence scoring
  │   │
  │   └─→ [Document Classification]
  │       ├─→ Document type (news, research, wiki, etc.)
  │       ├─→ Domain classification
  │       └─→ Quality assessment
  │
  ├─→ [6] GRAPH SCHEMA VALIDATION
  │   ├─→ Validate entity types against schema
  │   ├─→ Validate relationship types
  │   ├─→ Check property constraints
  │   └─→ Apply confidence thresholds
  │
  ├─→ [7] GRAPH OPERATIONS
  │   ├─→ MERGE Entity Nodes (idempotent)
  │   │   └─→ Upsert: properties + metadata
  │   ├─→ CREATE Relationship Edges
  │   │   └─→ Link: entity → entity + properties
  │   ├─→ CREATE Chunk Relationships
  │   │   └─→ Link: chunk → mentioned_entities
  │   └─→ Update Indexes (if needed)
  │
  ├─→ [8] MEMORY UPDATE
  │   ├─→ Record ingestion timestamp
  │   ├─→ Update knowledge base stats
  │   ├─→ Add to conversation memory (optional)
  │   └─→ Trigger post-processing (optional)
  │
  └─→ OUTPUT: Knowledge Base Updated ✓
```

## Neo4j Data Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                     NEO4J GRAPH STRUCTURE                          │
└─────────────────────────────────────────────────────────────────────┘

NODE TYPES:
──────────

[CHUNK]
├─ Properties:
│  ├─ id: UUID
│  ├─ text: String (512 tokens max)
│  ├─ embedding: Vector[384] ← Neo4j Vector Index
│  ├─ source: String (URL or filename)
│  ├─ timestamp: DateTime
│  ├─ domain: String
│  └─ metadata: JSON
└─ Indexes:
   ├─ Vector Index: "active_rag" (cosine similarity)
   └─ Property Index: source, timestamp


[ENTITY] (Abstract - subtypes below)
├─ Properties:
│  ├─ id: UUID
│  ├─ name: String
│  ├─ entity_type: ENUM (Person|Organization|Location|Concept|Event)
│  ├─ description: String (optional)
│  ├─ properties: JSON (dynamic)
│  ├─ confidence: Float (0.0-1.0)
│  ├─ created_at: DateTime
│  └─ updated_at: DateTime
└─ Subtypes:
   ├─ [PERSON]
   │  ├─ name, birth_date, occupation, biography
   │  └─ relationships: KNOWS, WORKED_WITH, FAMILY_OF
   │
   ├─ [ORGANIZATION]
   │  ├─ name, type, founded_date, location
   │  └─ relationships: PART_OF, FOUNDED_BY, EMPLOYEE_OF
   │
   ├─ [LOCATION]
   │  ├─ name, lat/long, country, type
   │  └─ relationships: PART_OF, NEAR, CONTAINS
   │
   ├─ [CONCEPT]
   │  ├─ name, category, definition
   │  └─ relationships: RELATED_TO, PART_OF
   │
   └─ [EVENT]
       ├─ name, date, location, description
       └─ relationships: OCCURRED_IN, INVOLVED, CAUSED_BY


RELATIONSHIP TYPES:
──────────────────

(:CHUNK)-[MENTIONS]->(ENTITY)
├─ Properties:
│  ├─ frequency: Integer (how many times mentioned)
│  ├─ context: String (snippet around mention)
│  └─ confidence: Float

(:ENTITY)-[RELATIONSHIP]->(ENTITY)
├─ Properties:
│  ├─ type: String (dynamic: KNOWS, WORKS_FOR, RELATED_TO, etc.)
│  ├─ confidence: Float
│  ├─ evidence_count: Integer
│  └─ last_updated: DateTime

(:CHUNK)-[CITED_BY]->(CHUNK)
├─ Properties:
│  └─ reference_type: String


INDEXES:
────────

Vector Index: "active_rag"
├─ Type: vector-1.0
├─ Node Label: CHUNK
├─ Property: embedding
├─ Dimensions: 384
├─ Similarity: COSINE
└─ Search: query(embedding_vector, top_k=3)

Property Indexes:
├─ CHUNK.source (composite with timestamp)
├─ ENTITY.name (full-text search)
├─ ENTITY.entity_type
└─ ENTITY.created_at


CYPHER EXAMPLES:
───────────────

// Vector similarity search
CALL db.index.vector.queryNodes("active_rag", 3, $embedding) 
YIELD node, score 
RETURN node.text, score 
ORDER BY score DESC

// Multi-hop entity traversal
MATCH path = (e1:ENTITY)-[*1..3]-(e2:ENTITY) 
WHERE e1.name = $entity_name 
RETURN path, relationships(path)

// Entity mentions in chunks
MATCH (c:CHUNK)-[r:MENTIONS]->(e:ENTITY)
WHERE e.entity_type = $type
RETURN c.text, e.name, r.frequency
ORDER BY r.frequency DESC
```

## Tool Communication Protocol

```
TOOL SCHEMA FORMAT (OpenAI Compatible):
───────────────────────────────────────

{
  "type": "function",
  "function": {
    "name": "tool_name",
    "description": "What the tool does",
    "parameters": {
      "type": "object",
      "properties": {
        "param1": {"type": "string", "description": "..."},
        "param2": {"type": "integer", "description": "..."}
      },
      "required": ["param1"]
    }
  }
}


TOOL INVOCATION FLOW:
──────────────────────

Agent Request (to LLM):
{
  "role": "assistant",
  "content": "I'll search for information about quantum computing.\n\n" +
             "Action: query_memory\n" +
             "Action Input: {\"query\": \"quantum computing basics\"}\n"
}

Tool Execution:
  1. Parse action: "query_memory"
  2. Parse args: {"query": "quantum computing basics"}
  3. Call _vector_tool.execute(args)
  4. Get observation back

Agent Observation (back to LLM):
{
  "role": "user",
  "content": "Observation: Found 3 relevant chunks:\n" +
             "1. \"Quantum computers leverage superposition...\" (score: 0.92)\n" +
             "2. \"Entanglement enables quantum advantage...\" (score: 0.87)\n" +
             "3. \"Current quantum chips have 50-1000 qubits...\" (score: 0.84)\n" +
             "\nThought: I have good foundational information. Let me search for recent applications."
}

[Loop continues until "Final Answer" generated]


SYNC VS ASYNC EXECUTION:
────────────────────────

Sync Tools:
├─ _execute_tool(name, args) → str
├─ Blocking I/O
├─ Used for: calculator, local operations
└─ Fast for simple operations

Async Tools:
├─ _execute_tool_async(name, args) → Coroutine[str]
├─ Non-blocking I/O
├─ Used for: web_browser, Neo4j queries
└─ Necessary for web scraping (Playwright)
```

## Performance Considerations

```
LATENCY BREAKDOWN (per query):
──────────────────────────────

1. Initialization             50-100 ms
   └─ Config load, Neo4j connect, schema load

2. LLM Thinking (1st iteration) 500-2000 ms
   └─ Depends on model (fast local, slower cloud)

3. Tool Execution (per tool)   50-5000 ms
   ├─ calculator               <50 ms (fast)
   ├─ query_memory             50-200 ms (vector search + network)
   ├─ graph_query              100-500 ms (Cypher traversal)
   ├─ store_memory             200-1000 ms (NLP + writes)
   └─ web_browser              3000-10000 ms (HTTP + Playwright)

4. LLM Response (iteration N)   500-2000 ms
   └─ Same as iteration 1

5. Final answer generation     100-500 ms
   └─ Synthesis + streaming

TOTAL PER QUERY: 2-30 seconds (depends on tools used)


MEMORY USAGE:
─────────────

Base Agent: 200-300 MB
├─ LLM client
├─ Neo4j driver
└─ Tool instances

Per Conversation: +10-50 MB
└─ Conversation history in memory

Recommendations:
├─ Run with 2GB RAM minimum
├─ Use connection pooling (done)
├─ Cache frequent queries (done via query_monitor)
└─ Consider distributed setup for >100 concurrent users


THROUGHPUT:
───────────

Single-threaded: 1-3 queries/second
Multi-threaded (future): 10-100 queries/second

Bottlenecks:
├─ LLM API rate limits
├─ Neo4j graph traversal (at scale)
├─ Web scraping (rate limiting)
└─ Agent loop iterations (sequential)
```

---

**Generated:** 2026-03-31  
**For:** ActiveRAG Model Documentation
