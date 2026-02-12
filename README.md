# ActiveRag_Model

A **Refined Active RAG (Retrieval-Augmented Generation)** system that intelligently decides whether to answer directly from the LLM or retrieve external knowledge before responding.

## Architecture

![Refined Active RAG Architecture](1770878008505.png)

### Pipeline Flow

1. **User Query** — the user asks a question.
2. **AI Model Confidence Check** — the LLM evaluates whether it can answer the question reliably.
   - **High Confidence** → **Generate Answer Directly** → **Final Answer to User**
   - **Don't Know / Hallucination Risk** → proceed to step 3.
3. **Check Vector Memory / RAG** — search the local ChromaDB vector store for relevant documents.
   - **Data Found** → **Retrieve Context & Citations** → **Generate Answer with Citations**
   - **Data Missing** → proceed to step 4.
4. **Search Data Online** — use DuckDuckGo to find relevant web pages.
5. **Scrape & Extract Content** — download and extract text from the search results.
6. **Update Vector DB** — index the new content (with source URLs) into the vector store.
7. **Closed Loop** — re-query the vector store, retrieve context & citations, and generate the final answer.

## Project Structure

```
active_rag/
├── __init__.py               # Package exports
├── config.py                 # Centralized configuration (env vars / dataclass)
├── confidence_checker.py     # AI Model Confidence Check
├── vector_store.py           # ChromaDB vector store (search / add documents)
├── web_search.py             # Web search + scrape & extract content
├── answer_generator.py       # Generate answers (direct / with citations)
└── pipeline.py               # Main orchestrator (full Active RAG flow)
tests/
├── test_config.py
├── test_confidence_checker.py
├── test_vector_store.py
├── test_answer_generator.py
└── test_pipeline.py
main.py                       # CLI entry point
requirements.txt
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install and start Ollama (https://ollama.com)
# Then pull a model:
ollama pull llama3.2
```

### Configuration

All settings can be configured via environment variables or a `.env` file:

| Variable               | Default                          | Description                                        |
|------------------------|----------------------------------|----------------------------------------------------|
| `OLLAMA_BASE_URL`      | `http://localhost:11434/v1`      | Ollama API base URL (OpenAI-compatible endpoint)   |
| `MODEL_NAME`           | `llama3.2`                       | Ollama model name                                  |
| `CONFIDENCE_THRESHOLD` | `0.7`                            | Minimum confidence to skip RAG (0.0–1.0)           |
| `CHROMA_PERSIST_DIR`   | `./chroma_db`                    | ChromaDB persistence directory                     |
| `COLLECTION_NAME`      | `active_rag`                     | ChromaDB collection name                           |
| `TOP_K`                | `3`                              | Number of documents to retrieve                    |
| `MAX_SEARCH_RESULTS`   | `3`                              | Maximum web search results                         |

## Usage

### CLI — single query

```bash
python main.py "What is retrieval-augmented generation?"
```

### CLI — interactive mode

```bash
python main.py
```

### Programmatic

```python
from active_rag import ActiveRAGPipeline, Config

config = Config()  # Uses Ollama at localhost:11434 by default
pipeline = ActiveRAGPipeline(config)
result = pipeline.run("What is quantum computing?")

print(result.answer.text)
print(result.path)        # "direct" | "rag_memory" | "rag_web"
print(result.answer.citations)
```

## Testing

```bash
python -m pytest tests/ -v
```