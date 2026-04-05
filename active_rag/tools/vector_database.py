"""Vector Database query tool for previous knowledge."""

import json
from active_rag.config import Config
from active_rag.vector_store import VectorStore

def get_schema():
    return {
        "type": "function",
        "function": {
            "name": "query_memory",
            "description": "Searches the local knowledge base (vector database) for previously ingested documents, PDFs, or old queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search string to find in memory.",
                    }
                },
                "required": ["query"],
            },
        }
    }

class VectorDatabaseTool:
    def __init__(self, config: Config):
        self._store = VectorStore(config)
        self.schema = get_schema()
        
    def execute(self, kwargs: dict) -> str:
        query = kwargs.get("query", "")
        if not query:
            return "Error: no query provided."
            
        result = self._store.search(query)
        if not result.found:
            return "No matching documents found in the knowledge base."
            
        snippets = []
        for r in result.results:
            snippets.append(f"[{r.source_url}]: {r.content}")
            
        return "\n\n".join(snippets)

    async def execute_async(self, kwargs: dict) -> str:
        """Asynchronous execution (already calls sync search but fits the async interface)."""
        return self.execute(kwargs)

    def count(self) -> int:
        """Return number of documents in the vector store."""
        return self._store.count()
