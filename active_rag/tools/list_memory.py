"""List Memory tool to fetch the entire database dump globally."""

import json
from active_rag.config import Config
from active_rag.vector_store import VectorStore

def get_schema():
    return {
        "type": "function",
        "function": {
            "name": "list_memory",
            "description": "Retrieve absolutely everything stored in the vector database globally. Use this when the user asks to 'list all', 'retrieve all', 'show indexes' or 'dump database'.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        }
    }

class ListMemoryTool:
    def __init__(self, config: Config, vector_store: Optional[VectorStore] = None) -> None:
        self._config = config
        self._vector_store = vector_store
        self.schema = get_schema()

    def execute(self, kwargs: dict) -> str:
        try:
            store = self._vector_store or VectorStore(self._config)
            all_docs = store.get_all_documents()
            
            if not all_docs:
                return "The database is currently completely empty."
            
            snippets = []
            for doc_info in all_docs:
                src = doc_info.get("source_url", "Unknown Source")
                content = doc_info.get("content", "")
                # Escape square brackets for Rich
                src = src.replace("[", "[[").replace("]", "]]")
                snippets.append(f"[{src}]: {content}")
                
            return "Current Database Index Dump:\n\n" + "\n---\n".join(snippets)
            
        except Exception as e:
            return f"Error listing memories: {e}"
