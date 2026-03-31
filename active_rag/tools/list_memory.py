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
    def __init__(self, config: Config):
        self._config = config
        self.schema = get_schema()
        
    def execute(self, kwargs: dict) -> str:
        try:
            # We initialize a new vector store instance purely to access the raw collection
            store = VectorStore(self._config)
            all_data = store._collection.get(include=["documents", "metadatas"])
            
            if not all_data or not all_data.get("documents"):
                return "The database is currently completely empty."
            
            docs = all_data["documents"]
            metas = all_data["metadatas"]
            
            snippets = []
            for doc, meta in zip(docs, metas):
                src = meta.get("source_url", "Unknown Source") if meta else "Unknown Source"
                snippets.append(f"[{src}]: {doc}")
                
            return "Current Database Index Dump:\n\n" + "\n---\n".join(snippets)
            
        except Exception as e:
            return f"Error listing memories: {e}"
