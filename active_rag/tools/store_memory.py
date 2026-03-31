"""Store Memory tool to permanently memorize facts via ChromaDB."""

import json
from active_rag.config import Config
from active_rag.vector_store import VectorStore

def get_schema():
    return {
        "type": "function",
        "function": {
            "name": "store_memory",
            "description": "Permanently memorize a specific fact or detail into the vector database. Use this when the user says 'remember this', 'store this', or states a specific fact they want tracked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The exact text or fact to permanently memorize.",
                    }
                },
                "required": ["fact"],
            },
        }
    }

class StoreMemoryTool:
    def __init__(self, config: Config):
        self._store = VectorStore(config)
        self.schema = get_schema()
        
    def execute(self, kwargs: dict) -> str:
        fact = kwargs.get("fact", "")
        if not fact:
            return "Error: no fact provided to store."
            
        try:
            self._store.add_documents(
                contents=[fact],
                source_urls=["User Injected Memory"],
            )
            return "Successfully memorized into the local database."
        except Exception as e:
            return f"Error storing memory: {e}"
