"""Web Browser tool wrapping Playwright WebSearcher."""

import json
from active_rag.config import Config
from active_rag.web_search import WebSearcher

def get_schema():
    return {
        "type": "function",
        "function": {
            "name": "web_browser",
            "description": "Searches the live internet and renders JavaScript pages to retrieve fresh information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up securely via DuckDuckGo and scrape.",
                    }
                },
                "required": ["query"],
            },
        }
    }


class WebBrowserTool:
    def __init__(self, config: Config):
        self._searcher = WebSearcher(config)
        self.schema = get_schema()
        
    def execute(self, kwargs: dict) -> str:
        query = kwargs.get("query", "")
        if not query:
            return "Error: no query provided."
            
        pages = self._searcher.search_and_scrape(query)
        if not pages:
            return "No results found or pages failed to render."
            
        result_text = []
        for i, p in enumerate(pages):
            result_text.append(f"--- Source {i+1}: {p.url} ---\n{p.content[:1500]}")
            
        return "\n\n".join(result_text)
