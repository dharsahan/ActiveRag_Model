"""Web Browser tool wrapping Playwright WebSearcher with Continuous Learning."""

import json
import logging
import asyncio
from typing import List, Optional

from active_rag.config import Config
from active_rag.web_search import WebSearcher, ScrapedPage
from active_rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

def get_schema():
    return {
        "type": "function",
        "function": {
            "name": "web_browser",
            "description": (
                "Searches the live internet and renders JavaScript pages to retrieve fresh information. "
                "Automatically indexes found content into the knowledge base for future use (Continuous Learning)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up securely via DuckDuckGo and scrape.",
                    },
                    "headless": {
                        "type": "boolean",
                        "description": "Whether to run the browser in headless mode. Set to false to see the browser window.",
                        "default": True
                    }
                },
                "required": ["query"],
            },
        }
    }


class WebBrowserTool:
    def __init__(self, config: Config, vector_store: Optional[VectorStore] = None):
        self._config = config
        self._searcher = WebSearcher(config)
        self._vector_store = vector_store or VectorStore(config)
        
        # Graph components (optional/lazy)
        self._graph_client = None
        self._entity_extractor = None
        self._initialize_graph()
        
        self.schema = get_schema()

    def _initialize_graph(self):
        """Initialize graph components if enabled."""
        if not self._config.enable_graph_features:
            return

        try:
            # Reuse vector store's neo4j client if possible
            if hasattr(self._vector_store, "_neo4j"):
                self._graph_client = self._vector_store._neo4j
            else:
                from active_rag.knowledge_graph.neo4j_client import Neo4jClient
                self._graph_client = Neo4jClient(
                    self._config.neo4j_uri,
                    self._config.neo4j_username,
                    self._config.neo4j_password,
                )
            
            from active_rag.nlp_pipeline.entity_extractor import EntityExtractor
            self._entity_extractor = EntityExtractor()
            logger.info("WebBrowserTool: Graph enrichment enabled.")
        except Exception as e:
            logger.warning(f"WebBrowserTool: Graph enrichment disabled (Neo4j unavailable): {e}")
            self._graph_client = None

    def _update_knowledge_systems(self, pages: List[ScrapedPage], query: str):
        """Update both Vector and Graph knowledge systems with new data."""
        if not pages:
            return

        # 1. Update Vector Store (Neo4j)
        chunk_ids_map = {} # map page index to its chunk IDs
        try:
            for i, page in enumerate(pages):
                cids = self._vector_store.add_documents(
                    contents=[page.content],
                    source_urls=[page.url]
                )
                chunk_ids_map[i] = cids
            logger.info(f"Indexed {len(pages)} pages into vector store.")
        except Exception as e:
            logger.warning(f"Failed to index pages into vector store: {e}")

        # 2. Update Knowledge Graph (Neo4j)
        if self._graph_client and self._entity_extractor:
            try:
                from active_rag.schemas.entities import ContentDomain
                from active_rag.nlp_pipeline.relation_extractor import RelationExtractor
                
                rel_extractor = RelationExtractor(self._config)
                
                # Process top 2 pages for entity and relation extraction
                for i, page in enumerate(pages[:2]):
                    entities = self._entity_extractor.extract_entities(page.content, ContentDomain.MIXED_WEB)
                    
                    # Add top entities to graph
                    for entity in entities[:5]:
                        try:
                            props = entity["properties"].copy()
                            props["source_url"] = page.url
                            props["source_type"] = "web"
                            props["discovered_via"] = "web_browser_tool"
                            props["query"] = query
                            
                            self._graph_client.create_entity(entity["label"], props)
                            
                            # Link Chunk to Entity (MENTIONS)
                            if i in chunk_ids_map:
                                for cid in chunk_ids_map[i]:
                                    self._graph_client.create_relationship(
                                        subject_id=cid,
                                        subject_label="Chunk",
                                        predicate="MENTIONS",
                                        object_id=props["id"],
                                        object_label=entity["label"],
                                        properties={"context": "web_scrape"}
                                    )
                        except Exception:
                            continue
                    
                    # Extract and create relationships between entities
                    if len(entities) >= 2:
                        relations = rel_extractor.extract_relations(page.content, entities)
                        for rel in relations:
                            try:
                                self._graph_client.create_relationship(
                                    subject_id=rel["subject_id"],
                                    subject_label=rel["subject_label"],
                                    predicate=rel["predicate"],
                                    object_id=rel["object_id"],
                                    object_label=rel["object_label"],
                                    properties=rel.get("properties", {})
                                )
                            except Exception:
                                continue
                                
                logger.info("Knowledge graph enriched with new entities and relationships.")
            except Exception as e:
                logger.warning(f"Failed to update knowledge graph: {e}")

    def execute(self, kwargs: dict) -> str:
        """Synchronous wrapper for backward compatibility."""
        query = kwargs.get("query", "")
        headless = kwargs.get("headless", None)
        if not query:
            return "Error: no query provided."
            
        pages = self._searcher.search_and_scrape(query, headless=headless)
        if not pages:
            return "No results found or pages failed to render."
            
        # Background/Inline knowledge update
        self._update_knowledge_systems(pages, query)

        result_text = []
        for i, p in enumerate(pages):
            result_text.append(f"--- Source {i+1}: {p.url} ---\n{p.content[:1500]}")
            
        return "\n\n".join(result_text)

    async def execute_async(self, kwargs: dict) -> str:
        """Asynchronous execution for FastAPI/streaming environments."""
        query = kwargs.get("query", "")
        headless = kwargs.get("headless", None)
        if not query:
            return "Error: no query provided."
            
        pages = await self._searcher.search_and_scrape_async(query, headless=headless)
        if not pages:
            return "No results found or pages failed to render."
            
        # Inline knowledge update (could be backgrounded)
        self._update_knowledge_systems(pages, query)

        result_text = []
        for i, p in enumerate(pages):
            result_text.append(f"--- Source {i+1}: {p.url} ---\n{p.content[:1500]}")
            
        return "\n\n".join(result_text)
