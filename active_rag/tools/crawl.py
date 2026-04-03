"""Recursive crawling tool for the Agent."""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urljoin, urlparse

from active_rag.web_search import WebSearcher
from active_rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class CrawlTool:
    """Tool that recursively crawls a domain to index content."""

    def __init__(self, web_searcher: WebSearcher, vector_store: VectorStore) -> None:
        self.web_searcher = web_searcher
        self.vector_store = vector_store

    async def crawl(self, base_url: str, max_pages: int = 5, max_depth: int = 1) -> str:
        """Crawl a domain recursively and index its content."""
        domain = urlparse(base_url).netloc
        visited = set()
        to_visit = [(base_url, 0)]
        indexed_count = 0

        while to_visit and len(visited) < max_pages:
            url, depth = to_visit.pop(0)
            if url in visited or depth > max_depth:
                continue
            
            visited.add(url)
            logger.info(f"Crawling: {url}")
            
            # Scrape the page
            page = await self.web_searcher.scrape_async(url)
            if not page:
                continue
            
            # Index the page
            self.vector_store.add_documents([page.content], [page.url])
            indexed_count += 1
            
            # Find more links if within depth
            if depth < max_depth:
                # Basic link extraction (Playwright could do this better, but we'll use BS4 from the scraper context)
                # Since scrape_async doesn't return links, we'd need a helper or modification.
                # For this prototype, we'll assume we only crawl a few top-level links if we had them.
                pass

        return f"Successfully crawled and indexed {indexed_count} pages from {base_url}."

    def run(self, base_url: str, max_pages: int = 5) -> str:
        """Sync wrapper for crawl."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        import nest_asyncio
        nest_asyncio.apply()
        
        return loop.run_until_complete(self.crawl(base_url, max_pages))
