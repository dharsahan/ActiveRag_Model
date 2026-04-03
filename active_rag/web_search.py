"""Enhanced web search with parallel fetching, retries, and fallbacks.

Performs web searches and scrapes content from the results so it can
be indexed into the vector store.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable

import requests
import trafilatura
from bs4 import BeautifulSoup
from ddgs import DDGS
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from active_rag.config import Config

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 10  # seconds
_MAX_WORKERS = 5  # parallel scraping threads


@dataclass
class ScrapedPage:
    """Content scraped from a single web page."""

    url: str
    content: str
    title: str = ""
    word_count: int = 0


class WebSearcher:
    """Searches the web and scrapes content from result pages."""

    def __init__(
        self,
        config: Config,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._config = config
        self._progress_callback = progress_callback or (lambda x: None)
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })

    # ------------------------------------------------------------------
    # Search Data Online (with retry)
    # ------------------------------------------------------------------
    def search(self, query: str) -> list[dict[str, str]]:
        """Return a list of search results with URLs and titles."""
        self._progress_callback("Searching the web...")
        try:
            return self._search_with_retry(query)
        except Exception as e:
            logger.warning("Web search failed after retries: %s", str(e)[:100])
            return []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda rs: logger.debug("Search retry %d", rs.attempt_number),
    )
    def _search_with_retry(self, query: str) -> list[dict[str, str]]:
        """Internal search method with retry logic."""
        results = list(DDGS().text(query, max_results=self._config.max_search_results + 2))
        return [
            {"url": r["href"], "title": r.get("title", "")}
            for r in results
            if "href" in r
        ]

    # ------------------------------------------------------------------
    # Scrape & Extract Content (with async playwright & trafilatura)
    # ------------------------------------------------------------------
    async def scrape_async(self, url: str) -> ScrapedPage | None:
        """Fetch *url* using async playwright and return its extracted text content."""
        html_content = ""
        title = ""
        
        try:
            async with async_playwright() as p:
                async with await p.chromium.launch(headless=True) as browser:
                    async with await browser.new_context(
                        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
                    ) as context:
                        
                        # Block noisy resources
                        async def handle_route(route):
                            try:
                                if route.request.resource_type in ["image", "media", "font", "stylesheet"]:
                                    await route.abort()
                                else:
                                    await route.continue_()
                            except Exception:
                                # Silently ignore errors if the target is already closed
                                pass
                        
                        await context.route("**/*", handle_route)
                        
                        page = await context.new_page()
                        try:
                            await page.goto(url, wait_until="domcontentloaded", timeout=_REQUEST_TIMEOUT * 1000)
                            # Wait for potential JS rendering
                            await asyncio.sleep(1)
                            html_content = await page.content()
                            title = await page.title()
                        finally:
                            # Ensure we unroute before the context manager closes the context
                            try:
                                await context.unroute("**/*")
                            except:
                                pass

                # Use trafilatura for high-quality extraction
                extracted_text = trafilatura.extract(html_content, include_comments=False, include_tables=True)
                
                if not extracted_text:
                    # Fallback to BeautifulSoup if trafilatura fails
                    soup = BeautifulSoup(html_content, "html.parser")
                    for s in soup(["script", "style", "nav", "footer", "header", "aside"]):
                        s.decompose()
                    extracted_text = soup.get_text(separator="\n")

                if not extracted_text or len(extracted_text) < 100:
                    return None

                # Clean up text
                lines = [line.strip() for line in extracted_text.split("\n") if line.strip()]
                text = "\n".join(lines)
                
                # Semantic chunking
                from active_rag.chunker import TextChunker
                chunker = TextChunker(chunk_size=2000, overlap=200)
                chunks = chunker.chunk(text)
                text = chunks[0] if chunks else text
                
                word_count = len(text.split())
                return ScrapedPage(url=url, content=text, title=title, word_count=word_count)
                
        except Exception as e:
            logger.debug("Failed to fetch/render URL using Playwright: %s - %s", url, str(e)[:50])
            return None

    def scrape(self, url: str) -> ScrapedPage | None:
        """Synchronous wrapper for scrape_async (for backward compatibility)."""
        try:
            # Try to get existing loop or create new one
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if loop.is_running():
                # For cases where an event loop is already running (e.g. FastAPI)
                import nest_asyncio
                nest_asyncio.apply()
            
            return loop.run_until_complete(self.scrape_async(url))
        except Exception as e:
            logger.error(f"Sync scrape failed for {url}: {e}")
            return None

    # ------------------------------------------------------------------
    # Parallel Search and Scrape
    # ------------------------------------------------------------------
    async def search_and_scrape_async(self, query: str) -> list[ScrapedPage]:
        """Search the web for *query* and scrape all result pages concurrently."""
        results = self.search(query)
        if not results:
            return []

        urls = [r["url"] for r in results][:self._config.max_search_results]
        self._progress_callback(f"Found {len(urls)} results, scraping concurrently...")

        # Concurrently scrape all URLs
        tasks = [self.scrape_async(url) for url in urls]
        pages_raw = await asyncio.gather(*tasks)
        
        pages = [p for p in pages_raw if p is not None]
        
        # Sort by word count (prefer more content)
        pages.sort(key=lambda p: p.word_count, reverse=True)
        
        self._progress_callback(f"Successfully scraped {len(pages)} pages")
        return pages

    def search_and_scrape(self, query: str) -> list[ScrapedPage]:
        """Synchronous wrapper for search_and_scrape_async."""
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                
            return loop.run_until_complete(self.search_and_scrape_async(query))
        except Exception as e:
            logger.error(f"Sync search_and_scrape failed: {e}")
            return []

    def search_urls_only(self, query: str) -> list[str]:
        """Return just URLs for backward compatibility."""
        results = self.search(query)
        return [r["url"] for r in results]
