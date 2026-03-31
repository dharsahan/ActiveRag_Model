"""Enhanced web search with parallel fetching, retries, and fallbacks.

Performs web searches and scrapes content from the results so it can
be indexed into the vector store.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
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
    # Scrape & Extract Content (with smart extraction)
    # ------------------------------------------------------------------
    def scrape(self, url: str) -> ScrapedPage | None:
        """Fetch *url* and return its extracted text content."""
        try:
            resp = self._session.get(url, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.debug("Failed to fetch URL: %s - %s", url, str(e)[:50])
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract title
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        # Remove unwanted elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", 
                         "advertisement", "noscript", "iframe", "form"]):
            tag.decompose()

        # Try to find main content
        main_content = None
        for selector in ["main", "article", "[role='main']", ".content", ".post-content", "#content"]:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if main_content:
            text = main_content.get_text(separator="\n", strip=True)
        else:
            text = soup.get_text(separator="\n", strip=True)

        # Clean up text
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        text = "\n".join(lines)
        
        # Semantic chunking instead of hard truncation
        from active_rag.chunker import TextChunker
        chunker = TextChunker(chunk_size=2000, overlap=200)
        chunks = chunker.chunk(text)
        text = chunks[0] if chunks else text
        
        if not text.strip() or len(text) < 100:
            return None

        word_count = len(text.split())
        return ScrapedPage(url=url, content=text, title=title, word_count=word_count)

    def _scrape_with_progress(self, url: str, index: int, total: int) -> ScrapedPage | None:
        """Scrape with progress reporting."""
        self._progress_callback(f"Scraping page {index + 1}/{total}...")
        return self.scrape(url)

    # ------------------------------------------------------------------
    # Parallel Search and Scrape
    # ------------------------------------------------------------------
    def search_and_scrape(self, query: str) -> list[ScrapedPage]:
        """Search the web for *query* and scrape all result pages in parallel."""
        results = self.search(query)
        if not results:
            return []

        urls = [r["url"] for r in results]
        pages: list[ScrapedPage] = []
        
        self._progress_callback(f"Found {len(urls)} results, scraping...")

        # Parallel scraping for speed
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            future_to_url = {
                executor.submit(self._scrape_with_progress, url, i, len(urls)): url
                for i, url in enumerate(urls)
            }
            
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    page = future.result()
                    if page:
                        pages.append(page)
                        # Stop if we have enough good pages
                        if len(pages) >= self._config.max_search_results:
                            break
                except Exception as e:
                    logger.debug("Error scraping %s: %s", url, str(e)[:50])

        # Sort by word count (prefer more content)
        pages.sort(key=lambda p: p.word_count, reverse=True)
        
        self._progress_callback(f"Successfully scraped {len(pages)} pages")
        return pages[:self._config.max_search_results]

    def search_urls_only(self, query: str) -> list[str]:
        """Return just URLs for backward compatibility."""
        results = self.search(query)
        return [r["url"] for r in results]
