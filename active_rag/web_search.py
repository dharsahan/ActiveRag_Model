"""Search Data Online + Scrape & Extract Content.

Performs web searches and scrapes content from the results so it can
be indexed into the vector store.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from active_rag.config import Config

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 10  # seconds


@dataclass
class ScrapedPage:
    """Content scraped from a single web page."""

    url: str
    content: str


class WebSearcher:
    """Searches the web and scrapes content from result pages."""

    def __init__(self, config: Config) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Search Data Online
    # ------------------------------------------------------------------
    def search(self, query: str) -> list[str]:
        """Return a list of URLs relevant to *query*."""
        try:
            results = DDGS().text(
                query, max_results=self._config.max_search_results
            )
            return [r["href"] for r in results if "href" in r]
        except Exception:
            logger.exception("Web search failed for query: %s", query)
            return []

    # ------------------------------------------------------------------
    # Scrape & Extract Content
    # ------------------------------------------------------------------
    def scrape(self, url: str) -> ScrapedPage | None:
        """Fetch *url* and return its extracted text content."""
        try:
            resp = requests.get(
                url,
                timeout=_REQUEST_TIMEOUT,
                headers={"User-Agent": "ActiveRAG/1.0"},
            )
            resp.raise_for_status()
        except requests.RequestException:
            logger.warning("Failed to fetch URL: %s", url)
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script/style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        # Truncate very long pages to keep vector-store entries manageable.
        text = text[:5000]
        if not text.strip():
            return None
        return ScrapedPage(url=url, content=text)

    def search_and_scrape(self, query: str) -> list[ScrapedPage]:
        """Search the web for *query* and scrape all result pages."""
        urls = self.search(query)
        pages: list[ScrapedPage] = []
        for url in urls:
            page = self.scrape(url)
            if page:
                pages.append(page)
        return pages
