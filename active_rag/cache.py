"""Response caching for improved performance."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from diskcache import Cache

from active_rag.config import Config

logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """A cached pipeline response."""
    
    answer_text: str
    answer_citations: list[str]
    answer_source: str
    confidence_score: float | None
    confidence_reasoning: str | None
    path: str
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CachedResponse":
        return cls(**data)


class ResponseCache:
    """Disk-based cache for pipeline responses."""
    
    def __init__(self, config: Config, cache_dir: str | None = None) -> None:
        self._config = config
        cache_path = cache_dir or str(Path(config.chroma_persist_dir).parent / ".cache")
        self._cache = Cache(cache_path)
        self._ttl = 3600 * 24  # 24 hours default TTL
    
    def _make_key(self, query: str) -> str:
        """Create a cache key from the query."""
        # Normalize query for better cache hits
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def get(self, query: str) -> CachedResponse | None:
        """Get a cached response for the query."""
        key = self._make_key(query)
        try:
            data = self._cache.get(key)
            if data:
                logger.debug("Cache hit for query: %s", query[:50])
                return CachedResponse.from_dict(json.loads(data))
        except Exception:
            logger.debug("Cache miss for query: %s", query[:50])
        return None
    
    def set(self, query: str, response: CachedResponse) -> None:
        """Cache a response for the query."""
        key = self._make_key(query)
        try:
            self._cache.set(key, json.dumps(response.to_dict()), expire=self._ttl)
            logger.debug("Cached response for query: %s", query[:50])
        except Exception:
            logger.warning("Failed to cache response")
    
    def clear(self) -> None:
        """Clear all cached responses."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "volume": self._cache.volume(),
        }
