"""LRU cache for frequent graph query patterns.

Provides configurable TTL, max size, and hit/miss metrics tracking.
Cache can be invalidated on graph writes.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    def to_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "invalidations": self.invalidations,
            "total_requests": self.total_requests,
            "hit_rate": f"{self.hit_rate:.2%}",
        }


@dataclass
class _CacheEntry:
    """Internal cache entry with TTL tracking."""
    value: Any
    created_at: float
    ttl: float

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl


class GraphCache:
    """LRU cache for graph query results with TTL and metrics."""

    def __init__(self, max_size: int = 256, default_ttl: float = 300.0) -> None:
        """
        Args:
            max_size: Maximum number of cached entries.
            default_ttl: Default time-to-live in seconds (5 minutes).
        """
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._metrics = CacheMetrics()

    @property
    def metrics(self) -> CacheMetrics:
        return self._metrics

    @property
    def size(self) -> int:
        return len(self._cache)

    @staticmethod
    def _make_key(query_type: str, **params) -> str:
        """Generate a deterministic cache key from query type and parameters."""
        parts = [query_type]
        for k in sorted(params.keys()):
            parts.append(f"{k}={params[k]}")
        raw = "|".join(parts)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, query_type: str, **params) -> Optional[Any]:
        """Look up a cached result.

        Args:
            query_type: Type of query (e.g., "neighborhood", "find_paths")
            **params: Query parameters for key generation

        Returns:
            Cached value or None
        """
        key = self._make_key(query_type, **params)

        if key not in self._cache:
            self._metrics.misses += 1
            return None

        entry = self._cache[key]

        # Check TTL
        if entry.is_expired:
            del self._cache[key]
            self._metrics.misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._metrics.hits += 1
        return entry.value

    def put(self, query_type: str, value: Any, ttl: Optional[float] = None, **params) -> None:
        """Store a result in the cache.

        Args:
            query_type: Type of query
            value: Result to cache
            ttl: Optional TTL override
            **params: Query parameters for key generation
        """
        key = self._make_key(query_type, **params)
        actual_ttl = ttl if ttl is not None else self._default_ttl

        # Evict if at capacity
        if key not in self._cache and len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)  # Remove least recently used
            self._metrics.evictions += 1

        self._cache[key] = _CacheEntry(
            value=value,
            created_at=time.time(),
            ttl=actual_ttl,
        )
        self._cache.move_to_end(key)

    def invalidate(self, query_type: Optional[str] = None) -> int:
        """Invalidate cache entries.

        Args:
            query_type: If provided, only invalidate entries matching this type.
                       If None, clear entire cache.

        Returns:
            Number of entries invalidated.
        """
        if query_type is None:
            count = len(self._cache)
            self._cache.clear()
            self._metrics.invalidations += count
            return count

        # Partial invalidation by key prefix
        prefix = hashlib.md5(query_type.encode()).hexdigest()[:8]
        to_remove = [k for k in self._cache if k.startswith(prefix)]
        for k in to_remove:
            del self._cache[k]
        self._metrics.invalidations += len(to_remove)
        return len(to_remove)

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        expired = [k for k, v in self._cache.items() if v.is_expired]
        for k in expired:
            del self._cache[k]
        return len(expired)
