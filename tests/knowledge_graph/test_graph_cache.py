"""Tests for the graph query cache."""

import time
import pytest

from active_rag.knowledge_graph.graph_cache import GraphCache, CacheMetrics


class TestGraphCache:
    """Tests for LRU cache with TTL."""

    def test_put_and_get(self):
        cache = GraphCache(max_size=10)
        cache.put("neighborhood", {"id": "123"}, entity_id="e1")
        result = cache.get("neighborhood", entity_id="e1")
        assert result == {"id": "123"}

    def test_cache_miss(self):
        cache = GraphCache()
        result = cache.get("neighborhood", entity_id="nonexistent")
        assert result is None
        assert cache.metrics.misses == 1

    def test_hit_tracking(self):
        cache = GraphCache()
        cache.put("test", "value", key="k1")
        cache.get("test", key="k1")
        cache.get("test", key="k1")
        assert cache.metrics.hits == 2
        assert cache.metrics.hit_rate > 0.0

    def test_ttl_expiry(self):
        cache = GraphCache(default_ttl=0.1)
        cache.put("test", "value", key="k1")
        time.sleep(0.15)
        result = cache.get("test", key="k1")
        assert result is None

    def test_max_size_eviction(self):
        cache = GraphCache(max_size=3)
        cache.put("t", "v1", key="a")
        cache.put("t", "v2", key="b")
        cache.put("t", "v3", key="c")
        cache.put("t", "v4", key="d")  # Should evict "a"
        assert cache.size == 3
        assert cache.metrics.evictions == 1
        assert cache.get("t", key="a") is None

    def test_invalidate_all(self):
        cache = GraphCache()
        cache.put("t1", "v1", key="a")
        cache.put("t2", "v2", key="b")
        count = cache.invalidate()
        assert count == 2
        assert cache.size == 0

    def test_cleanup_expired(self):
        cache = GraphCache(default_ttl=0.1)
        cache.put("t", "v1", key="a")
        cache.put("t", "v2", key="b")
        time.sleep(0.15)
        removed = cache.cleanup_expired()
        assert removed == 2
        assert cache.size == 0

    def test_metrics_to_dict(self):
        metrics = CacheMetrics(hits=10, misses=5)
        d = metrics.to_dict()
        assert d["total_requests"] == 15
        assert "66" in d["hit_rate"]  # ~66.67%

    def test_deterministic_key_generation(self):
        cache = GraphCache()
        key1 = cache._make_key("test", a="1", b="2")
        key2 = cache._make_key("test", b="2", a="1")  # Different order, same key
        assert key1 == key2
