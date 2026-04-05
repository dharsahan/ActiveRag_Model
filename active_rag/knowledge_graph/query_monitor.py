"""Query performance monitoring for graph operations.

Tracks execution time, cache hit rates, and traversal depth.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class QueryMetric:
    """Single query execution metric."""
    query_type: str
    duration_ms: float
    cache_hit: bool
    graph_hops: int = 0
    result_count: int = 0
    timestamp: float = field(default_factory=time.time)


class QueryMonitor:
    """Tracks and reports query performance metrics."""

    def __init__(self, max_history: int = 1000) -> None:
        self._history: List[QueryMetric] = []
        self._max_history = max_history

    @contextmanager
    def track(self, query_type: str, cache_hit: bool = False, graph_hops: int = 0):
        """Context manager to track query execution time.

        Usage:
            with monitor.track("multi_hop", graph_hops=2) as metric:
                result = graph_ops.multi_hop_query(...)
                metric.result_count = len(result["entities"])
        """
        metric = QueryMetric(
            query_type=query_type,
            duration_ms=0.0,
            cache_hit=cache_hit,
            graph_hops=graph_hops,
        )
        start = time.perf_counter()
        try:
            yield metric
        finally:
            metric.duration_ms = (time.perf_counter() - start) * 1000
            self._record(metric)

    def record(self, query_type: str, duration_ms: float, cache_hit: bool = False,
               graph_hops: int = 0, result_count: int = 0) -> None:
        """Record a query metric directly."""
        self._record(QueryMetric(
            query_type=query_type,
            duration_ms=duration_ms,
            cache_hit=cache_hit,
            graph_hops=graph_hops,
            result_count=result_count,
        ))

    def _record(self, metric: QueryMetric) -> None:
        """Store a metric, evicting old entries if needed."""
        self._history.append(metric)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def get_performance_report(self) -> Dict:
        """Generate a performance summary report.

        Returns:
            Dict with per-query-type stats and overall metrics.
        """
        if not self._history:
            return {"total_queries": 0, "query_types": {}}

        # Group by query type
        by_type: Dict[str, List[QueryMetric]] = {}
        for m in self._history:
            by_type.setdefault(m.query_type, []).append(m)

        type_stats = {}
        for qtype, metrics in by_type.items():
            durations = [m.duration_ms for m in metrics]
            cache_hits = sum(1 for m in metrics if m.cache_hit)
            type_stats[qtype] = {
                "count": len(metrics),
                "avg_ms": sum(durations) / len(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "p95_ms": sorted(durations)[int(len(durations) * 0.95)] if len(durations) > 1 else durations[0],
                "cache_hit_rate": f"{cache_hits / len(metrics):.2%}",
            }

        all_durations = [m.duration_ms for m in self._history]
        total_cache_hits = sum(1 for m in self._history if m.cache_hit)

        return {
            "total_queries": len(self._history),
            "avg_duration_ms": sum(all_durations) / len(all_durations),
            "overall_cache_hit_rate": f"{total_cache_hits / len(self._history):.2%}",
            "query_types": type_stats,
        }

    def clear(self) -> None:
        """Clear all recorded metrics."""
        self._history.clear()
