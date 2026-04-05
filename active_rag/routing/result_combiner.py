"""Result combiner for hybrid vector-graph RAG.

Merges, deduplicates, and ranks results from vector similarity search
and graph traversal into a unified ranked list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SourcedChunk:
    """A piece of retrieved context with provenance."""
    content: str
    source: str                     # "vector" or "graph"
    score: float                    # Normalised 0.0–1.0 (higher = better)
    metadata: dict = field(default_factory=dict)
    reasoning_path: Optional[str] = None  # Graph-only: human-readable path


@dataclass
class CombinedResult:
    """Unified result after merging vector + graph retrieval."""
    chunks: List[SourcedChunk]
    vector_count: int
    graph_count: int
    strategy_used: str  # "vector", "graph", "hybrid"

    @property
    def context_text(self) -> str:
        """Concatenate all chunks into a single context string for the LLM."""
        parts: list[str] = []
        for i, chunk in enumerate(self.chunks, 1):
            header = f"[Source {i}: {chunk.source}"
            if chunk.reasoning_path:
                header += f" | Path: {chunk.reasoning_path}"
            header += f" | Score: {chunk.score:.2f}]"
            parts.append(f"{header}\n{chunk.content}")
        return "\n\n".join(parts)


class ResultCombiner:
    """Combines and ranks results from multiple retrieval sources."""

    def __init__(
        self,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
    ) -> None:
        """
        Args:
            vector_weight: Weight for vector similarity scores (0.0–1.0).
            graph_weight: Weight for graph-derived scores (0.0–1.0).
        """
        self._vector_weight = vector_weight
        self._graph_weight = graph_weight

    def combine(
        self,
        vector_results: List[SourcedChunk] | None = None,
        graph_results: List[SourcedChunk] | None = None,
        top_k: int = 5,
    ) -> CombinedResult:
        """Merge and rank results from vector and graph sources.

        Args:
            vector_results: Chunks from ChromaDB vector search.
            graph_results: Chunks from Neo4j graph traversal.
            top_k: Maximum number of results to return.

        Returns:
            CombinedResult with ranked, deduplicated chunks.
        """
        vector_results = vector_results or []
        graph_results = graph_results or []

        # Apply source weights
        weighted: List[SourcedChunk] = []
        for chunk in vector_results:
            weighted.append(SourcedChunk(
                content=chunk.content,
                source="vector",
                score=chunk.score * self._vector_weight,
                metadata=chunk.metadata,
                reasoning_path=chunk.reasoning_path,
            ))
        for chunk in graph_results:
            weighted.append(SourcedChunk(
                content=chunk.content,
                source="graph",
                score=chunk.score * self._graph_weight,
                metadata=chunk.metadata,
                reasoning_path=chunk.reasoning_path,
            ))

        # Deduplicate by content similarity (exact match on first 200 chars)
        deduped = self._deduplicate(weighted)

        # Sort by score descending
        deduped.sort(key=lambda c: c.score, reverse=True)

        # Determine strategy label
        has_vector = any(c.source == "vector" for c in deduped)
        has_graph = any(c.source == "graph" for c in deduped)
        if has_vector and has_graph:
            strategy = "hybrid"
        elif has_graph:
            strategy = "graph"
        else:
            strategy = "vector"

        top_chunks = deduped[:top_k]
        return CombinedResult(
            chunks=top_chunks,
            vector_count=sum(1 for c in top_chunks if c.source == "vector"),
            graph_count=sum(1 for c in top_chunks if c.source == "graph"),
            strategy_used=strategy,
        )

    @staticmethod
    def _deduplicate(chunks: List[SourcedChunk]) -> List[SourcedChunk]:
        """Remove duplicate chunks based on content prefix."""
        seen: set[str] = set()
        deduped: List[SourcedChunk] = []
        for chunk in chunks:
            key = chunk.content[:200].strip().lower()
            if key not in seen:
                seen.add(key)
                deduped.append(chunk)
        return deduped
