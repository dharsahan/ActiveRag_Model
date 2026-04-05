"""Enhanced multi-hop reasoning engine with subgraph extraction and path ranking.

Builds on GraphOperations to provide deep graph reasoning:
1. Extract seed entities from the query
2. Expand entity neighborhoods in the graph
3. Extract relevant subgraph (nodes + edges)
4. Score and rank reasoning paths by relevance
5. Return structured ReasoningResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ReasoningPath:
    """A single reasoning path through the knowledge graph."""
    nodes: List[Dict[str, Any]]
    relationships: List[str]
    reasoning_text: str  # Human-readable: "Einstein → AFFILIATED_WITH → Princeton"
    score: float         # 0.0–1.0 relevance score
    length: int

    @property
    def start_entity(self) -> str:
        return self.nodes[0].get("name", "Unknown") if self.nodes else "Unknown"

    @property
    def end_entity(self) -> str:
        return self.nodes[-1].get("name", "Unknown") if self.nodes else "Unknown"


@dataclass
class Subgraph:
    """Extracted subgraph around seed entities."""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    seed_entity_ids: List[str]

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)


@dataclass
class ReasoningResult:
    """Full result from the reasoning engine."""
    query: str
    seed_entities: List[Dict[str, Any]]
    subgraph: Subgraph
    ranked_paths: List[ReasoningPath]
    supporting_entities: List[Dict[str, Any]]
    reasoning_summary: str
    confidence: float  # Overall reasoning confidence 0.0–1.0

    @property
    def has_results(self) -> bool:
        return len(self.ranked_paths) > 0 or len(self.supporting_entities) > 0


class PathRanker:
    """Ranks reasoning paths by relevance to the query."""

    def __init__(
        self,
        length_penalty: float = 0.15,
        entity_match_bonus: float = 0.3,
        relationship_type_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self._length_penalty = length_penalty
        self._entity_match_bonus = entity_match_bonus
        self._rel_weights = relationship_type_weights or {}

    def rank_paths(
        self,
        paths: List[Dict[str, Any]],
        query_entities: List[str],
    ) -> List[ReasoningPath]:
        """Score and rank paths by relevance.

        Scoring:
        - Base score: 1.0
        - Length penalty: -0.15 per hop (shorter = better)
        - Entity match bonus: +0.3 if path touches a query entity
        - Relationship weight: configurable per relationship type
        """
        scored: List[ReasoningPath] = []

        for path_data in paths:
            nodes = path_data.get("nodes", [])
            relationships = path_data.get("relationship_types", [])
            length = path_data.get("length", len(nodes) - 1)
            reasoning = path_data.get("reasoning_path", "")

            # Base score, penalize long paths
            score = max(0.1, 1.0 - self._length_penalty * length)

            # Bonus for paths that touch query entities
            path_names = {n.get("name", "").lower() for n in nodes}
            for qe in query_entities:
                if qe.lower() in path_names:
                    score += self._entity_match_bonus

            # Relationship type weighting
            for rel in relationships:
                score += self._rel_weights.get(rel, 0.0)

            score = min(1.0, max(0.0, score))

            # Build reasoning text if not provided
            if not reasoning and len(nodes) >= 2:
                parts = []
                for i in range(len(nodes) - 1):
                    start = nodes[i].get("name", nodes[i].get("id", "?"))
                    rel = relationships[i] if i < len(relationships) else "RELATED_TO"
                    end = nodes[i + 1].get("name", nodes[i + 1].get("id", "?"))
                    parts.append(f"{start} —[{rel}]→ {end}")
                reasoning = " → ".join(parts)

            scored.append(ReasoningPath(
                nodes=nodes,
                relationships=relationships,
                reasoning_text=reasoning,
                score=score,
                length=length,
            ))

        # Sort by score descending
        scored.sort(key=lambda p: p.score, reverse=True)
        return scored


class SubgraphExtractor:
    """Extracts relevant subgraphs from the knowledge graph around seed entities."""

    def extract(
        self,
        graph_ops,
        seed_entity_ids: List[str],
        max_radius: int = 2,
    ) -> Subgraph:
        """Pull a connected subgraph around seed entities.

        Args:
            graph_ops: GraphOperations instance
            seed_entity_ids: Starting entity IDs
            max_radius: Maximum hop radius

        Returns:
            Subgraph with nodes and edges
        """
        all_nodes: Dict[str, Dict[str, Any]] = {}
        all_edges: List[Dict[str, Any]] = []

        for entity_id in seed_entity_ids:
            try:
                neighbors = graph_ops.get_entity_neighborhood(entity_id, radius=max_radius)
                for neighbor in neighbors:
                    nid = neighbor.get("id", "")
                    if nid and nid not in all_nodes:
                        all_nodes[nid] = neighbor

                # Get direct relationships for edges
                related = graph_ops.find_related_entities(entity_id, [], depth=1)
                for rel in related:
                    rel_id = rel.get("id", "")
                    edge = {
                        "from": entity_id,
                        "to": rel_id,
                        "type": rel.get("relationship_type", "RELATED_TO"),
                    }
                    all_edges.append(edge)
                    if rel_id and rel_id not in all_nodes:
                        all_nodes[rel_id] = rel

            except Exception as e:
                logger.warning(f"Failed to expand entity {entity_id}: {e}")

        return Subgraph(
            nodes=list(all_nodes.values()),
            edges=all_edges,
            seed_entity_ids=seed_entity_ids,
        )


class ReasoningEngine:
    """Multi-hop reasoning engine combining subgraph extraction and path ranking."""

    def __init__(self, graph_ops=None, entity_extractor=None) -> None:
        self._graph_ops = graph_ops
        self._entity_extractor = entity_extractor
        self._subgraph_extractor = SubgraphExtractor()
        self._path_ranker = PathRanker()

    def reason(self, query: str, max_hops: int = 2) -> ReasoningResult:
        """Execute full multi-hop reasoning pipeline.

        Args:
            query: Natural language question
            max_hops: Maximum traversal depth

        Returns:
            ReasoningResult with ranked paths, subgraph, and confidence
        """
        # 1. Extract seed entities from the query
        seed_entities = self._extract_seed_entities(query)

        if not seed_entities:
            return self._empty_result(query, "No entities detected in query")

        if not self._graph_ops:
            return self._empty_result(query, "Graph database not available")

        seed_ids = [e["properties"]["id"] for e in seed_entities]
        seed_names = [e["properties"]["name"] for e in seed_entities]

        # 2. Extract subgraph around seed entities
        subgraph = self._subgraph_extractor.extract(
            self._graph_ops, seed_ids, max_radius=max_hops
        )

        # 3. Find paths between seed entities (if multiple)
        raw_paths: List[Dict[str, Any]] = []
        if len(seed_ids) >= 2:
            for i in range(len(seed_ids)):
                for j in range(i + 1, len(seed_ids)):
                    try:
                        paths = self._graph_ops.find_paths(
                            seed_ids[i], seed_ids[j], max_depth=max_hops
                        )
                        raw_paths.extend(paths)
                    except Exception as e:
                        logger.warning(f"Path finding failed: {e}")

        # Also get multi-hop results for general context
        if len(seed_ids) >= 1:
            try:
                mh_result = self._graph_ops.multi_hop_query(query, max_hops=max_hops)
                # Convert multi-hop paths to our format
                for path in mh_result.get("paths", []):
                    raw_paths.append(path)
            except Exception as e:
                logger.warning(f"Multi-hop query failed: {e}")

        # 4. Rank paths
        ranked_paths = self._path_ranker.rank_paths(raw_paths, seed_names)

        # 5. Collect supporting entities (from subgraph, not in seed)
        supporting = [
            n for n in subgraph.nodes
            if n.get("id", "") not in set(seed_ids)
        ]
        # Sort by distance if available
        supporting.sort(key=lambda n: n.get("distance", 99))
        supporting = supporting[:10]

        # 6. Compute overall confidence
        confidence = self._compute_confidence(ranked_paths, subgraph, seed_entities)

        # 7. Build summary
        summary = self._build_summary(seed_names, ranked_paths, supporting)

        return ReasoningResult(
            query=query,
            seed_entities=seed_entities,
            subgraph=subgraph,
            ranked_paths=ranked_paths[:10],
            supporting_entities=supporting,
            reasoning_summary=summary,
            confidence=confidence,
        )

    def _extract_seed_entities(self, query: str) -> List[Dict[str, Any]]:
        """Extract entities from the query text using the NLP pipeline."""
        if not self._entity_extractor:
            return []

        try:
            from active_rag.schemas.entities import ContentDomain
            return self._entity_extractor.extract_entities(query, ContentDomain.MIXED_WEB)
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []

    def _compute_confidence(
        self,
        paths: List[ReasoningPath],
        subgraph: Subgraph,
        seed_entities: List[Dict[str, Any]],
    ) -> float:
        """Compute overall reasoning confidence.

        Factors:
        - Path quality: average path score
        - Subgraph density: more connections = more confident
        - Entity coverage: seed entities found in graph = more confident
        """
        if not paths and subgraph.node_count == 0:
            return 0.1

        # Path quality (avg of top 3 paths)
        top_scores = [p.score for p in paths[:3]]
        path_quality = sum(top_scores) / len(top_scores) if top_scores else 0.0

        # Subgraph density
        density = min(1.0, subgraph.edge_count / max(subgraph.node_count, 1) / 2.0)

        # Entity coverage
        seed_count = len(seed_entities)
        found_in_graph = sum(
            1 for n in subgraph.nodes
            if n.get("id", "") in {e["properties"]["id"] for e in seed_entities}
        )
        coverage = found_in_graph / max(seed_count, 1)

        # Weighted combination
        confidence = 0.5 * path_quality + 0.3 * coverage + 0.2 * density
        return min(1.0, max(0.0, confidence))

    def _build_summary(
        self,
        seed_names: List[str],
        paths: List[ReasoningPath],
        supporting: List[Dict[str, Any]],
    ) -> str:
        """Build a human-readable reasoning summary."""
        parts = []

        if seed_names:
            parts.append(f"Identified entities: {', '.join(seed_names)}")

        if paths:
            parts.append(f"Found {len(paths)} reasoning path(s)")
            if paths[0].reasoning_text:
                parts.append(f"Top path: {paths[0].reasoning_text}")
        else:
            parts.append("No direct reasoning paths found")

        if supporting:
            names = [s.get("name", "?") for s in supporting[:5]]
            parts.append(f"Related entities: {', '.join(names)}")

        return ". ".join(parts)

    @staticmethod
    def _empty_result(query: str, reason: str) -> ReasoningResult:
        return ReasoningResult(
            query=query,
            seed_entities=[],
            subgraph=Subgraph(nodes=[], edges=[], seed_entity_ids=[]),
            ranked_paths=[],
            supporting_entities=[],
            reasoning_summary=reason,
            confidence=0.0,
        )
