"""Community detection for entity clustering.

Groups related entities into communities using in-memory label propagation.
Works with graph data exported from Neo4j.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class Community:
    """A detected community of related entities."""
    community_id: int
    entities: List[Dict[str, Any]]
    dominant_label: str  # Most common entity type in this community
    size: int

    @property
    def entity_names(self) -> List[str]:
        return [e.get("name", "Unknown") for e in self.entities]


class CommunityDetector:
    """Detects communities of related entities using label propagation."""

    def detect_communities(
        self,
        graph_ops,
        entity_type: Optional[str] = None,
        max_entities: int = 200,
    ) -> List[Community]:
        """Detect entity communities from the graph.

        Uses a simple label propagation approach on entity neighborhoods:
        1. Query entities and their connections from Neo4j
        2. Build an adjacency list in memory
        3. Run label propagation to assign community IDs
        4. Group entities by community

        Args:
            graph_ops: GraphOperations instance
            entity_type: Optional filter by entity type (e.g., "Person")
            max_entities: Maximum entities to process

        Returns:
            List of Community objects sorted by size
        """
        # 1. Get entities and their connections
        adjacency, entities_map = self._build_adjacency(
            graph_ops, entity_type, max_entities
        )

        if not adjacency:
            return []

        # 2. Run label propagation
        labels = self._label_propagation(adjacency, max_iterations=10)

        # 3. Group by community label
        communities_map: Dict[int, List[str]] = defaultdict(list)
        for entity_id, label in labels.items():
            communities_map[label].append(entity_id)

        # 4. Build Community objects
        communities = []
        for cid, member_ids in communities_map.items():
            members = [entities_map[eid] for eid in member_ids if eid in entities_map]
            if not members:
                continue

            # Find dominant label type
            type_counts: Dict[str, int] = defaultdict(int)
            for m in members:
                for lbl in m.get("labels", []):
                    type_counts[lbl] += 1
            dominant = max(type_counts, key=type_counts.get) if type_counts else "Unknown"

            communities.append(Community(
                community_id=cid,
                entities=members,
                dominant_label=dominant,
                size=len(members),
            ))

        # Sort by size descending
        communities.sort(key=lambda c: c.size, reverse=True)
        return communities

    def _build_adjacency(
        self,
        graph_ops,
        entity_type: Optional[str],
        max_entities: int,
    ) -> tuple:
        """Build adjacency list from graph data.

        Returns:
            (adjacency dict, entities_map dict)
        """
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        entities_map: Dict[str, Dict[str, Any]] = {}

        try:
            # Search for entities
            entity_types = [entity_type] if entity_type else None
            entities = graph_ops.search_entities_by_name("", entity_types)
            entities = entities[:max_entities]

            for entity in entities:
                eid = entity.get("id", "")
                if not eid:
                    continue
                entities_map[eid] = entity
                adjacency.setdefault(eid, set())

                # Get direct neighbors
                try:
                    related = graph_ops.find_related_entities(eid, [], depth=1)
                    for rel in related:
                        rid = rel.get("id", "")
                        if rid:
                            adjacency[eid].add(rid)
                            adjacency.setdefault(rid, set()).add(eid)
                            if rid not in entities_map:
                                entities_map[rid] = rel
                except Exception:
                    pass

        except Exception as e:
            logger.warning(f"Failed to build adjacency: {e}")

        return adjacency, entities_map

    @staticmethod
    def _label_propagation(
        adjacency: Dict[str, Set[str]],
        max_iterations: int = 10,
    ) -> Dict[str, int]:
        """Simple label propagation algorithm.

        Each node starts with a unique label, then iteratively adopts
        the most common label among its neighbors.

        Args:
            adjacency: Adjacency list mapping node_id → set of neighbor_ids
            max_iterations: Maximum propagation rounds

        Returns:
            Dict mapping node_id → community_label (int)
        """
        # Initialize: each node gets its own unique label
        node_ids = list(adjacency.keys())
        labels = {nid: i for i, nid in enumerate(node_ids)}

        for _ in range(max_iterations):
            changed = False
            for nid in node_ids:
                neighbors = adjacency[nid]
                if not neighbors:
                    continue

                # Count neighbor labels
                label_counts: Dict[int, int] = defaultdict(int)
                for neighbor in neighbors:
                    if neighbor in labels:
                        label_counts[labels[neighbor]] += 1

                if not label_counts:
                    continue

                # Adopt the most common neighbor label
                best_label = max(label_counts, key=label_counts.get)
                if labels[nid] != best_label:
                    labels[nid] = best_label
                    changed = True

            if not changed:
                break

        return labels
