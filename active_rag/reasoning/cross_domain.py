"""Cross-domain relationship discovery.

Discovers connections between entities across different content domains
(Research ↔ Technical ↔ Business ↔ Mixed Web).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from active_rag.schemas.entities import ContentDomain

logger = logging.getLogger(__name__)

# Domain labels mapped to their primary entity types
_DOMAIN_ENTITY_TYPES: Dict[str, List[str]] = {
    ContentDomain.RESEARCH.value: ["Person", "Concept", "Organization"],
    ContentDomain.TECHNICAL.value: ["Component"],
    ContentDomain.BUSINESS.value: ["Person", "Organization", "Process"],
    ContentDomain.MIXED_WEB.value: ["Person", "Organization"],
}


@dataclass
class CrossDomainLink:
    """A discovered link between entities in different domains."""
    source_entity: Dict[str, Any]
    source_domain: str
    target_entity: Dict[str, Any]
    target_domain: str
    relationship_type: str
    path_length: int


class CrossDomainDiscovery:
    """Discovers relationships between entities across content domains."""

    def find_cross_domain_links(
        self,
        graph_ops,
        entity_id: str,
        source_domain: Optional[str] = None,
        max_depth: int = 2,
    ) -> List[CrossDomainLink]:
        """Find entities in other domains connected to the given entity.

        Args:
            graph_ops: GraphOperations instance
            entity_id: Starting entity ID
            source_domain: Domain of the starting entity (auto-detected if None)
            max_depth: Maximum traversal depth

        Returns:
            List of CrossDomainLink objects
        """
        links: List[CrossDomainLink] = []

        try:
            # Get neighborhood of the entity
            neighbors = graph_ops.get_entity_neighborhood(entity_id, radius=max_depth)

            # Detect source entity domain
            if not source_domain:
                source_domain = self._detect_domain_from_labels(
                    self._get_entity_labels(graph_ops, entity_id)
                )

            source_entity = {"id": entity_id}

            for neighbor in neighbors:
                neighbor_labels = neighbor.get("labels", [])
                neighbor_domain = self._detect_domain_from_labels(neighbor_labels)

                # Only include cross-domain links
                if neighbor_domain and neighbor_domain != source_domain:
                    links.append(CrossDomainLink(
                        source_entity=source_entity,
                        source_domain=source_domain,
                        target_entity=neighbor,
                        target_domain=neighbor_domain,
                        relationship_type="CROSS_DOMAIN",
                        path_length=neighbor.get("distance", 1),
                    ))

        except Exception as e:
            logger.warning(f"Cross-domain discovery failed for {entity_id}: {e}")

        # Sort by path length (closer = more relevant)
        links.sort(key=lambda l: l.path_length)
        return links

    def discover_bridges(
        self,
        graph_ops,
        max_entities: int = 50,
    ) -> List[Dict[str, Any]]:
        """Find entities that bridge multiple domains.

        A "bridge" entity appears in connections spanning multiple domains.

        Args:
            graph_ops: GraphOperations instance
            max_entities: Maximum entities to scan

        Returns:
            List of bridge entities with their domain connections
        """
        bridges: List[Dict[str, Any]] = []

        try:
            # Get a sample of entities
            entities = graph_ops.search_entities_by_name("")
            entities = entities[:max_entities]

            for entity in entities:
                eid = entity.get("id", "")
                if not eid:
                    continue

                entity_domain = self._detect_domain_from_labels(entity.get("labels", []))

                # Check neighbors for different domains
                try:
                    related = graph_ops.find_related_entities(eid, [], depth=1)
                    connected_domains = set()

                    for rel in related:
                        rel_domain = self._detect_domain_from_labels(rel.get("labels", []))
                        if rel_domain:
                            connected_domains.add(rel_domain)

                    # Bridge = connects to at least 2 different domains
                    if entity_domain:
                        connected_domains.add(entity_domain)

                    if len(connected_domains) >= 2:
                        bridges.append({
                            "entity": entity,
                            "entity_domain": entity_domain,
                            "connected_domains": list(connected_domains),
                            "bridge_strength": len(connected_domains),
                        })

                except Exception:
                    pass

        except Exception as e:
            logger.warning(f"Bridge discovery failed: {e}")

        # Sort by bridge strength (more domains = stronger bridge)
        bridges.sort(key=lambda b: b["bridge_strength"], reverse=True)
        return bridges

    def _get_entity_labels(self, graph_ops, entity_id: str) -> List[str]:
        """Get labels for a specific entity."""
        try:
            results = graph_ops.search_entities_by_name("")
            for r in results:
                if r.get("id") == entity_id:
                    return r.get("labels", [])
        except Exception:
            pass
        return []

    @staticmethod
    def _detect_domain_from_labels(labels: List[str]) -> str:
        """Detect content domain from entity labels."""
        label_set = set(labels)

        if "Concept" in label_set or "Method" in label_set:
            return ContentDomain.RESEARCH.value
        if "Component" in label_set:
            return ContentDomain.TECHNICAL.value
        if "Process" in label_set:
            return ContentDomain.BUSINESS.value
        if "Person" in label_set or "Organization" in label_set:
            return ContentDomain.MIXED_WEB.value  # General, could be any

        return ""
