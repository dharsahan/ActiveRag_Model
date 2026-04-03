from typing import List, Dict, Any, Optional
import re

class CypherQueryBuilder:
    """Helper class to build Cypher queries programmatically"""

    @staticmethod
    def find_related_entities(entity_id: str, relationship_types: List[str], depth: int = 1, limit: int = 50) -> str:
        """Build query to find entities related to a starting entity"""
        if depth == 1:
            # Single hop query
            if relationship_types:
                rel_pattern = "|".join(relationship_types)
                rel_filter = f"r:{rel_pattern}"
            else:
                rel_filter = "r"

            query = f"""
            MATCH (start {{id: $entity_id}})
            MATCH (start)-[{rel_filter}]-(related)
            RETURN related, r, type(r) as relationship_type
            LIMIT $limit
            """
        else:
            # Multi-hop query with variable length path
            if relationship_types:
                rel_pattern = "|".join(relationship_types)
                rel_filter = f"r:{rel_pattern}"
            else:
                rel_filter = "r"

            query = f"""
            MATCH (start {{id: $entity_id}})
            MATCH path = (start)-[{rel_filter}*1..{depth}]-(related)
            RETURN related, relationships(path) as path_relationships, length(path) as path_length
            LIMIT $limit
            """

        return query.strip()

    @staticmethod
    def find_paths(start_id: str, end_id: str, max_depth: int = 3, limit: int = 10) -> str:
        """Build query to find paths between two entities"""
        query = f"""
        MATCH (start {{id: $start_id}})
        MATCH (end {{id: $end_id}})
        MATCH path = shortestPath((start)-[*1..{max_depth}]-(end))
        RETURN path, length(path) as path_length,
               [rel in relationships(path) | type(rel)] as relationship_types,
               [node in nodes(path) | {{id: node.id, labels: labels(node), name: node.name}}] as path_nodes
        ORDER BY path_length
        LIMIT $limit
        """

        return query.strip()

    @staticmethod
    def entity_neighborhood(entity_id: str, radius: int = 2, limit: int = 100) -> str:
        """Build query to get entity's neighborhood (all connected entities within radius)"""
        query = f"""
        MATCH (center {{id: $entity_id}})
        MATCH path = (center)-[*1..{radius}]-(neighbor)
        WITH neighbor, min(length(path)) as distance
        RETURN neighbor, distance, labels(neighbor) as entity_labels
        ORDER BY distance, neighbor.name
        LIMIT $limit
        """

        return query.strip()

    @staticmethod
    def entity_by_name_search(name_pattern: str, entity_labels: Optional[List[str]] = None, limit: int = 20) -> str:
        """Build query to search entities by name pattern"""
        label_filter = ":".join(entity_labels) if entity_labels else ""

        query = f"""
        MATCH (e{':' + label_filter if label_filter else ''})
        WHERE e.name =~ $name_pattern
        RETURN e, labels(e) as entity_labels
        ORDER BY e.name
        LIMIT $limit
        """

        return query.strip()

    @staticmethod
    def document_entity_mentions(doc_id: str) -> str:
        """Build query to get all entities mentioned in a document"""
        query = """
        MATCH (doc:Document {id: $doc_id})-[:MENTIONS]->(entity)
        RETURN entity, labels(entity) as entity_labels
        ORDER BY entity.name
        """

        return query.strip()

    @staticmethod
    def concept_relationships(concept_id: str, depth: int = 2) -> str:
        """Build query to explore concept relationships (builds-on, relates-to, etc.)"""
        query = f"""
        MATCH (concept:Concept {{id: $concept_id}})
        MATCH path = (concept)-[r*1..{depth}]-(related:Concept)
        RETURN related,
               [rel in relationships(path) | type(rel)] as relationship_chain,
               length(path) as distance
        ORDER BY distance, related.name
        """

        return query.strip()