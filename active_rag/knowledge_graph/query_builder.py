from typing import List, Dict, Any, Optional
import re

class CypherQueryBuilder:
    """Helper class to build Cypher queries programmatically"""

    @staticmethod
    def find_related_entities(entity_id: str, relationship_types: List[str], depth: int = 1, limit: int = 50) -> str:
        """Build query to find entities related to a starting entity"""
        # Validate depth parameter to prevent injection
        if not isinstance(depth, int) or depth < 1 or depth > 10:
            raise ValueError(f"Invalid depth: {depth}. Must be integer between 1 and 10")

        # Validate relationship types to prevent injection
        safe_rels = []
        valid_rel_pattern = re.compile(r'^[A-Z_]+$')  # Only allow uppercase letters and underscores

        for rel in relationship_types:
            if valid_rel_pattern.match(rel):
                safe_rels.append(rel)
            else:
                raise ValueError(f"Invalid relationship type: {rel}")

        if depth == 1:
            # Single hop query
            if safe_rels:
                rel_pattern = "|".join(safe_rels)
                query = """
                MATCH (start {id: $entity_id})
                MATCH (start)-[r:""" + rel_pattern + """]-(related)
                RETURN related, r, type(r) as relationship_type
                LIMIT $limit
                """
            else:
                query = """
                MATCH (start {id: $entity_id})
                MATCH (start)-[r]-(related)
                RETURN related, r, type(r) as relationship_type
                LIMIT $limit
                """
        else:
            # Multi-hop query with variable length path - construct safely
            depth_range = "1.." + str(depth)
            if safe_rels:
                rel_pattern = "|".join(safe_rels)
                query = """
                MATCH (start {id: $entity_id})
                MATCH path = (start)-[r:""" + rel_pattern + """*""" + depth_range + """]-(related)
                RETURN related, relationships(path) as path_relationships, length(path) as path_length
                LIMIT $limit
                """
            else:
                query = """
                MATCH (start {id: $entity_id})
                MATCH path = (start)-[r*""" + depth_range + """]-(related)
                RETURN related, relationships(path) as path_relationships, length(path) as path_length
                LIMIT $limit
                """

        return query.strip()

    @staticmethod
    def find_paths(start_id: str, end_id: str, max_depth: int = 3, limit: int = 10) -> str:
        """Build query to find paths between two entities"""
        # Validate max_depth parameter to prevent injection
        if not isinstance(max_depth, int) or max_depth < 1 or max_depth > 10:
            raise ValueError(f"Invalid max_depth: {max_depth}. Must be integer between 1 and 10")

        depth_range = "1.." + str(max_depth)
        query = """
        MATCH (start {id: $start_id})
        MATCH (end {id: $end_id})
        MATCH path = shortestPath((start)-[*""" + depth_range + """]-(end))
        RETURN path, length(path) as path_length,
               [rel in relationships(path) | type(rel)] as relationship_types,
               [node in nodes(path) | {id: node.id, labels: labels(node), name: node.name}] as path_nodes
        ORDER BY path_length
        LIMIT $limit
        """

        return query.strip()