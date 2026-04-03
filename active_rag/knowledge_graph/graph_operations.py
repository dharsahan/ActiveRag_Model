from typing import List, Dict, Any, Optional, Tuple
import logging
import re
from .neo4j_client import Neo4jClient
from .query_builder import CypherQueryBuilder
from ..nlp_pipeline.entity_extractor import EntityExtractor

class GraphOperations:
    """High-level graph operations for querying and reasoning"""

    def __init__(self, client: Neo4jClient):
        self.client = client
        self.query_builder = CypherQueryBuilder()
        self.entity_extractor = EntityExtractor()

    def find_related_entities(self, entity_id: str, relationship_types: Optional[List[str]] = None, depth: int = 1) -> List[Dict[str, Any]]:
        """Find entities related to a starting entity"""
        if not relationship_types:
            relationship_types = []  # Empty list means all relationship types

        query = self.query_builder.find_related_entities(entity_id, relationship_types, depth, limit=50)

        results = []
        with self.client._driver.session() as session:
            result = session.run(query, entity_id=entity_id, limit=50)

            for record in result:
                related_entity = dict(record["related"])
                related_entity["labels"] = record.get("entity_labels", [])

                if depth == 1:
                    related_entity["relationship_type"] = record["relationship_type"]
                else:
                    related_entity["path_length"] = record["path_length"]
                    related_entity["path_relationships"] = [rel["type"] for rel in record["path_relationships"]]

                results.append(related_entity)

        return results

    def find_paths(self, start_id: str, end_id: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """Find paths between two entities"""
        query = self.query_builder.find_paths(start_id, end_id, max_depth, limit=10)

        paths = []
        with self.client._driver.session() as session:
            result = session.run(query, start_id=start_id, end_id=end_id, limit=10)

            for record in result:
                path_info = {
                    "length": record["path_length"],
                    "relationship_types": record["relationship_types"],
                    "nodes": record["path_nodes"],
                    "reasoning_path": self._create_reasoning_path(record["path_nodes"], record["relationship_types"])
                }
                paths.append(path_info)

        return paths

    def multi_hop_query(self, query_text: str, max_hops: int = 2) -> Dict[str, Any]:
        """Execute multi-hop reasoning query using NLP to extract entities"""
        # Extract entities from query text
        from ..schemas.entities import ContentDomain
        entities = self.entity_extractor.extract_entities(query_text, ContentDomain.MIXED_WEB)

        if not entities:
            return {"entities": [], "paths": [], "reasoning": "No entities found in query"}

        # Start with first entity and explore outward
        start_entity_id = entities[0]["properties"]["id"]

        # Find entities in neighborhood
        neighborhood = self.get_entity_neighborhood(start_entity_id, radius=max_hops)

        # Filter relevant entities based on query context
        relevant_entities = self._filter_relevant_entities(neighborhood, query_text, entities[1:] if len(entities) > 1 else [])

        # Find paths between start entity and relevant entities
        paths = []
        for target_entity in relevant_entities[:5]:  # Limit to top 5 for performance
            entity_paths = self.find_paths(start_entity_id, target_entity["id"], max_depth=max_hops)
            for path in entity_paths:
                path["target_entity"] = target_entity
                paths.append(path)

        return {
            "entities": relevant_entities,
            "paths": paths,
            "reasoning": f"Found {len(relevant_entities)} relevant entities with {len(paths)} connection paths"
        }

    def get_entity_neighborhood(self, entity_id: str, radius: int = 2) -> List[Dict[str, Any]]:
        """Get all entities within specified radius of a starting entity"""
        query = self.query_builder.entity_neighborhood(entity_id, radius, limit=100)

        neighbors = []
        with self.client._driver.session() as session:
            result = session.run(query, entity_id=entity_id, limit=100)

            for record in result:
                neighbor = dict(record["neighbor"])
                neighbor["distance"] = record["distance"]
                neighbor["labels"] = record["entity_labels"]
                neighbors.append(neighbor)

        return neighbors

    def search_entities_by_name(self, name_pattern: str, entity_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search for entities matching a name pattern"""
        # Convert simple string to regex pattern
        if not name_pattern.startswith('.*'):
            name_pattern = f".*{re.escape(name_pattern)}.*"

        query = self.query_builder.entity_by_name_search(name_pattern, entity_types, limit=20)

        entities = []
        with self.client._driver.session() as session:
            result = session.run(query, name_pattern=name_pattern, limit=20)

            for record in result:
                entity = dict(record["e"])
                entity["labels"] = record["entity_labels"]
                entities.append(entity)

        return entities

    def get_document_entities(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all entities mentioned in a specific document"""
        query = self.query_builder.document_entity_mentions(doc_id)

        entities = []
        with self.client._driver.session() as session:
            result = session.run(query, doc_id=doc_id)

            for record in result:
                entity = dict(record["entity"])
                entity["labels"] = record["entity_labels"]
                entities.append(entity)

        return entities

    def explore_concept_relationships(self, concept_id: str, depth: int = 2) -> List[Dict[str, Any]]:
        """Explore relationships between concepts (builds-on, relates-to, etc.)"""
        query = self.query_builder.concept_relationships(concept_id, depth)

        related_concepts = []
        with self.client._driver.session() as session:
            result = session.run(query, concept_id=concept_id)

            for record in result:
                concept = dict(record["related"])
                concept["relationship_chain"] = record["relationship_chain"]
                concept["distance"] = record["distance"]
                related_concepts.append(concept)

        return related_concepts

    def _create_reasoning_path(self, nodes: List[Dict[str, Any]], relationships: List[str]) -> str:
        """Create human-readable reasoning path description"""
        if len(nodes) < 2:
            return ""

        path_parts = []
        for i in range(len(nodes) - 1):
            start_node = nodes[i]
            end_node = nodes[i + 1]
            relationship = relationships[i] if i < len(relationships) else "RELATED_TO"

            path_parts.append(f"{start_node.get('name', start_node.get('id'))} -{relationship}-> {end_node.get('name', end_node.get('id'))}")

        return " → ".join(path_parts)

    def _filter_relevant_entities(self, entities: List[Dict[str, Any]], query_text: str, target_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter entities based on relevance to query and target entities"""
        relevant = []
        query_lower = query_text.lower()

        # If we have target entities from the query, prioritize matches
        target_names = {entity["properties"]["name"].lower() for entity in target_entities}

        for entity in entities:
            entity_name = entity.get("name", "").lower()

            # High relevance: exact matches with target entities
            if entity_name in target_names:
                entity["relevance_score"] = 1.0
                relevant.append(entity)
            # Medium relevance: entity name appears in query
            elif any(word in entity_name for word in query_lower.split()):
                entity["relevance_score"] = 0.7
                relevant.append(entity)
            # Lower relevance: entity is close to start (low distance)
            elif entity.get("distance", 10) <= 2:
                entity["relevance_score"] = 0.3
                relevant.append(entity)

        # Sort by relevance score and distance
        relevant.sort(key=lambda x: (-x.get("relevance_score", 0), x.get("distance", 0)))
        return relevant[:20]  # Return top 20 most relevant

    def get_graph_stats(self) -> Dict[str, int]:
        """Get basic statistics about the knowledge graph"""
        stats_query = """
        MATCH (n)
        WITH labels(n) as node_labels
        UNWIND node_labels as label
        RETURN label, count(*) as count
        ORDER BY count DESC
        """

        stats = {"total_nodes": 0, "node_types": {}}

        with self.client._driver.session() as session:
            result = session.run(stats_query)
            for record in result:
                label = record["label"]
                count = record["count"]
                stats["node_types"][label] = count
                if label != "Entity":  # Avoid double counting base Entity label
                    stats["total_nodes"] += count

        # Get relationship stats
        rel_query = "MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count ORDER BY count DESC"

        stats["total_relationships"] = 0
        stats["relationship_types"] = {}

        with self.client._driver.session() as session:
            result = session.run(rel_query)
            for record in result:
                rel_type = record["rel_type"]
                count = record["count"]
                stats["relationship_types"][rel_type] = count
                stats["total_relationships"] += count

        return stats