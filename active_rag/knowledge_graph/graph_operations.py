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
                # Inline reasoning path creation logic
                nodes = record["path_nodes"]
                relationships = record["relationship_types"]

                reasoning_path = ""
                if len(nodes) >= 2:
                    path_parts = []
                    for i in range(len(nodes) - 1):
                        start_node = nodes[i]
                        end_node = nodes[i + 1]
                        relationship = relationships[i] if i < len(relationships) else "RELATED_TO"

                        path_parts.append(f"{start_node.get('name', start_node.get('id'))} -{relationship}-> {end_node.get('name', end_node.get('id'))}")

                    reasoning_path = " → ".join(path_parts)

                path_info = {
                    "length": record["path_length"],
                    "relationship_types": record["relationship_types"],
                    "nodes": record["path_nodes"],
                    "reasoning_path": reasoning_path
                }
                paths.append(path_info)

        return paths

    def multi_hop_query(self, query_text: str, max_hops: int = 2) -> Dict[str, Any]:
        """Execute multi-hop reasoning query using NLP to extract entities"""
        # Extract entities from query text
        from ..schemas.entities import ContentDomain
        query_entities = self.entity_extractor.extract_entities(query_text, ContentDomain.MIXED_WEB)

        if not query_entities:
            return {"entities": [], "paths": [], "reasoning": "No entities found in query"}

        all_neighbors = []
        all_paths = []
        start_entity_ids = []

        # Try to find all query entities in the database
        with self.client._driver.session() as session:
            for q_ent in query_entities:
                entity_name = q_ent["properties"]["name"]
                entity_label = q_ent["label"]

                search_query = f"""
                MATCH (n:{entity_label})
                WHERE n.name = $entity_name
                RETURN n.id as entity_id
                ORDER BY n.id
                LIMIT 1
                """
                result = session.run(search_query, entity_name=entity_name)
                record = result.single()
                
                if record:
                    start_entity_ids.append(record["entity_id"])

        if not start_entity_ids:
            return {"entities": [], "paths": [], "reasoning": f"None of the query entities were found in database"}

        # Combine neighborhoods from all start entities
        for start_id in start_entity_ids:
            neighbors = self.get_entity_neighborhood(start_id, radius=max_hops)
            all_neighbors.extend(neighbors)

        # Deduplicate and score neighbors
        unique_neighbors = {}
        query_lower = query_text.lower()
        target_names = {entity["properties"]["name"].lower() for entity in query_entities}

        for entity in all_neighbors:
            eid = entity["id"]
            if eid in start_entity_ids:
                continue # Skip the start entities themselves
                
            entity_name = entity.get("name", "").lower()
            
            # Basic score based on distance
            relevance = 0.3 if entity.get("distance", 10) <= 2 else 0.1
            
            # Boost if name matches query
            if entity_name in target_names:
                relevance = 1.0
            elif any(word in entity_name for word in query_lower.split() if len(word) > 2):
                relevance = 0.7
                
            entity["relevance_score"] = relevance
            
            if eid not in unique_neighbors or relevance > unique_neighbors[eid].get("relevance_score", 0):
                unique_neighbors[eid] = entity

        relevant_entities = list(unique_neighbors.values())
        relevant_entities.sort(key=lambda x: (-x.get("relevance_score", 0), x.get("distance", 0)))
        relevant_entities = relevant_entities[:20]

        # Find paths between start entities and relevant entities
        for start_id in start_entity_ids:
            for target_entity in relevant_entities[:5]:
                if start_id == target_entity["id"]:
                    continue
                entity_paths = self.find_paths(start_id, target_entity["id"], max_depth=max_hops)
                for path in entity_paths:
                    path["target_entity"] = target_entity
                    all_paths.append(path)

        return {
            "entities": relevant_entities,
            "paths": all_paths,
            "reasoning": f"Searched from {len(start_entity_ids)} entities. Found {len(relevant_entities)} relevant neighbors with {len(all_paths)} paths."
        }

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the graph"""
        stats_query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->()
        RETURN
            count(DISTINCT n) as total_nodes,
            count(DISTINCT r) as total_relationships,
            collect(DISTINCT labels(n)) as node_labels,
            collect(DISTINCT type(r)) as relationship_types
        """

        with self.client._driver.session() as session:
            result = session.run(stats_query)
            record = result.single()

            if record:
                # Flatten node labels (they come as lists of lists)
                node_labels = []
                for label_list in record["node_labels"]:
                    if label_list:  # Skip empty label lists
                        node_labels.extend(label_list)

                # Remove None values from relationship types
                relationship_types = [rt for rt in record["relationship_types"] if rt is not None]

                return {
                    "total_nodes": record["total_nodes"] or 0,
                    "total_relationships": record["total_relationships"] or 0,
                    "node_types": list(set(node_labels)),  # Remove duplicates
                    "relationship_types": list(set(relationship_types))  # Remove duplicates
                }
            else:
                return {
                    "total_nodes": 0,
                    "total_relationships": 0,
                    "node_types": [],
                    "relationship_types": []
                }

    def search_entities_by_name(self, name_pattern: str, entity_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search for entities by name pattern"""
        # Validate entity types to prevent injection
        if entity_types:
            for entity_type in entity_types:
                if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', entity_type):
                    raise ValueError(f"Invalid entity type: '{entity_type}'")

        # Build the query
        if entity_types:
            label_filter = ":" + ":".join(entity_types)
        else:
            label_filter = ""

        query = f"""
        MATCH (n{label_filter})
        WHERE toLower(n.name) CONTAINS toLower($name_pattern)
        RETURN n, labels(n) as entity_labels
        ORDER BY n.name
        LIMIT 50
        """

        entities = []
        with self.client._driver.session() as session:
            result = session.run(query, name_pattern=name_pattern)

            for record in result:
                entity = dict(record["n"])
                entity["labels"] = record["entity_labels"]
                entities.append(entity)

        return entities

    def get_entity_neighborhood(self, entity_id: str, radius: int = 2) -> List[Dict[str, Any]]:
        """Get the neighborhood of entities around a given entity"""
        # Validate radius to prevent injection
        if not isinstance(radius, int) or radius < 1 or radius > 10:
            raise ValueError(f"Invalid radius: {radius}. Must be integer between 1 and 10")

        radius_range = "1.." + str(radius)
        neighborhood_query = """
        MATCH (center {id: $entity_id})
        MATCH path = (center)-[*""" + radius_range + """]-(neighbor)
        WITH neighbor, min(length(path)) as distance
        RETURN neighbor, distance, labels(neighbor) as entity_labels
        ORDER BY distance, neighbor.name
        LIMIT $limit
        """

        neighbors = []
        with self.client._driver.session() as session:
            result = session.run(neighborhood_query, entity_id=entity_id, limit=100)

            for record in result:
                neighbor = dict(record["neighbor"])
                neighbor["distance"] = record["distance"]
                neighbor["labels"] = record["entity_labels"]
                neighbors.append(neighbor)

        return neighbors