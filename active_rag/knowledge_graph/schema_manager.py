"""
Schema management and validation for the hybrid vector-graph RAG system.

This module provides comprehensive schema validation and constraint management
for entities and relationships across all content domains.
"""

from typing import Dict, Any, List, Optional
import logging
import re
from ..schemas.entities import (
    ENTITY_SCHEMAS,
    EntitySchema,
    ContentDomain,
    get_entity_schema,
    get_entities_by_domain,
    list_entity_types
)
from ..schemas.relationships import (
    RELATIONSHIP_SCHEMAS,
    RelationshipSchema,
    get_relationship_schema,
    get_relationships_by_domain,
    list_relationship_types,
    get_valid_relationships_for_entities
)
from .neo4j_client import Neo4jClient


class SchemaManager:
    """
    Manages graph schema validation and Neo4j constraint creation.

    Provides comprehensive validation for entities and relationships
    against predefined schemas across all content domains.
    """

    def __init__(self, client: Neo4jClient):
        """
        Initialize schema manager with Neo4j client.

        Args:
            client: Neo4j client instance for database operations
        """
        self.client = client
        self.logger = logging.getLogger(__name__)

    def create_base_constraints(self) -> bool:
        """
        Create unique constraints and indexes for entity IDs.

        Creates constraints to ensure data integrity and improve query performance.

        Returns:
            bool: True if constraints created successfully, False otherwise
        """
        constraints = [
            "CREATE CONSTRAINT entity_id FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT person_id FOR (p:Person) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT org_id FOR (o:Organization) REQUIRE o.id IS UNIQUE",
            "CREATE CONSTRAINT concept_id FOR (c:Concept) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT component_id FOR (c:Component) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT process_id FOR (p:Process) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT document_id FOR (d:Document) REQUIRE d.id IS UNIQUE"
        ]

        try:
            with self.client._driver.session() as session:
                for query in constraints:
                    try:
                        session.run(query)
                        self.logger.info(f"Created constraint: {query}")
                    except Exception as e:
                        # Constraint may already exist
                        if "already exists" not in str(e) and "An equivalent" not in str(e):
                            self.logger.warning(f"Failed to create constraint: {e}")

            self.logger.info("Successfully created/verified all constraints")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create constraints: {e}")
            return False

    def validate_entity(self, label: str, properties: Dict[str, Any]) -> bool:
        """
        Validate entity properties against schema.

        Args:
            label: Entity label/type
            properties: Dictionary of entity properties

        Returns:
            bool: True if entity is valid, False otherwise
        """
        if label not in ENTITY_SCHEMAS:
            self.logger.error(f"Unknown entity type: {label}")
            return False

        schema = ENTITY_SCHEMAS[label]

        # Check required properties
        for prop in schema.required_properties:
            if prop not in properties or properties[prop] is None:
                self.logger.error(f"Missing required property '{prop}' for entity '{label}'")
                return False

        # Validate property values (basic type checking)
        if not self._validate_property_values(properties):
            return False

        # Enhanced property format validation
        if not self._validate_property_formats(label, properties):
            return False

        self.logger.debug(f"Entity '{label}' validation passed")
        return True

    def validate_relationship(
        self,
        rel_type: str,
        from_label: str,
        to_label: str,
        properties: Dict[str, Any] = None
    ) -> bool:
        """
        Validate relationship against schema.

        Args:
            rel_type: Relationship type
            from_label: Source entity label
            to_label: Target entity label
            properties: Optional relationship properties

        Returns:
            bool: True if relationship is valid, False otherwise
        """
        if rel_type not in RELATIONSHIP_SCHEMAS:
            self.logger.error(f"Unknown relationship type: {rel_type}")
            return False

        schema = RELATIONSHIP_SCHEMAS[rel_type]

        # Check valid label combinations
        if from_label not in schema.from_labels:
            self.logger.error(f"Invalid from_label '{from_label}' for relationship '{rel_type}'")
            return False

        if to_label not in schema.to_labels:
            self.logger.error(f"Invalid to_label '{to_label}' for relationship '{rel_type}'")
            return False

        if properties:
            # Check required properties
            for prop in schema.required_properties:
                if prop not in properties or properties[prop] is None:
                    self.logger.error(f"Missing required property '{prop}' for relationship '{rel_type}'")
                    return False

            # Validate property values
            if not self._validate_property_values(properties):
                return False

            # Enhanced property format validation for relationship properties
            if not self._validate_relationship_property_formats(rel_type, properties):
                return False

        self.logger.debug(f"Relationship '{rel_type}' validation passed")
        return True

    def get_entity_schema(self, label: str) -> Optional[EntitySchema]:
        """Get entity schema by label"""
        return get_entity_schema(label)

    def get_relationship_schema(self, rel_type: str) -> Optional[RelationshipSchema]:
        """Get relationship schema by type"""
        return get_relationship_schema(rel_type)

    def list_entity_types(self) -> List[str]:
        """Get list of all available entity types"""
        return list_entity_types()

    def list_relationship_types(self) -> List[str]:
        """Get list of all available relationship types"""
        return list_relationship_types()

    def get_entities_by_domain(self, domain: ContentDomain) -> List[EntitySchema]:
        """Get all entity schemas for a specific domain"""
        return get_entities_by_domain(domain)

    def get_relationships_by_domain(self, domain: ContentDomain) -> List[RelationshipSchema]:
        """Get all relationship schemas for a specific domain"""
        return get_relationships_by_domain(domain)

    def get_valid_relationships_for_entities(self, from_label: str, to_label: str) -> List[RelationshipSchema]:
        """Get valid relationships between two entity types"""
        return get_valid_relationships_for_entities(from_label, to_label)

    def validate_graph_structure(self, entities: List[Dict], relationships: List[Dict]) -> Dict[str, Any]:
        """
        Validate entire graph structure (entities and relationships).

        Args:
            entities: List of entity dictionaries with 'label' and 'properties'
            relationships: List of relationship dictionaries with 'type', 'from_label', 'to_label', 'properties'

        Returns:
            dict: Validation results with counts and errors
        """
        results = {
            "valid_entities": 0,
            "invalid_entities": 0,
            "valid_relationships": 0,
            "invalid_relationships": 0,
            "errors": []
        }

        # Validate entities
        for entity in entities:
            try:
                if self.validate_entity(entity.get("label"), entity.get("properties", {})):
                    results["valid_entities"] += 1
                else:
                    results["invalid_entities"] += 1
                    results["errors"].append(f"Invalid entity: {entity}")
            except Exception as e:
                results["invalid_entities"] += 1
                results["errors"].append(f"Entity validation error: {e}")

        # Validate relationships
        for rel in relationships:
            try:
                if self.validate_relationship(
                    rel.get("type"),
                    rel.get("from_label"),
                    rel.get("to_label"),
                    rel.get("properties", {})
                ):
                    results["valid_relationships"] += 1
                else:
                    results["invalid_relationships"] += 1
                    results["errors"].append(f"Invalid relationship: {rel}")
            except Exception as e:
                results["invalid_relationships"] += 1
                results["errors"].append(f"Relationship validation error: {e}")

        return results

    def _validate_property_formats(self, label: str, properties: Dict[str, Any]) -> bool:
        """
        Validate property formats based on domain-specific rules.

        Args:
            label: Entity label/type
            properties: Dictionary of properties to validate

        Returns:
            bool: True if property formats are valid, False otherwise
        """
        # Email validation for Person entities
        if label == "Person" and "email" in properties:
            email = properties["email"]
            if email and not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
                self.logger.error(f"Invalid email format: {email}")
                return False

        # URL validation for Organization entities
        if label == "Organization" and "website" in properties:
            website = properties["website"]
            if website and not re.match(r'^https?://', website):
                self.logger.error(f"Invalid website URL format: {website}")
                return False

        # Version constraint validation for Component dependencies (in relationships)
        if label == "Component" and "version" in properties:
            version = properties["version"]
            if version and not re.match(r'^[0-9]+\.[0-9]+(\.[0-9]+)?(-[a-zA-Z0-9]+)?$', version):
                self.logger.error(f"Invalid version format: {version}")
                return False

        # ID format validation (consistent patterns)
        if "id" in properties:
            entity_id = properties["id"]
            if entity_id and not re.match(r'^[a-zA-Z0-9_-]+$', entity_id):
                self.logger.error(f"Invalid ID format: {entity_id}. Must contain only letters, numbers, underscores, and hyphens")
                return False

        return True

    def _validate_relationship_property_formats(self, rel_type: str, properties: Dict[str, Any]) -> bool:
        """
        Validate relationship property formats based on domain-specific rules.

        Args:
            rel_type: Relationship type
            properties: Dictionary of relationship properties to validate

        Returns:
            bool: True if property formats are valid, False otherwise
        """
        # Version constraint validation for DEPENDS_ON relationships
        if rel_type == "DEPENDS_ON" and "version_constraint" in properties:
            version_constraint = properties["version_constraint"]
            if version_constraint and not re.match(r'^[><=^~]*[0-9]+\.[0-9]+(\.[0-9]+)?(-[a-zA-Z0-9]+)?$', version_constraint):
                self.logger.error(f"Invalid version constraint format: {version_constraint}")
                return False

        # Year validation for time-based relationships
        if "year" in properties:
            year = properties["year"]
            if year and (not isinstance(year, int) or year < 1900 or year > 2100):
                self.logger.error(f"Invalid year: {year}. Must be between 1900 and 2100")
                return False

        if "start_year" in properties:
            start_year = properties["start_year"]
            if start_year and (not isinstance(start_year, int) or start_year < 1900 or start_year > 2100):
                self.logger.error(f"Invalid start_year: {start_year}. Must be between 1900 and 2100")
                return False

        if "end_year" in properties:
            end_year = properties["end_year"]
            if end_year and (not isinstance(end_year, int) or end_year < 1900 or end_year > 2100):
                self.logger.error(f"Invalid end_year: {end_year}. Must be between 1900 and 2100")
                return False

        return True

    def _validate_property_values(self, properties: Dict[str, Any]) -> bool:
        """
        Basic validation of property values.

        Args:
            properties: Dictionary of properties to validate

        Returns:
            bool: True if properties are valid, False otherwise
        """
        for key, value in properties.items():
            # Skip None values
            if value is None:
                continue

            # Basic type checking
            if not isinstance(key, str):
                self.logger.error(f"Property key must be string, got {type(key)}")
                return False

            # Value should be serializable (basic check)
            if not self._is_serializable(value):
                self.logger.error(f"Property value for '{key}' is not serializable")
                return False

        return True

    def _is_serializable(self, value: Any) -> bool:
        """Check if value is serializable for Neo4j storage"""
        try:
            # Neo4j supports these basic types
            if isinstance(value, (str, int, float, bool, list)):
                return True
            # Check if it's a simple dict
            elif isinstance(value, dict):
                return all(isinstance(k, str) and self._is_serializable(v) for k, v in value.items())
            return False
        except:
            return False