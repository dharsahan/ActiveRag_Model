import pytest
from unittest.mock import Mock
from active_rag.knowledge_graph.schema_manager import SchemaManager
from active_rag.schemas import ContentDomain


class TestSchemaManagerValidation:
    """Test suite for schema validation logic without database dependencies"""

    @pytest.fixture
    def mock_client(self):
        """Create simple mock Neo4j client"""
        return Mock()

    @pytest.fixture
    def schema_manager(self, mock_client):
        """Create schema manager for testing"""
        return SchemaManager(mock_client)

    def test_entity_validation_valid(self, schema_manager):
        """Test validation of valid entity properties"""
        # Test Person entity with required properties
        valid = schema_manager.validate_entity("Person", {"name": "John Doe", "id": "p1"})
        assert valid == True

        # Test Person entity with optional properties
        valid = schema_manager.validate_entity("Person", {
            "name": "Jane Smith",
            "id": "p2",
            "affiliation": "MIT",
            "email": "jane@mit.edu"
        })
        assert valid == True

        # Test Organization entity
        valid = schema_manager.validate_entity("Organization", {
            "name": "MIT",
            "id": "org1"
        })
        assert valid == True

        # Test Document entity
        valid = schema_manager.validate_entity("Document", {
            "title": "Research Paper",
            "id": "doc1",
            "content_hash": "abc123"
        })
        assert valid == True

        # Test Component entity
        valid = schema_manager.validate_entity("Component", {
            "name": "React",
            "id": "comp1"
        })
        assert valid == True

        # Test Process entity
        valid = schema_manager.validate_entity("Process", {
            "name": "Code Review",
            "id": "proc1"
        })
        assert valid == True

    def test_entity_validation_invalid(self, schema_manager):
        """Test validation of invalid entity properties"""
        # Missing required property 'name'
        valid = schema_manager.validate_entity("Person", {"id": "p1"})
        assert valid == False

        # Missing required property 'id'
        valid = schema_manager.validate_entity("Person", {"name": "John"})
        assert valid == False

        # Unknown entity type
        valid = schema_manager.validate_entity("UnknownType", {"name": "test", "id": "test1"})
        assert valid == False

        # Missing required property 'content_hash' for Document
        valid = schema_manager.validate_entity("Document", {"title": "Paper", "id": "doc1"})
        assert valid == False

    def test_relationship_validation_valid(self, schema_manager):
        """Test validation of valid relationships"""
        # Person AUTHORED Publication
        valid = schema_manager.validate_relationship("AUTHORED", "Person", "Publication")
        assert valid == True

        # Person AUTHORED Document
        valid = schema_manager.validate_relationship("AUTHORED", "Person", "Document")
        assert valid == True

        # Person AFFILIATED_WITH Organization
        valid = schema_manager.validate_relationship("AFFILIATED_WITH", "Person", "Organization")
        assert valid == True

        # Document MENTIONS Person
        valid = schema_manager.validate_relationship("MENTIONS", "Document", "Person")
        assert valid == True

        # Component DEPENDS_ON Component
        valid = schema_manager.validate_relationship("DEPENDS_ON", "Component", "Component")
        assert valid == True

        # Person MANAGES Person
        valid = schema_manager.validate_relationship("MANAGES", "Person", "Person")
        assert valid == True

    def test_relationship_validation_invalid(self, schema_manager):
        """Test validation of invalid relationships"""
        # Invalid relationship type
        valid = schema_manager.validate_relationship("INVALID_REL", "Person", "Document")
        assert valid == False

        # Invalid from_label for AUTHORED relationship
        valid = schema_manager.validate_relationship("AUTHORED", "Organization", "Document")
        assert valid == False

        # Invalid to_label for AUTHORED relationship
        valid = schema_manager.validate_relationship("AUTHORED", "Person", "Component")
        assert valid == False

        # Invalid from_label for DEPENDS_ON relationship
        valid = schema_manager.validate_relationship("DEPENDS_ON", "Person", "Component")
        assert valid == False

    def test_relationship_validation_with_properties(self, schema_manager):
        """Test relationship validation with properties"""
        # Valid relationship with properties
        valid = schema_manager.validate_relationship("AUTHORED", "Person", "Document", {
            "year": "2024",
            "role": "lead_author"
        })
        assert valid == True

        # Empty properties should still be valid for relationships without required properties
        valid = schema_manager.validate_relationship("AUTHORED", "Person", "Document", {})
        assert valid == True

        # Valid relationship with different properties
        valid = schema_manager.validate_relationship("AFFILIATED_WITH", "Person", "Organization", {
            "start_year": "2020",
            "role": "researcher"
        })
        assert valid == True

    def test_get_entity_schema(self, schema_manager):
        """Test retrieving entity schemas"""
        person_schema = schema_manager.get_entity_schema("Person")
        assert person_schema is not None
        assert person_schema.label == "Person"
        assert "name" in person_schema.required_properties
        assert "id" in person_schema.required_properties

        # Test another entity schema
        doc_schema = schema_manager.get_entity_schema("Document")
        assert doc_schema is not None
        assert doc_schema.label == "Document"
        assert "title" in doc_schema.required_properties
        assert "content_hash" in doc_schema.required_properties

        # Test non-existent schema
        unknown_schema = schema_manager.get_entity_schema("UnknownEntity")
        assert unknown_schema is None

    def test_get_relationship_schema(self, schema_manager):
        """Test retrieving relationship schemas"""
        authored_schema = schema_manager.get_relationship_schema("AUTHORED")
        assert authored_schema is not None
        assert authored_schema.type == "AUTHORED"
        assert "Person" in authored_schema.from_labels
        assert "Document" in authored_schema.to_labels or "Publication" in authored_schema.to_labels

        # Test another relationship schema
        mentions_schema = schema_manager.get_relationship_schema("MENTIONS")
        assert mentions_schema is not None
        assert mentions_schema.type == "MENTIONS"

        # Test non-existent schema
        unknown_schema = schema_manager.get_relationship_schema("UNKNOWN_REL")
        assert unknown_schema is None

    def test_list_entity_types(self, schema_manager):
        """Test listing available entity types"""
        entity_types = schema_manager.list_entity_types()
        assert isinstance(entity_types, list)
        assert "Person" in entity_types
        assert "Organization" in entity_types
        assert "Document" in entity_types
        assert "Component" in entity_types
        assert "Concept" in entity_types
        assert "Process" in entity_types
        assert "Publication" in entity_types

    def test_list_relationship_types(self, schema_manager):
        """Test listing available relationship types"""
        rel_types = schema_manager.list_relationship_types()
        assert isinstance(rel_types, list)
        assert "AUTHORED" in rel_types
        assert "AFFILIATED_WITH" in rel_types
        assert "MENTIONS" in rel_types
        assert "DEPENDS_ON" in rel_types
        assert "MANAGES" in rel_types

    def test_get_entities_by_domain(self, schema_manager):
        """Test getting entities by content domain"""
        research_entities = schema_manager.get_entities_by_domain(ContentDomain.RESEARCH)
        assert len(research_entities) > 0

        # Check that research entities are included
        research_labels = [e.label for e in research_entities]
        assert "Person" in research_labels
        assert "Organization" in research_labels
        assert "Concept" in research_labels

        technical_entities = schema_manager.get_entities_by_domain(ContentDomain.TECHNICAL)
        assert len(technical_entities) > 0

        technical_labels = [e.label for e in technical_entities]
        assert "Component" in technical_labels

    def test_get_relationships_by_domain(self, schema_manager):
        """Test getting relationships by content domain"""
        research_rels = schema_manager.get_relationships_by_domain(ContentDomain.RESEARCH)
        assert len(research_rels) > 0

        research_types = [r.type for r in research_rels]
        assert "AUTHORED" in research_types
        assert "AFFILIATED_WITH" in research_types

        technical_rels = schema_manager.get_relationships_by_domain(ContentDomain.TECHNICAL)
        assert len(technical_rels) > 0

        technical_types = [r.type for r in technical_rels]
        assert "DEPENDS_ON" in technical_types

    def test_get_valid_relationships_for_entities(self, schema_manager):
        """Test getting valid relationships between entity types"""
        # Person to Document relationships
        person_doc_rels = schema_manager.get_valid_relationships_for_entities("Person", "Document")
        assert len(person_doc_rels) > 0

        rel_types = [r.type for r in person_doc_rels]
        assert "AUTHORED" in rel_types

        # Component to Component relationships
        comp_comp_rels = schema_manager.get_valid_relationships_for_entities("Component", "Component")
        assert len(comp_comp_rels) > 0

        rel_types = [r.type for r in comp_comp_rels]
        assert "DEPENDS_ON" in rel_types

    def test_validate_graph_structure(self, schema_manager):
        """Test validation of entire graph structure"""
        entities = [
            {"label": "Person", "properties": {"name": "John", "id": "p1"}},
            {"label": "Document", "properties": {"title": "Paper", "id": "d1", "content_hash": "hash1"}},
            {"label": "Person", "properties": {"name": "Jane"}},  # Invalid - missing id
        ]

        relationships = [
            {"type": "AUTHORED", "from_label": "Person", "to_label": "Document", "properties": {}},
            {"type": "INVALID_REL", "from_label": "Person", "to_label": "Document", "properties": {}},  # Invalid
        ]

        results = schema_manager.validate_graph_structure(entities, relationships)

        assert results["valid_entities"] == 2
        assert results["invalid_entities"] == 1
        assert results["valid_relationships"] == 1
        assert results["invalid_relationships"] == 1
        assert len(results["errors"]) > 0


class TestSchemaRegistries:
    """Test the schema registries and utility functions directly"""

    def test_entity_schemas_loaded(self):
        """Test that all entity schemas are properly loaded"""
        from active_rag.schemas import ENTITY_SCHEMAS

        # Check that we have all expected entity types
        expected_entities = [
            "Person", "Organization", "Concept", "Publication", "Conference", "Journal",
            "Component", "API", "Service", "Configuration", "Technology",
            "Process", "Project", "Team", "Product", "Metric",
            "Document", "Website", "Topic", "Event", "Location"
        ]

        for entity in expected_entities:
            assert entity in ENTITY_SCHEMAS, f"Missing entity schema: {entity}"

    def test_relationship_schemas_loaded(self):
        """Test that all relationship schemas are properly loaded"""
        from active_rag.schemas import RELATIONSHIP_SCHEMAS

        # Check that we have all expected relationship types
        expected_relationships = [
            "AUTHORED", "AFFILIATED_WITH", "CITES", "COLLABORATES_WITH", "PUBLISHED_IN", "STUDIES", "REVIEWS",
            "DEPENDS_ON", "IMPLEMENTS", "DEPLOYED_ON", "CONSUMES", "CONFIGURES", "MONITORS",
            "MANAGES", "PARTICIPATES_IN", "OWNS", "BELONGS_TO", "SUPPORTS", "MEASURES",
            "MENTIONS", "LINKS_TO", "DISCUSSES", "LOCATED_IN", "OCCURS_AT", "TAGGED_WITH",
            "RELATED_TO", "SIMILAR_TO"
        ]

        for relationship in expected_relationships:
            assert relationship in RELATIONSHIP_SCHEMAS, f"Missing relationship schema: {relationship}"

    def test_schema_utility_functions(self):
        """Test schema utility functions"""
        from active_rag.schemas import (
            get_entity_schema,
            get_relationship_schema,
            list_entity_types,
            list_relationship_types
        )

        # Test entity utility functions
        person_schema = get_entity_schema("Person")
        assert person_schema is not None
        assert person_schema.label == "Person"

        entity_types = list_entity_types()
        assert "Person" in entity_types

        # Test relationship utility functions
        authored_schema = get_relationship_schema("AUTHORED")
        assert authored_schema is not None
        assert authored_schema.type == "AUTHORED"

        rel_types = list_relationship_types()
        assert "AUTHORED" in rel_types