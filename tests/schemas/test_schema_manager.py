import pytest
from unittest.mock import Mock, MagicMock
from active_rag.knowledge_graph.schema_manager import SchemaManager
from active_rag.schemas.entities import ContentDomain


def test_create_constraints():
    # Create mock client that simulates successful constraint creation
    mock_client = Mock()
    mock_session = MagicMock()
    mock_client._driver.session.return_value = mock_session
    mock_session.__enter__.return_value = mock_session
    mock_session.__exit__.return_value = None
    mock_session.run.return_value = None  # Simulate successful constraint creation

    schema = SchemaManager(mock_client)
    result = schema.create_base_constraints()
    assert result == True


def test_entity_validation():
    # Create mock client (not used in validation, only for initialization)
    mock_client = Mock()
    schema = SchemaManager(mock_client)
    valid = schema.validate_entity("Person", {"name": "John", "id": "p1"})
    assert valid == True


# Additional comprehensive test functions for better coverage
def test_entity_validation_invalid():
    """Test entity validation with missing required properties"""
    mock_client = Mock()
    schema = SchemaManager(mock_client)

    # Test missing required property
    invalid = schema.validate_entity("Person", {"name": "John"})  # Missing 'id'
    assert invalid == False

    # Test unknown entity type
    invalid = schema.validate_entity("Unknown", {"name": "Test", "id": "u1"})
    assert invalid == False

    # Test invalid property types
    invalid = schema.validate_entity("Person", {"name": None, "id": "p1"})  # None required property
    assert invalid == False


def test_relationship_validation_valid():
    """Test valid relationship validation scenarios"""
    mock_client = Mock()
    schema = SchemaManager(mock_client)

    # Test valid relationships
    valid = schema.validate_relationship("AUTHORED", "Person", "Document")
    assert valid == True

    valid = schema.validate_relationship("AFFILIATED_WITH", "Person", "Organization")
    assert valid == True

    valid = schema.validate_relationship("DEPENDS_ON", "Component", "Component")
    assert valid == True

    # Test with valid properties
    valid = schema.validate_relationship(
        "AUTHORED",
        "Person",
        "Document",
        {"year": 2023, "role": "primary"}
    )
    assert valid == True


def test_relationship_validation_invalid():
    """Test invalid relationship validation scenarios"""
    mock_client = Mock()
    schema = SchemaManager(mock_client)

    # Test unknown relationship type
    invalid = schema.validate_relationship("UNKNOWN_REL", "Person", "Document")
    assert invalid == False

    # Test invalid from_label
    invalid = schema.validate_relationship("AUTHORED", "Organization", "Document")
    assert invalid == False

    # Test invalid to_label
    invalid = schema.validate_relationship("AUTHORED", "Person", "Person")
    assert invalid == False

    # Test invalid year in properties
    invalid = schema.validate_relationship(
        "AUTHORED",
        "Person",
        "Document",
        {"year": 1800}  # Invalid year
    )
    assert invalid == False


def test_property_validation():
    """Test domain-specific property validation"""
    mock_client = Mock()
    schema = SchemaManager(mock_client)

    # Test email validation for Person
    invalid = schema.validate_entity("Person", {
        "name": "John",
        "id": "p1",
        "email": "invalid-email"
    })
    assert invalid == False

    valid = schema.validate_entity("Person", {
        "name": "John",
        "id": "p1",
        "email": "john@example.com"
    })
    assert valid == True

    # Test website URL validation for Organization
    invalid = schema.validate_entity("Organization", {
        "name": "Acme Corp",
        "id": "org1",
        "website": "not-a-url"
    })
    assert invalid == False

    valid = schema.validate_entity("Organization", {
        "name": "Acme Corp",
        "id": "org1",
        "website": "https://example.com"
    })
    assert valid == True

    # Test ID format validation
    invalid = schema.validate_entity("Person", {
        "name": "John",
        "id": "invalid@id!"
    })
    assert invalid == False

    valid = schema.validate_entity("Person", {
        "name": "John",
        "id": "valid-id_123"
    })
    assert valid == True

    # Test version format validation for Component
    invalid = schema.validate_entity("Component", {
        "name": "My Component",
        "id": "comp1",
        "version": "invalid-version"
    })
    assert invalid == False

    valid = schema.validate_entity("Component", {
        "name": "My Component",
        "id": "comp1",
        "version": "1.2.3"
    })
    assert valid == True


def test_schema_utility_functions():
    """Test schema introspection utilities"""
    mock_client = Mock()
    schema = SchemaManager(mock_client)

    # Test get_entity_schema
    person_schema = schema.get_entity_schema("Person")
    assert person_schema is not None
    assert person_schema.label == "Person"
    assert "name" in person_schema.required_properties
    assert "id" in person_schema.required_properties

    unknown_schema = schema.get_entity_schema("Unknown")
    assert unknown_schema is None

    # Test get_relationship_schema
    authored_schema = schema.get_relationship_schema("AUTHORED")
    assert authored_schema is not None
    assert authored_schema.type == "AUTHORED"
    assert "Person" in authored_schema.from_labels
    assert "Document" in authored_schema.to_labels

    unknown_rel_schema = schema.get_relationship_schema("UNKNOWN")
    assert unknown_rel_schema is None

    # Test list functions
    entity_types = schema.list_entity_types()
    assert len(entity_types) == 6
    assert "Person" in entity_types
    assert "Organization" in entity_types

    rel_types = schema.list_relationship_types()
    assert len(rel_types) == 5
    assert "AUTHORED" in rel_types
    assert "AFFILIATED_WITH" in rel_types


def test_graph_structure_validation():
    """Test validation of entire graph structures"""
    mock_client = Mock()
    schema = SchemaManager(mock_client)

    entities = [
        {"label": "Person", "properties": {"name": "John", "id": "p1"}},
        {"label": "Document", "properties": {"title": "Paper", "id": "d1", "content_hash": "hash123"}},
        {"label": "InvalidEntity", "properties": {"name": "Test"}}  # Invalid entity
    ]

    relationships = [
        {"type": "AUTHORED", "from_label": "Person", "to_label": "Document", "properties": {}},
        {"type": "UNKNOWN", "from_label": "Person", "to_label": "Document", "properties": {}}  # Invalid rel
    ]

    results = schema.validate_graph_structure(entities, relationships)

    assert results["valid_entities"] == 2
    assert results["invalid_entities"] == 1
    assert results["valid_relationships"] == 1
    assert results["invalid_relationships"] == 1
    assert len(results["errors"]) == 2


def test_domain_based_schema_retrieval():
    """Test retrieving schemas by content domain"""
    mock_client = Mock()
    schema = SchemaManager(mock_client)

    # Test get entities by domain
    research_entities = schema.get_entities_by_domain(ContentDomain.RESEARCH)
    assert len(research_entities) == 3  # Person, Organization, Concept

    technical_entities = schema.get_entities_by_domain(ContentDomain.TECHNICAL)
    assert len(technical_entities) == 1  # Component

    business_entities = schema.get_entities_by_domain(ContentDomain.BUSINESS)
    assert len(business_entities) == 1  # Process

    # Test get relationships by domain
    research_rels = schema.get_relationships_by_domain(ContentDomain.RESEARCH)
    assert len(research_rels) == 2  # AUTHORED, AFFILIATED_WITH

    technical_rels = schema.get_relationships_by_domain(ContentDomain.TECHNICAL)
    assert len(technical_rels) == 1  # DEPENDS_ON


def test_valid_relationships_for_entities():
    """Test finding valid relationships between entity types"""
    mock_client = Mock()
    schema = SchemaManager(mock_client)

    # Test Person to Document relationships
    person_doc_rels = schema.get_valid_relationships_for_entities("Person", "Document")
    rel_types = [rel.type for rel in person_doc_rels]
    assert "AUTHORED" in rel_types

    # Test Document to Person relationships
    doc_person_rels = schema.get_valid_relationships_for_entities("Document", "Person")
    rel_types = [rel.type for rel in doc_person_rels]
    assert "MENTIONS" in rel_types

    # Test Component to Component relationships
    comp_comp_rels = schema.get_valid_relationships_for_entities("Component", "Component")
    rel_types = [rel.type for rel in comp_comp_rels]
    assert "DEPENDS_ON" in rel_types

    # Test no valid relationships
    no_rels = schema.get_valid_relationships_for_entities("Document", "Process")
    assert len(no_rels) == 0


def test_relationship_property_format_validation():
    """Test relationship-specific property format validation"""
    mock_client = Mock()
    schema = SchemaManager(mock_client)

    # Test version constraint validation for DEPENDS_ON
    invalid = schema.validate_relationship(
        "DEPENDS_ON",
        "Component",
        "Component",
        {"version_constraint": "invalid-constraint"}
    )
    assert invalid == False

    valid = schema.validate_relationship(
        "DEPENDS_ON",
        "Component",
        "Component",
        {"version_constraint": "^1.2.3"}
    )
    assert valid == True

    # Test year validation
    invalid = schema.validate_relationship(
        "AUTHORED",
        "Person",
        "Document",
        {"year": 3000}  # Future year beyond reasonable range
    )
    assert invalid == False

    valid = schema.validate_relationship(
        "AUTHORED",
        "Person",
        "Document",
        {"year": 2023}
    )
    assert valid == True


def test_edge_cases():
    """Test edge cases and error handling"""
    mock_client = Mock()
    schema = SchemaManager(mock_client)

    # Test empty properties
    valid = schema.validate_entity("Person", {"name": "John", "id": "p1", "email": ""})
    assert valid == True  # Empty string should be allowed

    # Test None properties (should be allowed for optional properties)
    valid = schema.validate_entity("Person", {"name": "John", "id": "p1", "email": None})
    assert valid == True

    # Test properties with special characters in values
    valid = schema.validate_entity("Person", {
        "name": "John O'Connor",
        "id": "p1"
    })
    assert valid == True

    # Test very long strings
    long_name = "A" * 1000
    valid = schema.validate_entity("Person", {
        "name": long_name,
        "id": "p1"
    })
    assert valid == True  # Should handle long strings

    # Test serialization edge cases
    valid = schema.validate_entity("Person", {
        "name": "John",
        "id": "p1",
        "metadata": {"key": "value", "nested": {"data": 123}}
    })
    assert valid == True  # Nested dict should be serializable