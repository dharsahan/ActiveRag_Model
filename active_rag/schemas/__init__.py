"""
Schema definitions and validation for the hybrid vector-graph RAG system.

This module provides comprehensive schema management across four content domains:
- Research: Academic papers, authors, institutions, concepts
- Technical: APIs, components, services, configurations
- Business: Teams, processes, people, projects
- Mixed Web: General entities from web content

Exports:
- Entity schemas and utilities
- Relationship schemas and utilities
- Content domain classifications
- Schema validation functionality
"""

from .entities import (
    ENTITY_SCHEMAS,
    EntitySchema,
    ContentDomain,
    get_entity_schema,
    get_entities_by_domain,
    list_entity_types,
    # Individual entity schemas
    PERSON_SCHEMA,
    ORGANIZATION_SCHEMA,
    CONCEPT_SCHEMA,
    PUBLICATION_SCHEMA,
    CONFERENCE_SCHEMA,
    JOURNAL_SCHEMA,
    COMPONENT_SCHEMA,
    API_SCHEMA,
    SERVICE_SCHEMA,
    CONFIGURATION_SCHEMA,
    TECHNOLOGY_SCHEMA,
    PROCESS_SCHEMA,
    PROJECT_SCHEMA,
    TEAM_SCHEMA,
    PRODUCT_SCHEMA,
    METRIC_SCHEMA,
    DOCUMENT_SCHEMA,
    WEBSITE_SCHEMA,
    TOPIC_SCHEMA,
    EVENT_SCHEMA,
    LOCATION_SCHEMA
)

from .relationships import (
    RELATIONSHIP_SCHEMAS,
    RelationshipSchema,
    get_relationship_schema,
    get_relationships_by_domain,
    list_relationship_types,
    get_valid_relationships_for_entities,
    # Individual relationship schemas
    AUTHORED_REL,
    AFFILIATED_WITH_REL,
    CITES_REL,
    COLLABORATES_WITH_REL,
    PUBLISHED_IN_REL,
    STUDIES_REL,
    REVIEWS_REL,
    DEPENDS_ON_REL,
    IMPLEMENTS_REL,
    DEPLOYED_ON_REL,
    CONSUMES_REL,
    CONFIGURES_REL,
    MONITORS_REL,
    MANAGES_REL,
    PARTICIPATES_IN_REL,
    OWNS_REL,
    BELONGS_TO_REL,
    SUPPORTS_REL,
    MEASURES_REL,
    MENTIONS_REL,
    LINKS_TO_REL,
    DISCUSSES_REL,
    LOCATED_IN_REL,
    OCCURS_AT_REL,
    TAGGED_WITH_REL,
    RELATED_TO_REL,
    SIMILAR_TO_REL
)

__all__ = [
    # Core schema classes
    'EntitySchema',
    'RelationshipSchema',
    'ContentDomain',

    # Schema registries
    'ENTITY_SCHEMAS',
    'RELATIONSHIP_SCHEMAS',

    # Utility functions
    'get_entity_schema',
    'get_relationship_schema',
    'get_entities_by_domain',
    'get_relationships_by_domain',
    'list_entity_types',
    'list_relationship_types',
    'get_valid_relationships_for_entities',

    # Individual entity schemas
    'PERSON_SCHEMA',
    'ORGANIZATION_SCHEMA',
    'CONCEPT_SCHEMA',
    'PUBLICATION_SCHEMA',
    'CONFERENCE_SCHEMA',
    'JOURNAL_SCHEMA',
    'COMPONENT_SCHEMA',
    'API_SCHEMA',
    'SERVICE_SCHEMA',
    'CONFIGURATION_SCHEMA',
    'TECHNOLOGY_SCHEMA',
    'PROCESS_SCHEMA',
    'PROJECT_SCHEMA',
    'TEAM_SCHEMA',
    'PRODUCT_SCHEMA',
    'METRIC_SCHEMA',
    'DOCUMENT_SCHEMA',
    'WEBSITE_SCHEMA',
    'TOPIC_SCHEMA',
    'EVENT_SCHEMA',
    'LOCATION_SCHEMA',

    # Individual relationship schemas
    'AUTHORED_REL',
    'AFFILIATED_WITH_REL',
    'CITES_REL',
    'COLLABORATES_WITH_REL',
    'PUBLISHED_IN_REL',
    'STUDIES_REL',
    'REVIEWS_REL',
    'DEPENDS_ON_REL',
    'IMPLEMENTS_REL',
    'DEPLOYED_ON_REL',
    'CONSUMES_REL',
    'CONFIGURES_REL',
    'MONITORS_REL',
    'MANAGES_REL',
    'PARTICIPATES_IN_REL',
    'OWNS_REL',
    'BELONGS_TO_REL',
    'SUPPORTS_REL',
    'MEASURES_REL',
    'MENTIONS_REL',
    'LINKS_TO_REL',
    'DISCUSSES_REL',
    'LOCATED_IN_REL',
    'OCCURS_AT_REL',
    'TAGGED_WITH_REL',
    'RELATED_TO_REL',
    'SIMILAR_TO_REL'
]


def get_schema_summary() -> dict:
    """
    Get summary of all available schemas.

    Returns:
        dict: Summary containing entity and relationship counts by domain
    """
    summary = {
        'entities': {
            'total': len(ENTITY_SCHEMAS),
            'by_domain': {}
        },
        'relationships': {
            'total': len(RELATIONSHIP_SCHEMAS),
            'by_domain': {}
        }
    }

    # Count entities by domain
    for domain in ContentDomain:
        entities = get_entities_by_domain(domain)
        relationships = get_relationships_by_domain(domain)

        summary['entities']['by_domain'][domain.value] = len(entities)
        summary['relationships']['by_domain'][domain.value] = len(relationships)

    return summary


def validate_schema_consistency() -> dict:
    """
    Validate consistency across all schema definitions.

    Returns:
        dict: Validation results with any inconsistencies found
    """
    issues = []

    # Check that all entity labels in relationships exist
    entity_labels = set(ENTITY_SCHEMAS.keys())

    for rel_type, rel_schema in RELATIONSHIP_SCHEMAS.items():
        # Check from_labels
        for label in rel_schema.from_labels:
            if label not in entity_labels:
                issues.append(f"Relationship '{rel_type}' references unknown from_label '{label}'")

        # Check to_labels
        for label in rel_schema.to_labels:
            if label not in entity_labels:
                issues.append(f"Relationship '{rel_type}' references unknown to_label '{label}'")

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'total_entities': len(entity_labels),
        'total_relationships': len(RELATIONSHIP_SCHEMAS)
    }