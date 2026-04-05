"""
Entity schema definitions for the hybrid vector-graph RAG system.

This module defines entity schemas across four content domains:
- Research: Academic papers, authors, institutions, concepts
- Technical: Components and software services
- Business: Processes and workflows
- Mixed Web: General entities from web content
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum


class ContentDomain(Enum):
    """Content domains for entity classification"""
    RESEARCH = "research"
    TECHNICAL = "technical"
    BUSINESS = "business"
    MIXED_WEB = "mixed_web"


@dataclass
class EntitySchema:
    """Schema definition for graph entities"""
    label: str
    required_properties: List[str]
    optional_properties: List[str]
    domain: ContentDomain
    description: str = ""


# ============================================================================
# ENTITY SCHEMAS (6 Required Schemas Only)
# ============================================================================

PERSON_SCHEMA = EntitySchema(
    label="Person",
    required_properties=["name", "id"],
    optional_properties=["affiliation", "email", "orcid"],
    domain=ContentDomain.RESEARCH
)

ORGANIZATION_SCHEMA = EntitySchema(
    label="Organization",
    required_properties=["name", "id"],
    optional_properties=["type", "location", "website"],
    domain=ContentDomain.RESEARCH
)

CONCEPT_SCHEMA = EntitySchema(
    label="Concept",
    required_properties=["name", "id"],
    optional_properties=["definition", "domain", "aliases"],
    domain=ContentDomain.RESEARCH
)

COMPONENT_SCHEMA = EntitySchema(
    label="Component",
    required_properties=["name", "id"],
    optional_properties=["version", "type", "description"],
    domain=ContentDomain.TECHNICAL
)

PROCESS_SCHEMA = EntitySchema(
    label="Process",
    required_properties=["name", "id"],
    optional_properties=["description", "owner", "status"],
    domain=ContentDomain.BUSINESS
)

DOCUMENT_SCHEMA = EntitySchema(
    label="Document",
    required_properties=["title", "id", "content_hash"],
    optional_properties=["url", "type", "domain", "created_at"],
    domain=ContentDomain.MIXED_WEB
)

# ============================================================================
# ENTITY SCHEMAS REGISTRY
# ============================================================================

ENTITY_SCHEMAS = {
    "Person": PERSON_SCHEMA,
    "Organization": ORGANIZATION_SCHEMA,
    "Concept": CONCEPT_SCHEMA,
    "Component": COMPONENT_SCHEMA,
    "Process": PROCESS_SCHEMA,
    "Document": DOCUMENT_SCHEMA
}


def get_entity_schema(label: str) -> Optional[EntitySchema]:
    """Get entity schema by label"""
    return ENTITY_SCHEMAS.get(label)


def get_entities_by_domain(domain: ContentDomain) -> List[EntitySchema]:
    """Get all entity schemas for a specific domain"""
    return [schema for schema in ENTITY_SCHEMAS.values() if schema.domain == domain]


def list_entity_types() -> List[str]:
    """Get list of all available entity types"""
    return list(ENTITY_SCHEMAS.keys())