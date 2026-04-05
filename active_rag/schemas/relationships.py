"""
Relationship schema definitions for the hybrid vector-graph RAG system.

This module defines relationship schemas across four content domains:
- Research: Academic authorship, affiliation
- Technical: Dependencies between components
- Business: Management relationships
- Mixed Web: General relationships from web content
"""

from dataclasses import dataclass
from typing import List, Optional
from .entities import ContentDomain


@dataclass
class RelationshipSchema:
    """Schema definition for graph relationships"""
    type: str
    from_labels: List[str]
    to_labels: List[str]
    required_properties: List[str]
    optional_properties: List[str]
    domain: ContentDomain
    description: str = ""
    bidirectional: bool = False


# ============================================================================
# RELATIONSHIP SCHEMAS (5 Required Schemas Only)
# ============================================================================

AUTHORED_REL = RelationshipSchema(
    type="AUTHORED",
    from_labels=["Person"],
    to_labels=["Document"],
    required_properties=[],
    optional_properties=["year", "role"],
    domain=ContentDomain.RESEARCH
)

AFFILIATED_WITH_REL = RelationshipSchema(
    type="AFFILIATED_WITH",
    from_labels=["Person"],
    to_labels=["Organization"],
    required_properties=[],
    optional_properties=["start_year", "end_year", "role"],
    domain=ContentDomain.RESEARCH
)

DEPENDS_ON_REL = RelationshipSchema(
    type="DEPENDS_ON",
    from_labels=["Component"],
    to_labels=["Component"],
    required_properties=[],
    optional_properties=["version_constraint", "dependency_type"],
    domain=ContentDomain.TECHNICAL
)

MANAGES_REL = RelationshipSchema(
    type="MANAGES",
    from_labels=["Person"],
    to_labels=["Person", "Process"],
    required_properties=[],
    optional_properties=["since", "responsibility_level"],
    domain=ContentDomain.BUSINESS
)

MENTIONS_REL = RelationshipSchema(
    type="MENTIONS",
    from_labels=["Document"],
    to_labels=["Person", "Organization", "Concept", "Component"],
    required_properties=[],
    optional_properties=["context", "sentiment", "confidence"],
    domain=ContentDomain.MIXED_WEB
)

# ============================================================================
# RELATIONSHIP SCHEMAS REGISTRY
# ============================================================================

RELATIONSHIP_SCHEMAS = {
    "AUTHORED": AUTHORED_REL,
    "AFFILIATED_WITH": AFFILIATED_WITH_REL,
    "DEPENDS_ON": DEPENDS_ON_REL,
    "MANAGES": MANAGES_REL,
    "MENTIONS": MENTIONS_REL
}


def get_relationship_schema(rel_type: str) -> Optional[RelationshipSchema]:
    """Get relationship schema by type"""
    return RELATIONSHIP_SCHEMAS.get(rel_type)


def get_relationships_by_domain(domain: ContentDomain) -> List[RelationshipSchema]:
    """Get all relationship schemas for a specific domain"""
    return [schema for schema in RELATIONSHIP_SCHEMAS.values() if schema.domain == domain]


def list_relationship_types() -> List[str]:
    """Get list of all available relationship types"""
    return list(RELATIONSHIP_SCHEMAS.keys())


def get_valid_relationships_for_entities(from_label: str, to_label: str) -> List[RelationshipSchema]:
    """Get valid relationships between two entity types"""
    valid_rels = []
    for schema in RELATIONSHIP_SCHEMAS.values():
        if from_label in schema.from_labels and to_label in schema.to_labels:
            valid_rels.append(schema)
    return valid_rels