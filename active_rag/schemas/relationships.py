"""
Relationship schema definitions for the hybrid vector-graph RAG system.

This module defines comprehensive relationship schemas across four content domains:
- Research: Academic authorship, affiliation, citations, collaborations
- Technical: Dependencies, compositions, integrations, deployments
- Business: Management, ownership, participation, responsibilities
- Mixed Web: General relationships from web content
"""

from dataclasses import dataclass
from typing import List
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
# RESEARCH DOMAIN RELATIONSHIPS
# ============================================================================

AUTHORED_REL = RelationshipSchema(
    type="AUTHORED",
    from_labels=["Person"],
    to_labels=["Publication", "Document"],
    required_properties=[],
    optional_properties=["year", "role", "order", "corresponding_author"],
    domain=ContentDomain.RESEARCH,
    description="Person authored a publication or document"
)

AFFILIATED_WITH_REL = RelationshipSchema(
    type="AFFILIATED_WITH",
    from_labels=["Person"],
    to_labels=["Organization"],
    required_properties=[],
    optional_properties=["start_year", "end_year", "role", "department", "current"],
    domain=ContentDomain.RESEARCH,
    description="Person has affiliation with organization"
)

CITES_REL = RelationshipSchema(
    type="CITES",
    from_labels=["Publication", "Document"],
    to_labels=["Publication", "Document"],
    required_properties=[],
    optional_properties=["context", "citation_type", "page_number"],
    domain=ContentDomain.RESEARCH,
    description="Publication cites another publication"
)

COLLABORATES_WITH_REL = RelationshipSchema(
    type="COLLABORATES_WITH",
    from_labels=["Person"],
    to_labels=["Person"],
    required_properties=[],
    optional_properties=["project", "frequency", "start_year", "end_year"],
    domain=ContentDomain.RESEARCH,
    description="Person collaborates with another person",
    bidirectional=True
)

PUBLISHED_IN_REL = RelationshipSchema(
    type="PUBLISHED_IN",
    from_labels=["Publication"],
    to_labels=["Journal", "Conference"],
    required_properties=[],
    optional_properties=["year", "volume", "issue", "pages"],
    domain=ContentDomain.RESEARCH,
    description="Publication was published in venue"
)

STUDIES_REL = RelationshipSchema(
    type="STUDIES",
    from_labels=["Person", "Publication"],
    to_labels=["Concept"],
    required_properties=[],
    optional_properties=["focus_level", "methodology", "duration"],
    domain=ContentDomain.RESEARCH,
    description="Person or publication studies a concept"
)

REVIEWS_REL = RelationshipSchema(
    type="REVIEWS",
    from_labels=["Person"],
    to_labels=["Publication"],
    required_properties=[],
    optional_properties=["decision", "review_date", "review_type"],
    domain=ContentDomain.RESEARCH,
    description="Person reviews a publication"
)

# ============================================================================
# TECHNICAL DOMAIN RELATIONSHIPS
# ============================================================================

DEPENDS_ON_REL = RelationshipSchema(
    type="DEPENDS_ON",
    from_labels=["Component", "Service", "API"],
    to_labels=["Component", "Service", "API", "Technology"],
    required_properties=[],
    optional_properties=["version_constraint", "dependency_type", "required", "scope"],
    domain=ContentDomain.TECHNICAL,
    description="Component depends on another component"
)

IMPLEMENTS_REL = RelationshipSchema(
    type="IMPLEMENTS",
    from_labels=["Component", "Service"],
    to_labels=["API", "Technology"],
    required_properties=[],
    optional_properties=["version", "compliance_level", "implementation_date"],
    domain=ContentDomain.TECHNICAL,
    description="Component implements an interface or standard"
)

DEPLOYED_ON_REL = RelationshipSchema(
    type="DEPLOYED_ON",
    from_labels=["Service", "Component"],
    to_labels=["Service", "Technology"],
    required_properties=[],
    optional_properties=["environment", "deployment_date", "version", "configuration"],
    domain=ContentDomain.TECHNICAL,
    description="Service is deployed on infrastructure"
)

CONSUMES_REL = RelationshipSchema(
    type="CONSUMES",
    from_labels=["Service", "Component"],
    to_labels=["API", "Service"],
    required_properties=[],
    optional_properties=["endpoint", "method", "frequency", "authentication"],
    domain=ContentDomain.TECHNICAL,
    description="Service consumes an API or another service"
)

CONFIGURES_REL = RelationshipSchema(
    type="CONFIGURES",
    from_labels=["Configuration"],
    to_labels=["Service", "Component"],
    required_properties=[],
    optional_properties=["environment", "scope", "override", "priority"],
    domain=ContentDomain.TECHNICAL,
    description="Configuration applies to a service or component"
)

MONITORS_REL = RelationshipSchema(
    type="MONITORS",
    from_labels=["Service", "Component"],
    to_labels=["Service", "Component", "API"],
    required_properties=[],
    optional_properties=["metric_type", "threshold", "frequency", "alert_level"],
    domain=ContentDomain.TECHNICAL,
    description="Service monitors another service or component"
)

# ============================================================================
# BUSINESS DOMAIN RELATIONSHIPS
# ============================================================================

MANAGES_REL = RelationshipSchema(
    type="MANAGES",
    from_labels=["Person"],
    to_labels=["Person", "Team", "Process", "Project"],
    required_properties=[],
    optional_properties=["since", "responsibility_level", "authority", "reporting_structure"],
    domain=ContentDomain.BUSINESS,
    description="Person manages another person, team, process, or project"
)

PARTICIPATES_IN_REL = RelationshipSchema(
    type="PARTICIPATES_IN",
    from_labels=["Person", "Team"],
    to_labels=["Project", "Process", "Event"],
    required_properties=[],
    optional_properties=["role", "start_date", "end_date", "contribution_level"],
    domain=ContentDomain.BUSINESS,
    description="Person or team participates in project, process, or event"
)

OWNS_REL = RelationshipSchema(
    type="OWNS",
    from_labels=["Person", "Team", "Organization"],
    to_labels=["Product", "Process", "Project", "Component"],
    required_properties=[],
    optional_properties=["since", "ownership_type", "responsibility", "authority_level"],
    domain=ContentDomain.BUSINESS,
    description="Entity owns another entity"
)

BELONGS_TO_REL = RelationshipSchema(
    type="BELONGS_TO",
    from_labels=["Person"],
    to_labels=["Team", "Organization"],
    required_properties=[],
    optional_properties=["role", "start_date", "end_date", "full_time", "location"],
    domain=ContentDomain.BUSINESS,
    description="Person belongs to team or organization"
)

SUPPORTS_REL = RelationshipSchema(
    type="SUPPORTS",
    from_labels=["Team", "Person", "Process"],
    to_labels=["Product", "Service", "Project"],
    required_properties=[],
    optional_properties=["support_type", "level", "hours", "contact_method"],
    domain=ContentDomain.BUSINESS,
    description="Entity provides support for another entity"
)

MEASURES_REL = RelationshipSchema(
    type="MEASURES",
    from_labels=["Metric"],
    to_labels=["Product", "Service", "Process", "Project", "Team"],
    required_properties=[],
    optional_properties=["measurement_frequency", "target_value", "baseline"],
    domain=ContentDomain.BUSINESS,
    description="Metric measures performance of an entity"
)

# ============================================================================
# MIXED WEB DOMAIN RELATIONSHIPS (Cross-domain)
# ============================================================================

MENTIONS_REL = RelationshipSchema(
    type="MENTIONS",
    from_labels=["Document", "Website"],
    to_labels=["Person", "Organization", "Concept", "Component", "Product", "Location", "Event"],
    required_properties=[],
    optional_properties=["context", "sentiment", "confidence", "frequency", "position"],
    domain=ContentDomain.MIXED_WEB,
    description="Document or website mentions an entity"
)

LINKS_TO_REL = RelationshipSchema(
    type="LINKS_TO",
    from_labels=["Document", "Website"],
    to_labels=["Document", "Website"],
    required_properties=[],
    optional_properties=["link_type", "anchor_text", "context", "follow"],
    domain=ContentDomain.MIXED_WEB,
    description="Document or website links to another document or website"
)

DISCUSSES_REL = RelationshipSchema(
    type="DISCUSSES",
    from_labels=["Document", "Website"],
    to_labels=["Topic", "Concept"],
    required_properties=[],
    optional_properties=["depth", "perspective", "coverage", "sentiment"],
    domain=ContentDomain.MIXED_WEB,
    description="Document or website discusses a topic or concept"
)

LOCATED_IN_REL = RelationshipSchema(
    type="LOCATED_IN",
    from_labels=["Person", "Organization", "Event"],
    to_labels=["Location"],
    required_properties=[],
    optional_properties=["address", "since", "until", "temporary"],
    domain=ContentDomain.MIXED_WEB,
    description="Entity is located in a place"
)

OCCURS_AT_REL = RelationshipSchema(
    type="OCCURS_AT",
    from_labels=["Event"],
    to_labels=["Location"],
    required_properties=[],
    optional_properties=["venue", "address", "capacity", "accessibility"],
    domain=ContentDomain.MIXED_WEB,
    description="Event occurs at a location"
)

TAGGED_WITH_REL = RelationshipSchema(
    type="TAGGED_WITH",
    from_labels=["Document", "Website"],
    to_labels=["Topic"],
    required_properties=[],
    optional_properties=["confidence", "relevance", "source", "auto_generated"],
    domain=ContentDomain.MIXED_WEB,
    description="Content is tagged with a topic"
)

# ============================================================================
# CROSS-DOMAIN RELATIONSHIPS
# ============================================================================

RELATED_TO_REL = RelationshipSchema(
    type="RELATED_TO",
    from_labels=["Person", "Organization", "Concept", "Component", "Product", "Document"],
    to_labels=["Person", "Organization", "Concept", "Component", "Product", "Document"],
    required_properties=[],
    optional_properties=["relationship_type", "strength", "context", "bidirectional"],
    domain=ContentDomain.MIXED_WEB,
    description="Generic relationship between entities",
    bidirectional=True
)

SIMILAR_TO_REL = RelationshipSchema(
    type="SIMILAR_TO",
    from_labels=["Document", "Concept", "Product", "Component"],
    to_labels=["Document", "Concept", "Product", "Component"],
    required_properties=[],
    optional_properties=["similarity_score", "similarity_type", "computed_method"],
    domain=ContentDomain.MIXED_WEB,
    description="Entities are similar to each other",
    bidirectional=True
)

# ============================================================================
# RELATIONSHIP SCHEMAS REGISTRY
# ============================================================================

RELATIONSHIP_SCHEMAS = {
    # Research domain
    "AUTHORED": AUTHORED_REL,
    "AFFILIATED_WITH": AFFILIATED_WITH_REL,
    "CITES": CITES_REL,
    "COLLABORATES_WITH": COLLABORATES_WITH_REL,
    "PUBLISHED_IN": PUBLISHED_IN_REL,
    "STUDIES": STUDIES_REL,
    "REVIEWS": REVIEWS_REL,

    # Technical domain
    "DEPENDS_ON": DEPENDS_ON_REL,
    "IMPLEMENTS": IMPLEMENTS_REL,
    "DEPLOYED_ON": DEPLOYED_ON_REL,
    "CONSUMES": CONSUMES_REL,
    "CONFIGURES": CONFIGURES_REL,
    "MONITORS": MONITORS_REL,

    # Business domain
    "MANAGES": MANAGES_REL,
    "PARTICIPATES_IN": PARTICIPATES_IN_REL,
    "OWNS": OWNS_REL,
    "BELONGS_TO": BELONGS_TO_REL,
    "SUPPORTS": SUPPORTS_REL,
    "MEASURES": MEASURES_REL,

    # Mixed web domain
    "MENTIONS": MENTIONS_REL,
    "LINKS_TO": LINKS_TO_REL,
    "DISCUSSES": DISCUSSES_REL,
    "LOCATED_IN": LOCATED_IN_REL,
    "OCCURS_AT": OCCURS_AT_REL,
    "TAGGED_WITH": TAGGED_WITH_REL,

    # Cross-domain
    "RELATED_TO": RELATED_TO_REL,
    "SIMILAR_TO": SIMILAR_TO_REL
}


def get_relationship_schema(rel_type: str) -> RelationshipSchema:
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