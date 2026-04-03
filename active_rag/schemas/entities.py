"""
Entity schema definitions for the hybrid vector-graph RAG system.

This module defines comprehensive entity schemas across four content domains:
- Research: Academic papers, authors, institutions, concepts
- Technical: APIs, components, services, configurations
- Business: Teams, processes, people, projects
- Mixed Web: General entities from web content
"""

from dataclasses import dataclass
from typing import Dict, Any, List
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
# RESEARCH DOMAIN ENTITIES
# ============================================================================

PERSON_SCHEMA = EntitySchema(
    label="Person",
    required_properties=["name", "id"],
    optional_properties=["affiliation", "email", "orcid", "h_index", "specialization"],
    domain=ContentDomain.RESEARCH,
    description="Academic researchers, authors, and professionals"
)

ORGANIZATION_SCHEMA = EntitySchema(
    label="Organization",
    required_properties=["name", "id"],
    optional_properties=["type", "location", "website", "founded_year", "size"],
    domain=ContentDomain.RESEARCH,
    description="Academic institutions, research organizations, companies"
)

CONCEPT_SCHEMA = EntitySchema(
    label="Concept",
    required_properties=["name", "id"],
    optional_properties=["definition", "domain", "aliases", "category", "confidence"],
    domain=ContentDomain.RESEARCH,
    description="Academic concepts, theories, methodologies, and terminology"
)

PUBLICATION_SCHEMA = EntitySchema(
    label="Publication",
    required_properties=["title", "id"],
    optional_properties=["abstract", "year", "venue", "doi", "citation_count", "keywords"],
    domain=ContentDomain.RESEARCH,
    description="Research papers, articles, books, and academic publications"
)

CONFERENCE_SCHEMA = EntitySchema(
    label="Conference",
    required_properties=["name", "id"],
    optional_properties=["year", "location", "deadline", "acceptance_rate", "ranking"],
    domain=ContentDomain.RESEARCH,
    description="Academic conferences and venues"
)

JOURNAL_SCHEMA = EntitySchema(
    label="Journal",
    required_properties=["name", "id"],
    optional_properties=["publisher", "impact_factor", "issn", "subject_area", "frequency"],
    domain=ContentDomain.RESEARCH,
    description="Academic journals and periodicals"
)

# ============================================================================
# TECHNICAL DOMAIN ENTITIES
# ============================================================================

COMPONENT_SCHEMA = EntitySchema(
    label="Component",
    required_properties=["name", "id"],
    optional_properties=["version", "type", "description", "language", "repository"],
    domain=ContentDomain.TECHNICAL,
    description="Software components, libraries, modules, and services"
)

API_SCHEMA = EntitySchema(
    label="API",
    required_properties=["name", "id"],
    optional_properties=["version", "endpoint", "protocol", "authentication", "rate_limit"],
    domain=ContentDomain.TECHNICAL,
    description="Application Programming Interfaces"
)

SERVICE_SCHEMA = EntitySchema(
    label="Service",
    required_properties=["name", "id"],
    optional_properties=["type", "host", "port", "status", "dependencies", "health_check"],
    domain=ContentDomain.TECHNICAL,
    description="Technical services, microservices, and system components"
)

CONFIGURATION_SCHEMA = EntitySchema(
    label="Configuration",
    required_properties=["name", "id"],
    optional_properties=["environment", "value", "type", "sensitive", "last_updated"],
    domain=ContentDomain.TECHNICAL,
    description="Configuration parameters, settings, and environment variables"
)

TECHNOLOGY_SCHEMA = EntitySchema(
    label="Technology",
    required_properties=["name", "id"],
    optional_properties=["category", "version", "vendor", "license", "documentation_url"],
    domain=ContentDomain.TECHNICAL,
    description="Technologies, frameworks, programming languages, and tools"
)

# ============================================================================
# BUSINESS DOMAIN ENTITIES
# ============================================================================

PROCESS_SCHEMA = EntitySchema(
    label="Process",
    required_properties=["name", "id"],
    optional_properties=["description", "owner", "status", "steps", "duration", "frequency"],
    domain=ContentDomain.BUSINESS,
    description="Business processes, workflows, and procedures"
)

PROJECT_SCHEMA = EntitySchema(
    label="Project",
    required_properties=["name", "id"],
    optional_properties=["description", "status", "start_date", "end_date", "budget", "priority"],
    domain=ContentDomain.BUSINESS,
    description="Business projects, initiatives, and endeavors"
)

TEAM_SCHEMA = EntitySchema(
    label="Team",
    required_properties=["name", "id"],
    optional_properties=["department", "size", "manager", "responsibilities", "location"],
    domain=ContentDomain.BUSINESS,
    description="Organizational teams, departments, and groups"
)

PRODUCT_SCHEMA = EntitySchema(
    label="Product",
    required_properties=["name", "id"],
    optional_properties=["description", "version", "price", "category", "launch_date", "status"],
    domain=ContentDomain.BUSINESS,
    description="Business products, services, and offerings"
)

METRIC_SCHEMA = EntitySchema(
    label="Metric",
    required_properties=["name", "id"],
    optional_properties=["value", "unit", "target", "threshold", "measurement_date", "trend"],
    domain=ContentDomain.BUSINESS,
    description="Business metrics, KPIs, and performance indicators"
)

# ============================================================================
# MIXED WEB DOMAIN ENTITIES (Cross-domain)
# ============================================================================

DOCUMENT_SCHEMA = EntitySchema(
    label="Document",
    required_properties=["title", "id", "content_hash"],
    optional_properties=["url", "type", "domain", "created_at", "updated_at", "author", "language"],
    domain=ContentDomain.MIXED_WEB,
    description="General documents from web content, files, and various sources"
)

WEBSITE_SCHEMA = EntitySchema(
    label="Website",
    required_properties=["name", "id", "url"],
    optional_properties=["description", "category", "last_crawled", "domain_authority", "language"],
    domain=ContentDomain.MIXED_WEB,
    description="Websites, web portals, and online resources"
)

TOPIC_SCHEMA = EntitySchema(
    label="Topic",
    required_properties=["name", "id"],
    optional_properties=["description", "category", "keywords", "relevance_score", "sentiment"],
    domain=ContentDomain.MIXED_WEB,
    description="General topics and subjects from mixed web content"
)

EVENT_SCHEMA = EntitySchema(
    label="Event",
    required_properties=["name", "id"],
    optional_properties=["date", "location", "type", "description", "participants", "duration"],
    domain=ContentDomain.MIXED_WEB,
    description="Events, meetings, conferences, and occurrences"
)

LOCATION_SCHEMA = EntitySchema(
    label="Location",
    required_properties=["name", "id"],
    optional_properties=["country", "region", "coordinates", "type", "population", "timezone"],
    domain=ContentDomain.MIXED_WEB,
    description="Geographic locations, places, and addresses"
)

# ============================================================================
# ENTITY SCHEMAS REGISTRY
# ============================================================================

ENTITY_SCHEMAS = {
    # Research domain
    "Person": PERSON_SCHEMA,
    "Organization": ORGANIZATION_SCHEMA,
    "Concept": CONCEPT_SCHEMA,
    "Publication": PUBLICATION_SCHEMA,
    "Conference": CONFERENCE_SCHEMA,
    "Journal": JOURNAL_SCHEMA,

    # Technical domain
    "Component": COMPONENT_SCHEMA,
    "API": API_SCHEMA,
    "Service": SERVICE_SCHEMA,
    "Configuration": CONFIGURATION_SCHEMA,
    "Technology": TECHNOLOGY_SCHEMA,

    # Business domain
    "Process": PROCESS_SCHEMA,
    "Project": PROJECT_SCHEMA,
    "Team": TEAM_SCHEMA,
    "Product": PRODUCT_SCHEMA,
    "Metric": METRIC_SCHEMA,

    # Mixed web domain
    "Document": DOCUMENT_SCHEMA,
    "Website": WEBSITE_SCHEMA,
    "Topic": TOPIC_SCHEMA,
    "Event": EVENT_SCHEMA,
    "Location": LOCATION_SCHEMA
}


def get_entity_schema(label: str) -> EntitySchema:
    """Get entity schema by label"""
    return ENTITY_SCHEMAS.get(label)


def get_entities_by_domain(domain: ContentDomain) -> List[EntitySchema]:
    """Get all entity schemas for a specific domain"""
    return [schema for schema in ENTITY_SCHEMAS.values() if schema.domain == domain]


def list_entity_types() -> List[str]:
    """Get list of all available entity types"""
    return list(ENTITY_SCHEMAS.keys())