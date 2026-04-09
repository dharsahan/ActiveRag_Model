"""NLP Pipeline router — /api/v1/nlp endpoints.

Exposes entity extraction, relation extraction, document classification,
and sentiment analysis.
"""

from __future__ import annotations

import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from active_rag.dependencies import GraphResourceManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/nlp", tags=["NLP Pipeline"])


# --- Models ---

class EntityExtractionRequest(BaseModel):
    text: str = Field(..., description="Text to extract entities from")
    domain: Optional[str] = Field(
        None,
        description="Content domain: 'research', 'technical', 'business', or 'mixed_web'. Auto-detected if omitted.",
    )


class RelationExtractionRequest(BaseModel):
    text: str = Field(..., description="Text to extract relations from")
    entities: Optional[List[dict]] = Field(
        None,
        description="Pre-extracted entities. If omitted, entities are auto-extracted first.",
    )
    chunk_id: Optional[str] = Field(None, description="Optional chunk ID for linking")


class ClassifyRequest(BaseModel):
    text: str = Field(..., description="Text to classify")


class SentimentRequest(BaseModel):
    text: str = Field(..., description="Text to analyze")


def register(r: APIRouter, graph_resources: GraphResourceManager):
    """Register NLP pipeline endpoints."""

    @r.post("/entities/extract")
    async def extract_entities(req: EntityExtractionRequest):
        """Extract named entities from text.

        Supports domain-specific extraction:
        - **research**: People, Organizations, Concepts (academic)
        - **technical**: Components, APIs, Libraries
        - **business**: People, Organizations, Processes
        - **mixed_web**: General entities

        Includes entity disambiguation (fuzzy matching) and keyword/topic extraction.

        Use cases:
        - Parse legal documents for people, orgs, and case references
        - Extract technical components from architecture docs
        - Identify key entities in FAQ content
        """
        extractor = graph_resources.entity_extractor
        if extractor is None:
            raise HTTPException(status_code=503, detail="Entity extractor is not available")

        try:
            from active_rag.schemas.entities import ContentDomain

            # Auto-detect domain if not specified
            if req.domain:
                domain_map = {
                    "research": ContentDomain.RESEARCH,
                    "technical": ContentDomain.TECHNICAL,
                    "business": ContentDomain.BUSINESS,
                    "mixed_web": ContentDomain.MIXED_WEB,
                }
                domain = domain_map.get(req.domain.lower(), ContentDomain.MIXED_WEB)
            else:
                classifier = graph_resources.document_classifier
                if classifier:
                    domain = classifier.classify_document(req.text)
                else:
                    domain = ContentDomain.MIXED_WEB

            entities = extractor.extract_entities(req.text, domain)
            return {
                "domain": domain.value if hasattr(domain, "value") else str(domain),
                "count": len(entities),
                "entities": entities,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @r.post("/relations/extract")
    async def extract_relations(req: RelationExtractionRequest):
        """Extract relationships between entities in text.

        If entities are not provided, they are auto-extracted first.
        Returns subject-predicate-object triples.

        Use cases:
        - Build knowledge graph from legal case files
        - Map organizational hierarchies from HR documents
        - Extract technology dependencies from architecture docs
        """
        rel_extractor = graph_resources.relation_extractor
        if rel_extractor is None:
            raise HTTPException(status_code=503, detail="Relation extractor is not available")

        try:
            # Auto-extract entities if not provided
            entities = req.entities
            if not entities:
                ent_extractor = graph_resources.entity_extractor
                if ent_extractor is None:
                    raise HTTPException(status_code=503, detail="Entity extractor is not available")
                from active_rag.schemas.entities import ContentDomain
                entities = ent_extractor.extract_entities(req.text, ContentDomain.MIXED_WEB)

            relations = rel_extractor.extract_relations(
                req.text, entities, chunk_id=req.chunk_id
            )
            return {
                "entity_count": len(entities),
                "relation_count": len(relations),
                "entities": entities,
                "relations": relations,
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @r.post("/classify")
    async def classify_document(req: ClassifyRequest):
        """Classify text into a content domain.

        Returns one of: research, technical, business, mixed_web.

        Use cases:
        - Route incoming documents to the right processing pipeline
        - Auto-tag content for organization
        - Determine the best entity extraction strategy
        """
        classifier = graph_resources.document_classifier
        if classifier is None:
            raise HTTPException(status_code=503, detail="Document classifier is not available")

        try:
            domain = classifier.classify_document(req.text)
            return {
                "domain": domain.value if hasattr(domain, "value") else str(domain),
                "text_length": len(req.text),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @r.post("/sentiment")
    async def analyze_sentiment(req: SentimentRequest):
        """Analyze sentiment of text.

        Returns positive/negative/neutral label with a score.

        Use cases:
        - Gauge customer satisfaction from support tickets
        - Track sentiment in feedback documents
        - Classify FAQ entries by tone
        """
        extractor = graph_resources.entity_extractor
        if extractor is None:
            raise HTTPException(status_code=503, detail="Entity extractor is not available")

        try:
            result = extractor.analyze_sentiment(req.text)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
