"""
Tests for NLP entity extraction pipeline.
Following TDD approach - tests first, then implementation.
"""

import pytest
from active_rag.nlp_pipeline.entity_extractor import EntityExtractor
from active_rag.schemas.entities import ContentDomain


class TestEntityExtractor:
    """Test suite for EntityExtractor"""

    @pytest.fixture
    def extractor(self):
        """Create EntityExtractor instance for tests"""
        return EntityExtractor()

    def test_research_entity_extraction(self, extractor):
        """Test extraction of research-specific entities"""
        text = "Einstein published his theory of relativity while working at Princeton University. The research was conducted by Dr. Marie Curie at the Radium Institute."
        entities = extractor.extract_entities(text, ContentDomain.RESEARCH)

        # Check for Person entities
        person_entities = [e for e in entities if e["label"] == "Person"]
        assert len(person_entities) >= 1

        # Verify specific person extraction
        person_names = [p["properties"]["name"] for p in person_entities]
        assert any("Einstein" in name for name in person_names)

        # Check for Organization entities
        org_entities = [e for e in entities if e["label"] == "Organization"]
        assert len(org_entities) >= 1

        # Verify Princeton extraction
        org_names = [o["properties"]["name"] for o in org_entities]
        assert any("Princeton" in name for name in org_names)

    def test_technical_entity_extraction(self, extractor):
        """Test extraction of technical/API entities"""
        text = "The authentication API depends on the Redis cache component for session storage. The UserService connects to the DatabaseManager."
        entities = extractor.extract_entities(text, ContentDomain.TECHNICAL)

        # Check for Component entities
        component_entities = [e for e in entities if e["label"] == "Component"]
        assert len(component_entities) >= 2

        # Verify specific components
        component_names = [c["properties"]["name"] for c in component_entities]
        assert any("API" in name or "Service" in name for name in component_names)

    def test_business_entity_extraction(self, extractor):
        """Test extraction of business process and people entities"""
        text = "The project manager John Smith leads the Development Process. The team reports to Director Jane Doe at TechCorp Inc."
        entities = extractor.extract_entities(text, ContentDomain.BUSINESS)

        # Check for Person entities
        person_entities = [e for e in entities if e["label"] == "Person"]
        assert len(person_entities) >= 1

        # Check for Process entities
        process_entities = [e for e in entities if e["label"] == "Process"]
        assert len(process_entities) >= 1

        # Verify process extraction
        process_names = [p["properties"]["name"] for p in process_entities]
        assert any("Process" in name for name in process_names)

    def test_mixed_web_entity_extraction(self, extractor):
        """Test extraction from general web content"""
        text = "Apple Inc. CEO Tim Cook announced the new product. Microsoft Corporation is also competing in this space."
        entities = extractor.extract_entities(text, ContentDomain.MIXED_WEB)

        # Should extract general Person and Organization entities
        person_entities = [e for e in entities if e["label"] == "Person"]
        org_entities = [e for e in entities if e["label"] == "Organization"]

        assert len(person_entities) >= 1
        assert len(org_entities) >= 2

    def test_entity_id_generation(self, extractor):
        """Test that entities get consistent IDs"""
        text1 = "Einstein worked at Princeton University."
        text2 = "Princeton University had Einstein as a researcher."

        entities1 = extractor.extract_entities(text1, ContentDomain.RESEARCH)
        entities2 = extractor.extract_entities(text2, ContentDomain.RESEARCH)

        # Same entities should have same IDs
        princeton_entities1 = [e for e in entities1 if "Princeton" in e["properties"]["name"]]
        princeton_entities2 = [e for e in entities2 if "Princeton" in e["properties"]["name"]]

        if princeton_entities1 and princeton_entities2:
            assert princeton_entities1[0]["properties"]["id"] == princeton_entities2[0]["properties"]["id"]

    def test_empty_text_handling(self, extractor):
        """Test handling of empty or minimal text"""
        empty_entities = extractor.extract_entities("", ContentDomain.RESEARCH)
        assert empty_entities == []

        minimal_entities = extractor.extract_entities("Hi", ContentDomain.RESEARCH)
        assert minimal_entities == []

    def test_entity_deduplication(self, extractor):
        """Test that duplicate entities are removed"""
        text = "Einstein and Einstein worked together. Einstein was brilliant."
        entities = extractor.extract_entities(text, ContentDomain.RESEARCH)

        # Should not have duplicate Einstein entities
        person_entities = [e for e in entities if e["label"] == "Person"]
        einstein_entities = [e for e in person_entities if "Einstein" in e["properties"]["name"]]

        # All Einstein entities should have the same ID (deduplicated)
        if len(einstein_entities) > 1:
            ids = [e["properties"]["id"] for e in einstein_entities]
            assert len(set(ids)) == 1, "Duplicate entities should be deduplicated"