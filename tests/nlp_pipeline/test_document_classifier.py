"""
Tests for document classification pipeline.
"""

import pytest
from active_rag.nlp_pipeline.document_classifier import DocumentClassifier
from active_rag.schemas.entities import ContentDomain


class TestDocumentClassifier:
    """Test suite for DocumentClassifier"""

    @pytest.fixture
    def classifier(self):
        """Create DocumentClassifier instance for tests"""
        return DocumentClassifier()

    def test_research_document_classification(self, classifier):
        """Test classification of research documents"""
        research_text = """
        This paper presents a novel approach to machine learning in natural language processing.
        The research was conducted at Stanford University by Dr. Jane Smith and her team.
        We conducted experiments using a randomized controlled trial methodology.
        The findings indicate significant improvements in accuracy metrics.
        """

        domain = classifier.classify_document(research_text)
        assert domain == ContentDomain.RESEARCH

    def test_technical_document_classification(self, classifier):
        """Test classification of technical documents"""
        technical_text = """
        The UserAuthenticationAPI provides secure login functionality for the application.
        This service connects to the DatabaseManager component through the ConnectionPool.
        Configure the API endpoints in the config.py file and deploy using Docker.
        The service handles JWT token generation and validation.
        """

        domain = classifier.classify_document(technical_text)
        assert domain == ContentDomain.TECHNICAL

    def test_business_document_classification(self, classifier):
        """Test classification of business documents"""
        business_text = """
        The project manager will coordinate with the development team to ensure Q4 deliverables.
        Our business strategy focuses on improving customer satisfaction metrics.
        The stakeholder meeting is scheduled with the VP of Engineering next week.
        We need to optimize our workflow process for better team efficiency.
        """

        domain = classifier.classify_document(business_text)
        assert domain == ContentDomain.BUSINESS

    def test_mixed_web_document_classification(self, classifier):
        """Test classification of general web content"""
        mixed_text = """
        Apple Inc. announced new product features today.
        The company released a statement about future developments.
        This news has been covered by major media outlets.
        Users are excited about the announcement.
        """

        domain = classifier.classify_document(mixed_text)
        # Should classify as MIXED_WEB when no strong domain indicators, or BUSINESS is also acceptable
        assert domain in [ContentDomain.MIXED_WEB, ContentDomain.BUSINESS]

    def test_empty_text_classification(self, classifier):
        """Test handling of empty or minimal text"""
        empty_domain = classifier.classify_document("")
        assert empty_domain == ContentDomain.MIXED_WEB

        minimal_domain = classifier.classify_document("Hi there")
        assert minimal_domain == ContentDomain.MIXED_WEB

    def test_ambiguous_content_classification(self, classifier):
        """Test classification of content that could fit multiple domains"""
        ambiguous_text = """
        The team is working on a new research project involving machine learning APIs.
        The business requirements specify that we need to develop these components.
        """

        # Should classify into one of the specific domains, not default to MIXED_WEB
        domain = classifier.classify_document(ambiguous_text)
        assert domain in [ContentDomain.RESEARCH, ContentDomain.TECHNICAL, ContentDomain.BUSINESS]