#!/usr/bin/env python3
"""
Direct test of NLP modules without full package dependencies.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test DocumentClassifier
def test_document_classifier():
    """Test the document classifier in isolation"""

    # Import modules directly
    import spacy
    from enum import Enum
    import re
    import hashlib
    from typing import Dict, List, Any

    # Define ContentDomain locally to avoid dependency issues
    class ContentDomain(Enum):
        RESEARCH = "research"
        TECHNICAL = "technical"
        BUSINESS = "business"
        MIXED_WEB = "mixed_web"

    # Import the classifier code directly
    exec(open('active_rag/nlp_pipeline/document_classifier.py').read())

    try:
        classifier = DocumentClassifier()
        print("✓ DocumentClassifier initialized successfully")

        # Test research classification
        research_text = """This paper presents a novel approach to machine learning in natural language processing.
        The research was conducted at Stanford University by Dr. Jane Smith and her team.
        We conducted experiments using a randomized controlled trial methodology.
        The findings indicate significant improvements in accuracy metrics."""

        result = classifier.classify_document(research_text)
        print(f"✓ Research classification: {result}")

        # Test technical classification
        tech_text = """The UserAuthenticationAPI provides secure login functionality for the application.
        This service connects to the DatabaseManager component through the ConnectionPool.
        Configure the API endpoints in the config.py file and deploy using Docker.
        The service handles JWT token generation and validation."""

        result = classifier.classify_document(tech_text)
        print(f"✓ Technical classification: {result}")

        # Test business classification
        business_text = """The project manager will coordinate with the development team to ensure Q4 deliverables.
        Our business strategy focuses on improving customer satisfaction metrics.
        The stakeholder meeting is scheduled with the VP of Engineering next week.
        We need to optimize our workflow process for better team efficiency."""

        result = classifier.classify_document(business_text)
        print(f"✓ Business classification: {result}")

        # Test empty text
        result = classifier.classify_document("")
        print(f"✓ Empty text classification: {result}")

        print("✓ All DocumentClassifier tests completed!")
        return True

    except Exception as e:
        print(f"✗ DocumentClassifier test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_entity_extractor():
    """Test the entity extractor in isolation"""

    # Import modules directly
    import spacy
    from enum import Enum
    import re
    import hashlib
    from typing import Dict, List, Any

    # Define ContentDomain locally
    class ContentDomain(Enum):
        RESEARCH = "research"
        TECHNICAL = "technical"
        BUSINESS = "business"
        MIXED_WEB = "mixed_web"

    # Import the extractor code directly
    exec(open('active_rag/nlp_pipeline/entity_extractor.py').read())

    try:
        extractor = EntityExtractor()
        print("✓ EntityExtractor initialized successfully")

        # Test research entities
        research_text = "Einstein published his theory of relativity while working at Princeton University. The research was conducted by Dr. Marie Curie at the Radium Institute."
        entities = extractor.extract_entities(research_text, ContentDomain.RESEARCH)
        print(f"✓ Research entities extracted: {len(entities)} entities")
        for entity in entities:
            print(f"  - {entity['label']}: {entity['properties']['name']}")

        # Test technical entities
        tech_text = "The authentication API depends on the Redis cache component for session storage. The UserService connects to the DatabaseManager."
        entities = extractor.extract_entities(tech_text, ContentDomain.TECHNICAL)
        print(f"✓ Technical entities extracted: {len(entities)} entities")
        for entity in entities:
            print(f"  - {entity['label']}: {entity['properties']['name']}")

        # Test business entities
        business_text = "The project manager John Smith leads the Development Process. The team reports to Director Jane Doe at TechCorp Inc."
        entities = extractor.extract_entities(business_text, ContentDomain.BUSINESS)
        print(f"✓ Business entities extracted: {len(entities)} entities")
        for entity in entities:
            print(f"  - {entity['label']}: {entity['properties']['name']}")

        # Test empty text
        entities = extractor.extract_entities("", ContentDomain.RESEARCH)
        print(f"✓ Empty text entities: {len(entities)} entities")

        print("✓ All EntityExtractor tests completed!")
        return True

    except Exception as e:
        print(f"✗ EntityExtractor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing NLP Pipeline Components...")
    print("=" * 50)

    success = True

    print("\n1. Testing DocumentClassifier:")
    success &= test_document_classifier()

    print("\n2. Testing EntityExtractor:")
    success &= test_entity_extractor()

    print("\n" + "=" * 50)
    if success:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ Some tests failed")
        sys.exit(1)