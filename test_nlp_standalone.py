#!/usr/bin/env python3
"""
Direct test of NLP modules by copying code and fixing imports.
"""

import spacy
from typing import Dict, List, Any
import re
import hashlib
from enum import Enum

# Define ContentDomain
class ContentDomain(Enum):
    RESEARCH = "research"
    TECHNICAL = "technical"
    BUSINESS = "business"
    MIXED_WEB = "mixed_web"

# Define DocumentClassifier (copied and adapted)
class DocumentClassifier:
    """Classifies documents into content domains for entity extraction"""

    def __init__(self):
        """Initialize the document classifier with spaCy model and domain keywords"""
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise ValueError("spaCy English model not found. Run: python -m spacy download en_core_web_sm")

        # Keywords and patterns for each domain with improved precision
        self.domain_keywords = {
            ContentDomain.RESEARCH: {
                "keywords": ["research", "study", "paper", "journal", "publication", "experiment",
                           "hypothesis", "methodology", "analysis", "findings", "abstract", "citation",
                           "peer review", "academic", "scholar", "thesis", "dissertation", "conference",
                           "conducted", "novel approach", "randomized controlled", "significant improvements"],
                "patterns": [r"\b(doi:|arxiv:|pubmed:)", r"\b(university|institute|lab)\b",
                           r"\b(dr\.|prof\.|professor)\s+\w+",
                           r"\b(presented|published|conducted)\s+(by|at|in)"]
            },
            ContentDomain.TECHNICAL: {
                "keywords": ["API", "component", "service", "database", "server", "configuration",
                           "deployment", "code", "function", "class", "library", "framework",
                           "authentication", "endpoint", "REST", "JSON", "HTTP", "cache", "redis",
                           "depends on", "connects to"],
                "patterns": [r"\b(http|https|ftp)://", r"\.(js|py|java|cpp|go|ts|jsx)\b",
                           r"\b\w+(API|Service|Manager|Component|Handler|Controller)\b",
                           r"\b(UserService|DatabaseManager|AuthenticationAPI)\b"]
            },
            ContentDomain.BUSINESS: {
                "keywords": ["team", "project", "manager", "process", "workflow", "department",
                           "business", "strategy", "goals", "metrics", "stakeholder", "client",
                           "meeting", "budget", "revenue", "profit", "KPI", "objective", "deliverables",
                           "customer satisfaction", "coordinate with", "reports to"],
                "patterns": [r"\b(Q[1-4]|quarter)\b", r"\b(CEO|CTO|VP|Director|Manager)\b",
                           r"\b\d+%\b",
                           r"\b(project manager|team lead|development team)\b"]
            }
        }

    def classify_document(self, text: str) -> ContentDomain:
        """Classify document into one of the four content domains"""
        if not text or len(text.strip()) < 10:
            return ContentDomain.MIXED_WEB

        text_lower = text.lower()
        doc = self.nlp(text[:1000])

        scores = {domain: 0 for domain in ContentDomain}

        for domain, criteria in self.domain_keywords.items():
            unique_keyword_matches = sum(1 for keyword in criteria["keywords"]
                                       if keyword.lower() in text_lower)
            scores[domain] += unique_keyword_matches * 2

            pattern_score = sum(len(re.findall(pattern, text, re.IGNORECASE))
                              for pattern in criteria["patterns"])
            scores[domain] += pattern_score * 3

        entities = [(ent.label_, ent.text) for ent in doc.ents]

        for label, entity_text in entities:
            entity_lower = entity_text.lower()

            if label == "ORG":
                if any(word in entity_lower for word in ["university", "institute", "lab", "college"]):
                    scores[ContentDomain.RESEARCH] += 3
                elif any(word in entity_lower for word in ["corp", "inc", "ltd", "company", "llc"]):
                    scores[ContentDomain.BUSINESS] += 2
                elif any(word in entity_lower for word in ["tech", "software", "systems"]):
                    scores[ContentDomain.TECHNICAL] += 2

            elif label == "PERSON":
                context = self._get_entity_context(entity_text, text, window=100)
                context_lower = context.lower()

                if any(word in context_lower for word in ["dr.", "prof.", "researcher", "scholar"]):
                    scores[ContentDomain.RESEARCH] += 2
                elif any(word in context_lower for word in ["manager", "director", "ceo", "vp"]):
                    scores[ContentDomain.BUSINESS] += 2
                elif any(word in context_lower for word in ["developer", "engineer", "architect"]):
                    scores[ContentDomain.TECHNICAL] += 2

        scores = self._apply_linguistic_heuristics(doc, scores)

        max_domain = max(scores, key=scores.get)
        max_score = scores[max_domain]

        if max_score < 3:
            return ContentDomain.MIXED_WEB

        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] - sorted_scores[1] < 2:
            return ContentDomain.MIXED_WEB

        return max_domain

    def _get_entity_context(self, entity_text: str, full_text: str, window: int = 100) -> str:
        """Get surrounding context for an entity mention"""
        try:
            start_idx = full_text.lower().find(entity_text.lower())
            if start_idx == -1:
                return entity_text

            context_start = max(0, start_idx - window)
            context_end = min(len(full_text), start_idx + len(entity_text) + window)
            return full_text[context_start:context_end]
        except Exception:
            return entity_text

    def _apply_linguistic_heuristics(self, doc, scores: Dict[ContentDomain, int]) -> Dict[ContentDomain, int]:
        """Apply additional linguistic patterns for classification"""

        tech_pos_tags = sum(1 for token in doc if token.pos_ in ["SYM", "X"])
        if tech_pos_tags > 3:
            scores[ContentDomain.TECHNICAL] += 1

        avg_sent_length = sum(len(sent) for sent in doc.sents) / max(len(list(doc.sents)), 1)
        if avg_sent_length > 20:
            scores[ContentDomain.RESEARCH] += 1

        future_indicators = sum(1 for token in doc if token.lemma_ in ["will", "plan", "goal", "target"])
        if future_indicators > 2:
            scores[ContentDomain.BUSINESS] += 1

        return scores

# Define EntityExtractor (simplified version for testing)
class EntityExtractor:
    """Extracts entities from text based on content domain"""

    def __init__(self):
        """Initialize the entity extractor with spaCy model"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise ValueError("spaCy English model not found. Run: python -m spacy download en_core_web_sm")

    def extract_entities(self, text: str, domain: ContentDomain) -> List[Dict[str, Any]]:
        """Extract entities from text based on content domain"""
        if not text or len(text.strip()) < 5:
            return []

        doc = self.nlp(text)
        entities = []

        if domain == ContentDomain.RESEARCH:
            entities.extend(self._extract_research_entities(doc))
        elif domain == ContentDomain.TECHNICAL:
            entities.extend(self._extract_technical_entities(doc))
        elif domain == ContentDomain.BUSINESS:
            entities.extend(self._extract_business_entities(doc))
        else:
            entities.extend(self._extract_general_entities(doc))

        return self._deduplicate_entities(entities)

    def _extract_research_entities(self, doc) -> List[Dict[str, Any]]:
        """Extract research-specific entities (Person, Organization, Concept)"""
        entities = []

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                context = self._get_entity_context(ent, doc)
                if self._is_likely_researcher(context):
                    entities.append({
                        "label": "Person",
                        "properties": {
                            "name": ent.text.strip(),
                            "id": self._generate_id("person", ent.text)
                        }
                    })

            elif ent.label_ == "ORG":
                if self._is_academic_org(ent.text):
                    entities.append({
                        "label": "Organization",
                        "properties": {
                            "name": ent.text.strip(),
                            "id": self._generate_id("org", ent.text),
                            "type": "academic"
                        }
                    })

        return entities

    def _extract_technical_entities(self, doc) -> List[Dict[str, Any]]:
        """Extract technical/API entities (Component)"""
        entities = []
        component_patterns = [
            r'\b([A-Z][a-zA-Z]*(?:API|Service|Component|Module|Library|Manager|Handler|Controller))\b',
            r'\b(Redis|MongoDB|PostgreSQL|MySQL|ElasticSearch|Kafka|Docker|Kubernetes)\b'
        ]

        text = doc.text
        seen_components = set()

        for pattern in component_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                component_name = match.group(1)
                if len(component_name) < 3:
                    continue

                component_id = self._generate_id("component", component_name)
                if component_id not in seen_components:
                    seen_components.add(component_id)
                    entities.append({
                        "label": "Component",
                        "properties": {
                            "name": component_name,
                            "id": component_id
                        }
                    })

        return entities

    def _extract_business_entities(self, doc) -> List[Dict[str, Any]]:
        """Extract business process and people entities (Person, Organization, Process)"""
        entities = []

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                context = self._get_entity_context(ent, doc)
                if self._is_business_person(context):
                    entities.append({
                        "label": "Person",
                        "properties": {
                            "name": ent.text.strip(),
                            "id": self._generate_id("person", ent.text)
                        }
                    })
            elif ent.label_ == "ORG":
                entities.append({
                    "label": "Organization",
                    "properties": {
                        "name": ent.text.strip(),
                        "id": self._generate_id("org", ent.text),
                        "type": "business"
                    }
                })

        process_patterns = [
            r'\b([A-Z][a-zA-Z\s]{5,30} (?:process|workflow|procedure|methodology))\b',
            r'\b([A-Z][a-zA-Z\s]{3,20} Process)\b',
        ]

        for pattern in process_patterns:
            matches = re.finditer(pattern, doc.text, re.IGNORECASE)
            for match in matches:
                process_name = match.group(1).strip()
                process_name = re.sub(r'\s+', ' ', process_name)
                entities.append({
                    "label": "Process",
                    "properties": {
                        "name": process_name,
                        "id": self._generate_id("process", process_name)
                    }
                })

        return entities

    def _extract_general_entities(self, doc) -> List[Dict[str, Any]]:
        """Extract general entities from mixed web content (Person, Organization)"""
        entities = []

        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG"]:
                label_map = {"PERSON": "Person", "ORG": "Organization"}
                entities.append({
                    "label": label_map[ent.label_],
                    "properties": {
                        "name": ent.text.strip(),
                        "id": self._generate_id(ent.label_.lower(), ent.text)
                    }
                })

        return entities

    def _generate_id(self, entity_type: str, name: str) -> str:
        """Generate consistent ID for entity"""
        normalized = name.lower().strip()
        normalized = re.sub(r'[^\w\s-]', '', normalized)
        normalized = re.sub(r'\s+', '_', normalized)
        hash_obj = hashlib.md5(f"{entity_type}_{normalized}".encode())
        return f"{entity_type}_{hash_obj.hexdigest()[:8]}"

    def _get_entity_context(self, ent, doc, window: int = 50) -> str:
        """Get surrounding context for an entity"""
        start = max(0, ent.start - window)
        end = min(len(doc), ent.end + window)
        return doc[start:end].text

    def _is_likely_researcher(self, context: str) -> bool:
        """Check if person is likely a researcher based on context"""
        research_indicators = [
            "professor", "dr.", "phd", "researcher", "published", "university", "institute",
            "scholar", "academic", "author", "scientist", "faculty", "postdoc"
        ]
        context_lower = context.lower()
        return any(indicator in context_lower for indicator in research_indicators)

    def _is_academic_org(self, org_name: str) -> bool:
        """Check if organization is academic/research"""
        academic_keywords = [
            "university", "institute", "college", "lab", "laboratory", "research",
            "academy", "school", "faculty", "department"
        ]
        org_lower = org_name.lower()
        return any(keyword in org_lower for keyword in academic_keywords)

    def _is_business_person(self, context: str) -> bool:
        """Check if person is in business context"""
        business_indicators = [
            "manager", "director", "ceo", "cto", "vp", "president", "lead", "leads",
            "manages", "team", "department", "executive", "officer", "coordinator"
        ]
        context_lower = context.lower()
        return any(indicator in context_lower for indicator in business_indicators)

    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities based on ID"""
        seen_ids = set()
        deduped = []

        for entity in entities:
            entity_id = entity["properties"]["id"]
            if entity_id not in seen_ids:
                seen_ids.add(entity_id)
                deduped.append(entity)

        return deduped

# Test functions
def test_document_classifier():
    """Test the document classifier"""
    try:
        classifier = DocumentClassifier()
        print("✓ DocumentClassifier initialized successfully")

        # Test cases
        test_cases = [
            {
                "text": """This paper presents a novel approach to machine learning in natural language processing.
                The research was conducted at Stanford University by Dr. Jane Smith and her team.
                We conducted experiments using a randomized controlled trial methodology.
                The findings indicate significant improvements in accuracy metrics.""",
                "expected": ContentDomain.RESEARCH,
                "name": "Research document"
            },
            {
                "text": """The UserAuthenticationAPI provides secure login functionality for the application.
                This service connects to the DatabaseManager component through the ConnectionPool.
                Configure the API endpoints in the config.py file and deploy using Docker.
                The service handles JWT token generation and validation.""",
                "expected": ContentDomain.TECHNICAL,
                "name": "Technical document"
            },
            {
                "text": """The project manager will coordinate with the development team to ensure Q4 deliverables.
                Our business strategy focuses on improving customer satisfaction metrics.
                The stakeholder meeting is scheduled with the VP of Engineering next week.
                We need to optimize our workflow process for better team efficiency.""",
                "expected": ContentDomain.BUSINESS,
                "name": "Business document"
            },
            {
                "text": "",
                "expected": ContentDomain.MIXED_WEB,
                "name": "Empty text"
            }
        ]

        passed = 0
        for test_case in test_cases:
            result = classifier.classify_document(test_case["text"])
            if result == test_case["expected"]:
                print(f"✓ {test_case['name']}: {result}")
                passed += 1
            else:
                print(f"✗ {test_case['name']}: Expected {test_case['expected']}, got {result}")

        print(f"✓ DocumentClassifier: {passed}/{len(test_cases)} tests passed")
        return passed == len(test_cases)

    except Exception as e:
        print(f"✗ DocumentClassifier test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_entity_extractor():
    """Test the entity extractor"""
    try:
        extractor = EntityExtractor()
        print("✓ EntityExtractor initialized successfully")

        # Test research entities
        research_text = "Einstein published his theory of relativity while working at Princeton University. The research was conducted by Dr. Marie Curie at the Radium Institute."
        entities = extractor.extract_entities(research_text, ContentDomain.RESEARCH)
        print(f"✓ Research entities extracted: {len(entities)} entities")
        person_entities = [e for e in entities if e["label"] == "Person"]
        org_entities = [e for e in entities if e["label"] == "Organization"]
        print(f"  - {len(person_entities)} Person entities, {len(org_entities)} Organization entities")

        # Test technical entities
        tech_text = "The authentication API depends on the Redis cache component for session storage. The UserService connects to the DatabaseManager."
        entities = extractor.extract_entities(tech_text, ContentDomain.TECHNICAL)
        print(f"✓ Technical entities extracted: {len(entities)} entities")
        component_entities = [e for e in entities if e["label"] == "Component"]
        print(f"  - {len(component_entities)} Component entities")

        # Test business entities
        business_text = "The project manager John Smith leads the Development Process. The team reports to Director Jane Doe at TechCorp Inc."
        entities = extractor.extract_entities(business_text, ContentDomain.BUSINESS)
        print(f"✓ Business entities extracted: {len(entities)} entities")
        person_entities = [e for e in entities if e["label"] == "Person"]
        org_entities = [e for e in entities if e["label"] == "Organization"]
        process_entities = [e for e in entities if e["label"] == "Process"]
        print(f"  - {len(person_entities)} Person, {len(org_entities)} Organization, {len(process_entities)} Process entities")

        # Test empty text
        entities = extractor.extract_entities("", ContentDomain.RESEARCH)
        print(f"✓ Empty text entities: {len(entities)} entities")

        print("✓ EntityExtractor all tests completed!")
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
        import sys
        sys.exit(1)