"""
Domain-specific entity extraction for the hybrid RAG system.
Extracts entities based on content domain classification.
"""

import spacy
import hashlib
import re
from typing import Dict, List, Any, Tuple
from ..schemas.entities import ContentDomain, ENTITY_SCHEMAS


class EntityExtractor:
    """Extracts entities from text based on content domain"""

    def __init__(self):
        """Initialize the entity extractor with spaCy model"""
        # Load spaCy model
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
        else:  # MIXED_WEB
            entities.extend(self._extract_general_entities(doc))

        return self._deduplicate_entities(entities)

    def _extract_research_entities(self, doc) -> List[Dict[str, Any]]:
        """Extract research-specific entities (Person, Organization, Concept)"""
        entities = []

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Check if likely to be an author/researcher
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
                # Check if academic/research organization
                if self._is_academic_org(ent.text):
                    entities.append({
                        "label": "Organization",
                        "properties": {
                            "name": ent.text.strip(),
                            "id": self._generate_id("org", ent.text),
                            "type": "academic"
                        }
                    })

        # Extract concepts from noun phrases and technical terms
        concepts = self._extract_concepts(doc)
        entities.extend(concepts)

        return entities

    def _extract_technical_entities(self, doc) -> List[Dict[str, Any]]:
        """Extract technical/API entities (Component)"""
        entities = []

        # Look for technical components using various patterns
        component_patterns = [
            r'\b([A-Z][a-zA-Z]*(?:API|Service|Component|Module|Library|Manager|Handler|Controller))\b',
            r'\b([a-z]+(?:_[a-z]+)*\.(?:js|py|java|cpp|go|ts|jsx))\b',  # File names
            r'\b([A-Z][a-zA-Z]*[A-Z][a-zA-Z]*)\b',  # CamelCase terms
            r'\b(Redis|MongoDB|PostgreSQL|MySQL|ElasticSearch|Kafka|Docker|Kubernetes)\b'  # Common tech
        ]

        text = doc.text
        seen_components = set()

        for pattern in component_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                component_name = match.group(1)
                # Avoid very short or common words
                if len(component_name) < 3 or component_name.lower() in ['the', 'and', 'for', 'are']:
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

        # Extract from named entities that look technical
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT", "ORG"] and self._is_technical_entity(ent.text):
                component_id = self._generate_id("component", ent.text)
                if component_id not in seen_components:
                    seen_components.add(component_id)
                    entities.append({
                        "label": "Component",
                        "properties": {
                            "name": ent.text.strip(),
                            "id": component_id
                        }
                    })

        return entities

    def _extract_business_entities(self, doc) -> List[Dict[str, Any]]:
        """Extract business process and people entities (Person, Organization, Process)"""
        entities = []

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Business people (managers, etc)
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
                # Business organizations
                entities.append({
                    "label": "Organization",
                    "properties": {
                        "name": ent.text.strip(),
                        "id": self._generate_id("org", ent.text),
                        "type": "business"
                    }
                })

        # Extract processes from text patterns
        process_patterns = [
            r'\b([A-Z][a-zA-Z\s]{5,30} (?:process|workflow|procedure|methodology))\b',
            r'\b([A-Z][a-zA-Z\s]{3,20} Process)\b',
            r'\b((?:Development|Testing|Deployment|Review|Approval) Process)\b'
        ]

        for pattern in process_patterns:
            matches = re.finditer(pattern, doc.text, re.IGNORECASE)
            for match in matches:
                process_name = match.group(1).strip()
                # Clean up the process name
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

        # Fallback: Extract single-letter entities (A, B, C) in relationship contexts
        # Patterns like "A is the mom", "between B and C"
        short_person_patterns = [
            r'\b([A-Z])\s+is\b',
            r'\bof\s+([A-Z])\b',
            r'\band\s+([A-Z])\b',
            r'\b([A-Z])\s+and\b',
            r'\b([A-Z])\s+for\b'
        ]
        
        text = doc.text
        for pattern in short_person_patterns:
            for match in re.finditer(pattern, text):
                name = match.group(1)
                entities.append({
                    "label": "Person",
                    "properties": {
                        "name": name,
                        "id": self._generate_id("person", name)
                    }
                })

        return entities

    def _generate_id(self, entity_type: str, name: str) -> str:
        """Generate consistent ID for entity"""
        normalized = name.lower().strip()
        # Remove common punctuation and normalize whitespace
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

    def _is_technical_entity(self, entity_text: str) -> bool:
        """Check if entity text looks like a technical component"""
        tech_indicators = [
            r'\b\w*(?:API|Service|Component|Module|Library|Framework|Database|Cache)\b',
            r'\b[A-Z][a-zA-Z]*[A-Z][a-zA-Z]*\b',  # CamelCase
            r'\b\w+\.(js|py|java|go|ts|jsx|cpp)\b'  # File extensions
        ]
        for pattern in tech_indicators:
            if re.search(pattern, entity_text, re.IGNORECASE):
                return True
        return False

    def _extract_concepts(self, doc) -> List[Dict[str, Any]]:
        """Extract conceptual entities from noun phrases (for research domain)"""
        concepts = []
        seen_concepts = set()

        # Look for noun phrases that might be concepts
        for chunk in doc.noun_chunks:
            # Skip short, common, or stop word phrases
            if len(chunk.text.strip()) < 5:
                continue

            chunk_text = chunk.text.strip()
            chunk_lower = chunk_text.lower()

            # Skip if mostly stop words
            if sum(1 for token in chunk if token.is_stop) > len(chunk) * 0.6:
                continue

            # Skip common generic phrases
            if chunk_lower in ["the study", "the research", "the paper", "the analysis"]:
                continue

            # Check if it looks like a technical/research concept
            if len(chunk) >= 2:
                concept_id = self._generate_id("concept", chunk_text)
                if concept_id not in seen_concepts:
                    seen_concepts.add(concept_id)
                    concepts.append({
                        "label": "Concept",
                        "properties": {
                            "name": chunk_text,
                            "id": concept_id
                        }
                    })

        # Look for capitalized terms that might be concepts
        concept_patterns = [
            r'\b([A-Z][a-zA-Z]+ [A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b',  # Multi-word caps
            r'\b((?:[A-Z][a-zA-Z]*\s+){1,2}(?:Theory|Method|Algorithm|Model|Framework|Approach))\b'
        ]

        for pattern in concept_patterns:
            matches = re.finditer(pattern, doc.text)
            for match in matches:
                concept_text = match.group(1).strip()
                concept_id = self._generate_id("concept", concept_text)
                if concept_id not in seen_concepts and len(concept_text) > 8:
                    seen_concepts.add(concept_id)
                    concepts.append({
                        "label": "Concept",
                        "properties": {
                            "name": concept_text,
                            "id": concept_id
                        }
                    })

        # Limit to avoid noise, prioritize longer concepts
        concepts.sort(key=lambda x: len(x["properties"]["name"]), reverse=True)
        return concepts[:5]

    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities based on ID and filter out common roles"""
        seen_ids = set()
        deduped = []
        
        # Terms that are likely roles or descriptions rather than specific names
        stop_names = {
            "grandmother", "grandfather", "grandchild", "grandson", "granddaughter",
            "mother", "father", "mom", "dad", "parent", "child", "son", "daughter",
            "brother", "sister", "sibling", "uncle", "aunt", "cousin",
            "manager", "director", "ceo", "cto", "employee", "user", "assistant"
        }

        for entity in entities:
            entity_id = entity["properties"]["id"]
            entity_name = entity["properties"].get("name", "").lower()
            
            if entity_id not in seen_ids and entity_name not in stop_names:
                seen_ids.add(entity_id)
                deduped.append(entity)

        return deduped