"""
Domain-specific entity extraction for the hybrid RAG system.
Extracts entities based on content domain classification.
Enhanced with Entity Disambiguation (Fuzzy Matching), Keyword Extraction, and Sentiment Analysis.
"""

import spacy
import hashlib
import re
import logging
from typing import Dict, List, Any, Tuple, Optional
from rapidfuzz import process, fuzz
from ..schemas.entities import ContentDomain, ENTITY_SCHEMAS

logger = logging.getLogger(__name__)

class EntityExtractor:
    """Extracts and disambiguates entities from text based on content domain"""

    def __init__(self):
        """Initialize the entity extractor with spaCy model and disambiguation maps"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise ValueError("spaCy English model not found. Run: python -m spacy download en_core_web_sm")
        
        # Disambiguation / Alias Map
        self._alias_map = {
            "google": "Google",
            "alphabet inc": "Google",
            "alphabet": "Google",
            "microsoft corp": "Microsoft",
            "msft": "Microsoft",
            "apple inc": "Apple",
            "amazon.com": "Amazon",
            "meta platforms": "Meta",
            "facebook": "Meta",
            "openai": "OpenAI",
            "anthropic": "Anthropic",
            "nvidia": "NVIDIA",
        }
        
        # Cache for known entity IDs to ensure consistency in a single run
        self._known_entities = {} # (label, name) -> id

    def extract_entities(self, text: str, domain: ContentDomain) -> List[Dict[str, Any]]:
        """Extract entities from text based on content domain"""
        if not text or len(text.strip()) < 5:
            return []

        doc = self.nlp(text)
        entities = []

        # 1. Domain-specific extraction
        if domain == ContentDomain.RESEARCH:
            entities.extend(self._extract_research_entities(doc))
        elif domain == ContentDomain.TECHNICAL:
            entities.extend(self._extract_technical_entities(doc))
        elif domain == ContentDomain.BUSINESS:
            entities.extend(self._extract_business_entities(doc))
        else:  # MIXED_WEB
            entities.extend(self._extract_general_entities(doc))

        # 2. Extract Keywords / Topics
        keywords = self._extract_keywords(doc)
        entities.extend(keywords)

        # 3. Disambiguate and Deduplicate
        return self._disambiguate_and_deduplicate(entities)

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Perform basic sentiment analysis using spaCy and simple heuristics"""
        doc = self.nlp(text)
        
        pos_words = {"great", "excellent", "good", "happy", "success", "resolved", "fixed", "improving", "fast", "efficient"}
        neg_words = {"bad", "error", "fail", "slow", "broken", "issue", "bug", "crash", "hallucination", "incorrect"}
        
        pos_count = sum(1 for token in doc if token.text.lower() in pos_words)
        neg_count = sum(1 for token in doc if token.text.lower() in neg_words)
        
        score = (pos_count - neg_count) / max(1, pos_count + neg_count)
        
        sentiment = "neutral"
        if score > 0.2: sentiment = "positive"
        elif score < -0.2: sentiment = "negative"
        
        return {
            "score": round(score, 2),
            "label": sentiment,
            "counts": {"positive": pos_count, "negative": neg_count}
        }

    def _extract_research_entities(self, doc) -> List[Dict[str, Any]]:
        """Extract research-specific entities (Person, Organization, Concept)"""
        entities = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                context = self._get_entity_context(ent, doc)
                if self._is_likely_researcher(context):
                    entities.append({"label": "Person", "properties": {"name": ent.text.strip()}})
            elif ent.label_ == "ORG":
                if self._is_academic_org(ent.text):
                    entities.append({"label": "Organization", "properties": {"name": ent.text.strip(), "type": "academic"}})
        
        entities.extend(self._extract_concepts(doc))
        return entities

    def _extract_technical_entities(self, doc) -> List[Dict[str, Any]]:
        """Extract technical/API entities (Component)"""
        entities = []
        component_patterns = [
            r'\b([A-Z][a-zA-Z]*(?:API|Service|Component|Module|Library|Manager|Handler|Controller))\b',
            r'\b([a-z]+(?:_[a-z]+)*\.(?:js|py|java|cpp|go|ts|jsx))\b',
            r'\b(Redis|MongoDB|PostgreSQL|MySQL|ElasticSearch|Kafka|Docker|Kubernetes|Neo4j|ChromaDB)\b'
        ]
        
        text = doc.text
        for pattern in component_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                comp_name = match.group(1)
                if len(comp_name) > 2 and comp_name.lower() not in ['the', 'and', 'for']:
                    entities.append({"label": "Component", "properties": {"name": comp_name}})
        
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT", "ORG"] and self._is_technical_entity(ent.text):
                entities.append({"label": "Component", "properties": {"name": ent.text.strip()}})
        return entities

    def _extract_business_entities(self, doc) -> List[Dict[str, Any]]:
        """Extract business process and people entities"""
        entities = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                context = self._get_entity_context(ent, doc)
                if self._is_business_person(context):
                    entities.append({"label": "Person", "properties": {"name": ent.text.strip()}})
            elif ent.label_ == "ORG":
                entities.append({"label": "Organization", "properties": {"name": ent.text.strip(), "type": "business"}})
        
        process_patterns = [r'\b([A-Z][a-zA-Z\s]{3,30} (?:process|workflow|procedure|methodology))\b']
        for pattern in process_patterns:
            for match in re.finditer(pattern, doc.text, re.IGNORECASE):
                entities.append({"label": "Process", "properties": {"name": match.group(1).strip()}})
        return entities

    def _extract_general_entities(self, doc) -> List[Dict[str, Any]]:
        """Extract general entities with short-name support"""
        entities = []
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG"]:
                entities.append({"label": "Person" if ent.label_ == "PERSON" else "Organization", 
                                "properties": {"name": ent.text.strip()}})
        
        # Regex fallback for single-letter names (A, B, C)
        short_patterns = [r'\b([A-Z])\s+is\b', r'\bof\s+([A-Z])\b', r'\band\s+([A-Z])\b', r'\b([A-Z])\s+and\b']
        for pattern in short_patterns:
            for match in re.finditer(pattern, doc.text):
                entities.append({"label": "Person", "properties": {"name": match.group(1)}})
        return entities

    def _extract_keywords(self, doc) -> List[Dict[str, Any]]:
        """Extract top 3 key phrases as Topic entities"""
        keywords = []
        candidates = []
        for chunk in doc.noun_chunks:
            if not chunk.root.is_stop and len(chunk.text) > 4:
                candidates.append(chunk.text.strip())
        
        # Frequency count
        from collections import Counter
        counts = Counter(candidates)
        top_topics = counts.most_common(3)
        
        for topic, _ in top_topics:
            keywords.append({"label": "Topic", "properties": {"name": topic}})
        return keywords

    def _disambiguate_and_deduplicate(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resolve aliases and perform fuzzy matching to deduplicate entities."""
        processed = []
        seen_names_by_label = {} # label -> list of canonical names

        for ent in entities:
            label = ent["label"]
            name = ent["properties"]["name"].strip()
            name_lower = name.lower()
            
            # 1. Alias Resolution
            canonical_name = name
            for alias, target in self._alias_map.items():
                if alias in name_lower:
                    canonical_name = target
                    break
            
            # 2. Fuzzy Matching Disambiguation
            if label not in seen_names_by_label:
                seen_names_by_label[label] = []
            
            # Check against previously seen names for this label
            match = None
            if seen_names_by_label[label] and len(canonical_name) > 3:
                match = process.extractOne(canonical_name, seen_names_by_label[label], 
                                         scorer=fuzz.WRatio, score_cutoff=90)
            
            if match:
                # Use the existing canonical name
                canonical_name = match[0]
            else:
                seen_names_by_label[label].append(canonical_name)

            # 3. ID Generation
            ent_id = self._generate_id(label.lower(), canonical_name)
            ent["properties"]["name"] = canonical_name
            ent["properties"]["id"] = ent_id
            
            # Stop-word / Role filtering
            stop_names = {"grandmother", "grandfather", "mom", "dad", "mother", "father", "child", "user", "assistant"}
            if canonical_name.lower() in stop_names and label == "Person":
                continue

            # Final deduplication by ID
            if not any(p["properties"]["id"] == ent_id for p in processed):
                processed.append(ent)

        return processed

    def _generate_id(self, entity_type: str, name: str) -> str:
        normalized = name.lower().strip()
        normalized = re.sub(r'[^\w\s-]', '', normalized)
        normalized = re.sub(r'\s+', '_', normalized)
        hash_obj = hashlib.md5(f"{entity_type}_{normalized}".encode())
        return f"{entity_type}_{hash_obj.hexdigest()[:8]}"

    def _get_entity_context(self, ent, doc, window: int = 50) -> str:
        start = max(0, ent.start - window)
        end = min(len(doc), ent.end + window)
        return doc[start:end].text

    def _is_likely_researcher(self, context: str) -> bool:
        indicators = ["professor", "dr.", "phd", "university", "published", "scientist"]
        return any(ind in context.lower() for ind in indicators)

    def _is_academic_org(self, org_name: str) -> bool:
        indicators = ["university", "institute", "college", "academy", "school"]
        return any(ind in org_name.lower() for ind in indicators)

    def _is_business_person(self, context: str) -> bool:
        indicators = ["ceo", "director", "manager", "lead", "executive"]
        return any(ind in context.lower() for ind in indicators)

    def _is_technical_entity(self, entity_text: str) -> bool:
        indicators = [r'API|Service|Library|Database|Framework', r'\.(js|py|java|go)']
        return any(re.search(ind, entity_text, re.IGNORECASE) for ind in indicators)

    def _extract_concepts(self, doc) -> List[Dict[str, Any]]:
        concepts = []
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 8 and not chunk.root.is_stop:
                concepts.append({"label": "Concept", "properties": {"name": chunk.text.strip()}})
        concepts.sort(key=lambda x: len(x["properties"]["name"]), reverse=True)
        return concepts[:3]
