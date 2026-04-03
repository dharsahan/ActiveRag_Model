"""
Document classification for determining content domains.
Classifies documents into Research, Technical, Business, or Mixed Web domains.
"""

import spacy
from typing import Dict, List, Any
import re
from ..schemas.entities import ContentDomain


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
                           r"\b(dr\.|prof\.|professor)\s+\w+",  # Academic titles
                           r"\b(presented|published|conducted)\s+(by|at|in)"]
            },
            ContentDomain.TECHNICAL: {
                "keywords": ["API", "component", "service", "database", "server", "configuration",
                           "deployment", "code", "function", "class", "library", "framework",
                           "authentication", "endpoint", "REST", "JSON", "HTTP", "cache", "redis",
                           "depends on", "connects to"],
                "patterns": [r"\b(http|https|ftp)://", r"\.(js|py|java|cpp|go|ts|jsx)\b",
                           r"\b\w+(API|Service|Manager|Component|Handler|Controller)\b",
                           r"\b(UserService|DatabaseManager|AuthenticationAPI)\b"]  # Specific tech names
            },
            ContentDomain.BUSINESS: {
                "keywords": ["team", "project", "manager", "process", "workflow", "department",
                           "business", "strategy", "goals", "metrics", "stakeholder", "client",
                           "meeting", "budget", "revenue", "profit", "KPI", "objective", "deliverables",
                           "customer satisfaction", "coordinate with", "reports to"],
                "patterns": [r"\b(Q[1-4]|quarter)\b", r"\b(CEO|CTO|VP|Director|Manager)\b",
                           r"\b\d+%\b",  # Percentages common in business docs
                           r"\b(project manager|team lead|development team)\b"]
            }
        }

    def classify_document(self, text: str) -> ContentDomain:
        """Classify document into one of the four content domains"""
        if not text or len(text.strip()) < 10:
            return ContentDomain.MIXED_WEB

        text_lower = text.lower()
        # Analyze first 1000 characters for performance while maintaining accuracy
        doc = self.nlp(text[:1000])

        # Initialize scores for each domain
        scores = {domain: 0 for domain in ContentDomain}

        # Score based on keywords and patterns with balanced weighting
        for domain, criteria in self.domain_keywords.items():
            # Keyword matching - normalize by counting unique matches instead of frequency
            unique_keyword_matches = sum(1 for keyword in criteria["keywords"]
                                       if keyword.lower() in text_lower)
            scores[domain] += unique_keyword_matches * 2  # Balanced weight for keywords

            # Pattern matching with moderate weight
            pattern_score = sum(len(re.findall(pattern, text, re.IGNORECASE))
                              for pattern in criteria["patterns"])
            scores[domain] += pattern_score * 3  # Higher weight for specific patterns

        # Additional heuristics based on named entities from spaCy
        entities = [(ent.label_, ent.text) for ent in doc.ents]

        for label, entity_text in entities:
            entity_lower = entity_text.lower()

            if label == "ORG":
                # Academic organizations boost research score
                if any(word in entity_lower for word in ["university", "institute", "lab", "college"]):
                    scores[ContentDomain.RESEARCH] += 3
                # Business organizations
                elif any(word in entity_lower for word in ["corp", "inc", "ltd", "company", "llc"]):
                    scores[ContentDomain.BUSINESS] += 2
                # Tech companies
                elif any(word in entity_lower for word in ["tech", "software", "systems"]):
                    scores[ContentDomain.TECHNICAL] += 2

            elif label == "PERSON":
                # Check context around person names for domain hints
                context = self._get_entity_context(entity_text, text, window=100)
                context_lower = context.lower()

                if any(word in context_lower for word in ["dr.", "prof.", "researcher", "scholar"]):
                    scores[ContentDomain.RESEARCH] += 2
                elif any(word in context_lower for word in ["manager", "director", "ceo", "vp"]):
                    scores[ContentDomain.BUSINESS] += 2
                elif any(word in context_lower for word in ["developer", "engineer", "architect"]):
                    scores[ContentDomain.TECHNICAL] += 2

        # Domain-specific linguistic patterns
        scores = self._apply_linguistic_heuristics(doc, scores)

        # Return domain with highest score, default to MIXED_WEB if no clear winner
        max_domain = max(scores, key=scores.get)
        max_score = scores[max_domain]

        # Require minimum threshold and check for ties
        if max_score < 3:  # Increased threshold for more confident classification
            return ContentDomain.MIXED_WEB

        # Check if there's a significant margin over other domains
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] - sorted_scores[1] < 2:
            # Too close to call confidently
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

        # Technical documents often have more technical vocabulary
        tech_pos_tags = sum(1 for token in doc if token.pos_ in ["SYM", "X"])  # Symbols, technical terms
        if tech_pos_tags > 3:
            scores[ContentDomain.TECHNICAL] += 1

        # Research documents often have more complex sentence structures
        avg_sent_length = sum(len(sent) for sent in doc.sents) / max(len(list(doc.sents)), 1)
        if avg_sent_length > 20:  # Longer sentences typical in academic writing
            scores[ContentDomain.RESEARCH] += 1

        # Business documents often mention future tense (plans, goals)
        future_indicators = sum(1 for token in doc if token.lemma_ in ["will", "plan", "goal", "target"])
        if future_indicators > 2:
            scores[ContentDomain.BUSINESS] += 1

        return scores