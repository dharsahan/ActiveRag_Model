"""Query classifier for hybrid vector-graph RAG routing.

Classifies queries by intent (semantic, relational, hybrid) and complexity
(simple single-hop vs. multi-hop reasoning) using fast keyword heuristics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List


class QueryIntent(Enum):
    """What type of information the query seeks."""
    SEMANTIC = "semantic"        # Similarity-based: "What is quantum computing?"
    RELATIONAL = "relational"    # Entity connections: "Who manages the ML team?"
    HYBRID = "hybrid"            # Both: "Which papers by MIT researchers cite Einstein?"


class QueryComplexity(Enum):
    """How many reasoning steps the query needs."""
    SIMPLE = "simple"            # Single-hop lookup
    MULTI_HOP = "multi_hop"      # Requires graph traversal across multiple nodes


@dataclass
class ClassificationResult:
    """Result of query classification."""
    intent: QueryIntent
    complexity: QueryComplexity
    detected_entities: List[str]
    relational_signals: List[str]
    confidence: float  # 0.0–1.0 how confident this classification is


# --- Keyword / Pattern Detection ---

_RELATIONAL_KEYWORDS = [
    "who", "whom", "which person", "which team",
    "manages", "leads", "reports to", "works for", "works with",
    "authored", "published by", "written by", "created by",
    "depends on", "connected to", "related to", "associated with",
    "collaborated", "co-authored", "affiliated",
    "relationship", "connection", "link between",
    "cite", "cites", "cited by", "references",
]

_MULTI_HOP_PATTERNS = [
    r"\b(who|which)\b.*\b(that|who)\b",                # "who X that Y"
    r"\b(students|colleagues|collaborators)\b.*\bof\b",  # "students of X"
    r"\b(through|via|chain|path)\b",
    r"\b(indirectly|transitively)\b",
    r"\bhow\s+(?:is|are)\s+\w+\s+(?:connected|related)\b",
]

_SEMANTIC_KEYWORDS = [
    "what is", "what are", "explain", "describe", "define",
    "how does", "how to", "why does", "why is",
    "tell me about", "summarize", "overview",
    "meaning of", "difference between",
]

_ENTITY_PATTERNS = [
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",  # Proper names like "Albert Einstein"
    r"\b(?:MIT|NASA|IBM|Google|OpenAI|NVIDIA)\b",  # Known orgs
]


class QueryClassifier:
    """Classifies queries for hybrid retrieval routing."""

    def classify(self, query: str) -> ClassificationResult:
        """Classify a query by intent and complexity.

        Args:
            query: The user's natural language question.

        Returns:
            ClassificationResult with intent, complexity, and signals.
        """
        query_lower = query.lower().strip()

        # Detect signals
        relational_signals = self._detect_relational_signals(query_lower)
        semantic_signals = self._detect_semantic_signals(query_lower)
        entities = self._detect_entities(query)
        is_multi_hop = self._detect_multi_hop(query_lower)

        rel_score = len(relational_signals)
        sem_score = len(semantic_signals)

        # Determine intent
        if rel_score > 0 and sem_score > 0:
            intent = QueryIntent.HYBRID
        elif rel_score > 0:
            intent = QueryIntent.RELATIONAL
        else:
            intent = QueryIntent.SEMANTIC

        # Boost toward HYBRID if multiple entities are detected
        if len(entities) >= 2 and intent == QueryIntent.SEMANTIC:
            intent = QueryIntent.HYBRID

        # Determine complexity
        complexity = (
            QueryComplexity.MULTI_HOP
            if is_multi_hop or (rel_score >= 2 and len(entities) >= 2)
            else QueryComplexity.SIMPLE
        )

        # Confidence: higher when signals are clear and unambiguous
        total_signals = rel_score + sem_score
        confidence = min(1.0, 0.5 + 0.1 * total_signals) if total_signals > 0 else 0.4

        return ClassificationResult(
            intent=intent,
            complexity=complexity,
            detected_entities=entities,
            relational_signals=relational_signals,
            confidence=confidence,
        )

    # --- Private helpers ---

    def _detect_relational_signals(self, query_lower: str) -> List[str]:
        """Find relational keywords in the query."""
        return [kw for kw in _RELATIONAL_KEYWORDS if kw in query_lower]

    def _detect_semantic_signals(self, query_lower: str) -> List[str]:
        """Find semantic keywords in the query."""
        return [kw for kw in _SEMANTIC_KEYWORDS if kw in query_lower]

    def _detect_entities(self, query: str) -> List[str]:
        """Extract likely entity names from the query."""
        entities: List[str] = []
        for pattern in _ENTITY_PATTERNS:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: List[str] = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                deduped.append(e)
        return deduped

    def _detect_multi_hop(self, query_lower: str) -> bool:
        """Check if the query requires multi-hop reasoning."""
        return any(re.search(p, query_lower) for p in _MULTI_HOP_PATTERNS)
