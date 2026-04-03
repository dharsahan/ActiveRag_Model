"""
NLP Pipeline for Domain-specific Entity Extraction and Document Classification.

This module provides:
- DocumentClassifier: Classifies documents into content domains
- EntityExtractor: Extracts entities based on document domain

Supports four content domains:
- RESEARCH: Academic papers and scientific content
- TECHNICAL: Code, API docs, and technical documentation
- BUSINESS: Process docs, team communications, business content
- MIXED_WEB: General web content and mixed sources
"""

from .entity_extractor import EntityExtractor
from .document_classifier import DocumentClassifier

__all__ = ['EntityExtractor', 'DocumentClassifier']

# Version info
__version__ = '1.0.0'
__author__ = 'Active RAG System'
__description__ = 'Domain-specific NLP pipeline for entity extraction'