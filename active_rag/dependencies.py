"""Shared dependency injection for the Active RAG API.

Provides singleton resource managers that are shared across all routers.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from fastapi import Header, HTTPException

from active_rag.config import Config
from active_rag.agent import AgenticOrchestrator
from active_rag.memory import ConversationMemory
from active_rag.vector_store import VectorStore
from active_rag.document_loader import DocumentLoader

logger = logging.getLogger(__name__)


# --- Session Management ---

class SessionManager:
    """Manages conversation memories for multiple sessions."""

    def __init__(self, config: Config):
        self.config = config
        self._memories: Dict[str, ConversationMemory] = {}

    def get_memory(self, session_id: str) -> ConversationMemory:
        if session_id not in self._memories:
            self._memories[session_id] = ConversationMemory(self.config)
        return self._memories[session_id]

    def clear_session(self, session_id: str):
        if session_id in self._memories:
            self._memories[session_id].clear()

    def list_sessions(self) -> List[str]:
        return list(self._memories.keys())


# --- Pipeline / Resource Management ---

class ResourceManager:
    """Holds shared instances of heavy RAG components."""

    def __init__(self, config: Config):
        self.config = config
        self._pipelines = {}
        self._vector_store = None
        self._document_loader = None

    def get_pipeline(self, pipeline_type: str):
        if pipeline_type not in self._pipelines:
            if pipeline_type == "agent":
                self._pipelines["agent"] = AgenticOrchestrator(self.config)
            elif pipeline_type == "hybrid":
                from active_rag.hybrid_pipeline import HybridRAGPipeline
                self._pipelines["hybrid"] = HybridRAGPipeline(self.config)
            elif pipeline_type == "ultimate":
                from active_rag.ultimate_pipeline import UltimateActiveRAGPipeline
                self._pipelines["ultimate"] = UltimateActiveRAGPipeline(self.config)
            elif pipeline_type == "legacy":
                from active_rag.pipeline import ActiveRAGPipeline
                self._pipelines["legacy"] = ActiveRAGPipeline(self.config)
        return self._pipelines[pipeline_type]

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore(self.config)
        return self._vector_store

    @property
    def document_loader(self) -> DocumentLoader:
        if self._document_loader is None:
            self._document_loader = DocumentLoader(self.config)
        return self._document_loader


# --- Graph & NLP Resource Management ---

class GraphResourceManager:
    """Lazy-initialized graph, NLP, reasoning, and analytics components."""

    def __init__(self, config: Config):
        self.config = config
        self._graph_ops = None
        self._entity_extractor = None
        self._relation_extractor = None
        self._document_classifier = None
        self._reasoning_engine = None
        self._community_detector = None
        self._cross_domain = None
        self._evaluator = None
        self._graph_cache = None
        self._query_monitor = None
        self._initialized = False

    def _ensure_init(self):
        """Lazy-init all graph-dependent components once."""
        if self._initialized:
            return
        self._initialized = True

        # NLP components (no graph dependency)
        try:
            from active_rag.nlp_pipeline.entity_extractor import EntityExtractor
            self._entity_extractor = EntityExtractor()
        except Exception as e:
            logger.warning(f"EntityExtractor unavailable: {e}")

        try:
            from active_rag.nlp_pipeline.relation_extractor import RelationExtractor
            self._relation_extractor = RelationExtractor(self.config)
        except Exception as e:
            logger.warning(f"RelationExtractor unavailable: {e}")

        try:
            from active_rag.nlp_pipeline.document_classifier import DocumentClassifier
            self._document_classifier = DocumentClassifier()
        except Exception as e:
            logger.warning(f"DocumentClassifier unavailable: {e}")

        # Graph components
        if self.config.enable_graph_features:
            try:
                from active_rag.knowledge_graph.neo4j_client import Neo4jClient
                from active_rag.knowledge_graph.graph_operations import GraphOperations

                client = Neo4jClient(
                    self.config.neo4j_uri,
                    self.config.neo4j_username,
                    self.config.neo4j_password,
                )
                self._graph_ops = GraphOperations(client)
            except Exception as e:
                logger.warning(f"Graph backend unavailable: {e}")

        # Reasoning components (depend on graph)
        if self._graph_ops:
            try:
                from active_rag.reasoning.reasoning_engine import ReasoningEngine
                self._reasoning_engine = ReasoningEngine(self._graph_ops, self._entity_extractor)
            except Exception as e:
                logger.warning(f"ReasoningEngine unavailable: {e}")

            try:
                from active_rag.reasoning.community_detection import CommunityDetector
                self._community_detector = CommunityDetector()
            except Exception as e:
                logger.warning(f"CommunityDetector unavailable: {e}")

            try:
                from active_rag.reasoning.cross_domain import CrossDomainDiscovery
                self._cross_domain = CrossDomainDiscovery()
            except Exception as e:
                logger.warning(f"CrossDomainDiscovery unavailable: {e}")

        # Evaluator
        try:
            from active_rag.evaluator import AnswerEvaluator
            self._evaluator = AnswerEvaluator(self.config)
        except Exception as e:
            logger.warning(f"AnswerEvaluator unavailable: {e}")

        # Cache & monitor
        try:
            from active_rag.knowledge_graph.graph_cache import GraphCache
            self._graph_cache = GraphCache()
        except Exception as e:
            logger.warning(f"GraphCache unavailable: {e}")

        try:
            from active_rag.knowledge_graph.query_monitor import QueryMonitor
            self._query_monitor = QueryMonitor()
        except Exception as e:
            logger.warning(f"QueryMonitor unavailable: {e}")

    @property
    def graph_ops(self):
        self._ensure_init()
        return self._graph_ops

    @property
    def entity_extractor(self):
        self._ensure_init()
        return self._entity_extractor

    @property
    def relation_extractor(self):
        self._ensure_init()
        return self._relation_extractor

    @property
    def document_classifier(self):
        self._ensure_init()
        return self._document_classifier

    @property
    def reasoning_engine(self):
        self._ensure_init()
        return self._reasoning_engine

    @property
    def community_detector(self):
        self._ensure_init()
        return self._community_detector

    @property
    def cross_domain(self):
        self._ensure_init()
        return self._cross_domain

    @property
    def evaluator(self):
        self._ensure_init()
        return self._evaluator

    @property
    def graph_cache(self):
        self._ensure_init()
        return self._graph_cache

    @property
    def query_monitor(self):
        self._ensure_init()
        return self._query_monitor


# --- Optional API Key Auth ---

def verify_api_key(x_api_key: str = Header(None)):
    """Optional API key validation. Set ACTIVE_RAG_API_KEY env var to enable."""
    required_key = os.getenv("ACTIVE_RAG_API_KEY")
    if required_key and x_api_key != required_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key
