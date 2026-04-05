"""Vector Memory / RAG – Neo4j-backed vector store.

Replaces ChromaDB with Neo4j's native vector index for a pure GraphRAG architecture.
"""

from __future__ import annotations

import hashlib
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from active_rag.config import Config
from active_rag.knowledge_graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieved document with its source citation."""
    content: str
    source_url: str
    score: float


@dataclass
class VectorSearchResult:
    """Result of a vector memory search."""
    found: bool
    results: list[RetrievalResult] = field(default_factory=list)


class VectorStore:
    """Neo4j-backed vector store for the Active RAG pipeline."""

    def __init__(
        self,
        config: Config,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._config = config
        
        # Initialize Neo4j client
        self._neo4j = Neo4jClient(
            config.neo4j_uri,
            config.neo4j_username,
            config.neo4j_password
        )
        
        # Initialize Embedding Model
        try:
            self._embedder = SentenceTransformer(embedding_model_name)
            self._embedding_dimension = 384 # Dimension for all-MiniLM-L6-v2
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {e}")
            raise

        # Initialize CrossEncoder for re-ranking
        try:
            self._reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
        except Exception as e:
            logger.warning(f"Failed to load CrossEncoder: {e}. Re-ranking will be disabled.")
            self._reranker = None

        # Setup Vector Index in Neo4j
        self._setup_vector_index()

    def _setup_vector_index(self) -> None:
        """Create the vector index in Neo4j if it doesn't exist."""
        index_name = f"vector_index_{self._config.vector_index_name}"
        
        # Cypher for creating vector index (Neo4j 5.15+)
        # We store chunks as 'Chunk' nodes with 'embedding' property
        create_index_query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (n:Chunk) ON (n.embedding)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {self._embedding_dimension},
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """
        
        try:
            with self._neo4j._driver.session() as session:
                session.run(create_index_query)
                logger.info(f"Neo4j vector index '{index_name}' verified.")
        except Exception as e:
            logger.error(f"Failed to create Neo4j vector index: {e}")

    def search(
        self, query: str, min_score: float | None = None, return_all: bool = False,
        max_age_seconds: float | None = None,
    ) -> VectorSearchResult:
        """Search Neo4j for documents relevant to *query*."""
        
        # 1. Generate Query Embedding
        query_vector = self._embedder.encode(query).tolist()
        index_name = f"vector_index_{self._config.vector_index_name}"
        
        # 2. Vector Search using Cypher
        # We fetch more for re-ranking (3x top_k)
        limit = self._config.top_k * 3
        
        search_query = f"""
        CALL db.index.vector.queryNodes('{index_name}', $limit, $query_vector)
        YIELD node, score
        RETURN node.content AS content, node.source_url AS source_url, 
               node.indexed_at AS indexed_at, score
        """
        
        items = []
        now = time.time()
        
        try:
            with self._neo4j._driver.session() as session:
                result = session.run(search_query, limit=limit, query_vector=query_vector)
                
                for record in result:
                    # Filter by age if needed
                    if max_age_seconds is not None:
                        indexed_at = record["indexed_at"]
                        if indexed_at is None or (now - float(indexed_at)) > max_age_seconds:
                            continue
                            
                    items.append({
                        "content": record["content"],
                        "source_url": record["source_url"] or "",
                        "score": float(record["score"])
                    })
        except Exception as e:
            logger.error(f"Neo4j vector search failed: {e}")
            return VectorSearchResult(found=False)

        if not items:
            return VectorSearchResult(found=False)

        # 3. Hybrid BM25 Scoring (Optional within the pool)
        tokenized_query = query.lower().split()
        tokenized_corpus = [item["content"].lower().split() for item in items]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(tokenized_query)
        
        for i, item in enumerate(items):
            # Combine scores (70% vector, 30% BM25)
            bm25_sim = min(bm25_scores[i] / 20.0, 1.0) if bm25_scores[i] > 0 else 0
            item["hybrid_score"] = (item["score"] * 0.7) + (bm25_sim * 0.3)

        # 4. Neural Re-ranking
        if self._reranker and len(items) > 1:
            pairs = [[query, item["content"]] for item in items]
            rerank_scores = self._reranker.predict(pairs)
            for i, score in enumerate(rerank_scores):
                items[i]["rerank_score"] = float(score)
            items.sort(key=lambda x: x["rerank_score"], reverse=True)
        else:
            items.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # Determine threshold
        threshold = min_score if min_score is not None else (-3.0 if self._reranker else 0.2)
        
        final_results = [
            RetrievalResult(
                content=item["content"],
                source_url=item["source_url"],
                score=item.get("rerank_score", item["hybrid_score"])
            )
            for item in items
        ]
        
        if return_all:
            return VectorSearchResult(found=len(final_results) > 0, results=final_results)
            
        filtered = [r for r in final_results if r.score >= threshold]
        filtered = filtered[:self._config.top_k]
        
        return VectorSearchResult(found=len(filtered) > 0, results=filtered)

    def add_documents(
        self,
        contents: list[str],
        source_urls: list[str],
    ) -> list[str]:
        """Index new documents into Neo4j as Chunk nodes. Returns created IDs."""
        if not contents:
            return []

        # Generate embeddings
        embeddings = self._embedder.encode(contents).tolist()
        now = time.time()
        
        # Prepare data for UNWIND query
        data = []
        ids = []
        for i, content in enumerate(contents):
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            chunk_id = f"chunk-{content_hash}-{uuid.uuid4().hex[:8]}"
            ids.append(chunk_id)
            data.append({
                "id": chunk_id,
                "content": content,
                "source_url": source_urls[i] if i < len(source_urls) else "",
                "embedding": embeddings[i],
                "indexed_at": now
            })

        # Efficient batch insert using UNWIND
        insert_query = """
        UNWIND $data AS row
        MERGE (c:Chunk {id: row.id})
        SET c.content = row.content,
            c.source_url = row.source_url,
            c.embedding = row.embedding,
            c.indexed_at = row.indexed_at
        """
        
        try:
            with self._neo4j._driver.session() as session:
                session.run(insert_query, data=data)
                logger.info(f"Indexed {len(contents)} chunks into Neo4j.")
            return ids
        except Exception as e:
            logger.error(f"Failed to add documents to Neo4j: {e}")
            return []

    def count(self) -> int:
        """Return the number of chunks in the graph."""
        try:
            with self._neo4j._driver.session() as session:
                result = session.run("MATCH (n:Chunk) RETURN count(n) AS count")
                return result.single()["count"]
        except Exception:
            return 0

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retrieve all documents and their metadata from the graph."""
        query = "MATCH (n:Chunk) RETURN n.content AS content, n.source_url AS source_url"
        try:
            with self._neo4j._driver.session() as session:
                result = session.run(query)
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Failed to retrieve all documents: {e}")
            return []
