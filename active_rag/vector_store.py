"""Vector Memory / RAG – ChromaDB-backed vector store.

Handles:
* Checking whether relevant data exists (Check Vector Memory / RAG)
* Retrieving context & citations (Retrieve Context & Citations)
* Updating the vector DB with new content (Update Vector DB)
"""

from __future__ import annotations

import hashlib
import time
import uuid
import logging
from dataclasses import dataclass, field

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from active_rag.config import Config

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
    """Thin wrapper around ChromaDB for the Active RAG pipeline, enhanced with Hybrid Search."""

    def __init__(
        self,
        config: Config,
        embedding_function: chromadb.EmbeddingFunction | None = None,
    ) -> None:
        self._config = config
        persist_dir = config.chroma_persist_dir
        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.EphemeralClient()

        create_kwargs: dict = {"name": config.collection_name}
        if embedding_function is not None:
            create_kwargs["embedding_function"] = embedding_function
        
        # Use cosine similarity for better normalized scores
        self._collection = self._client.get_or_create_collection(
            **create_kwargs, 
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize CrossEncoder for re-ranking (lightweight model)
        try:
            # This will download the model on first run
            self._reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
        except Exception as e:
            logger.warning(f"Failed to load CrossEncoder: {e}. Re-ranking will be disabled.")
            self._reranker = None

    # ------------------------------------------------------------------
    # Check Vector Memory / RAG
    # ------------------------------------------------------------------
    def search(
        self, query: str, min_score: float | None = None, return_all: bool = False,
        max_age_seconds: float | None = None,
    ) -> VectorSearchResult:
        """Search the vector store for documents relevant to *query*.

        Uses Hybrid Search (Vector + BM25) and Neural Re-ranking.
        """
        if self._collection.count() == 0:
            return VectorSearchResult(found=False)

        # 1. Vector Search (retrieving more for re-ranking)
        # We fetch 3x the requested top_k to provide a good pool for re-ranking
        search_limit = min(self._config.top_k * 3, self._collection.count())
        
        results = self._collection.query(
            query_texts=[query],
            n_results=search_limit,
            include=["documents", "metadatas", "distances"],
        )

        if not results or not results["documents"] or not results["documents"][0]:
            return VectorSearchResult(found=False)

        documents = results["documents"][0]
        metadatas = (results["metadatas"] or [[]])[0]
        distances = (results["distances"] or [[]])[0]
        
        # 2. BM25 Scoring (within the vector results)
        tokenized_query = query.lower().split()
        tokenized_corpus = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Combine into initial list
        items: list[dict] = []
        now = time.time()
        
        for i in range(len(documents)):
            doc = documents[i]
            meta = metadatas[i]
            dist = distances[i]
            
            # Filter by age if max_age_seconds is set
            if max_age_seconds is not None and meta:
                indexed_at = meta.get("indexed_at")
                if indexed_at is None or (now - float(indexed_at)) > max_age_seconds:
                    continue

            # Since we switched to cosine, dist is 0 (identical) to 1 (orthogonal)
            # Similarity = 1 - dist
            vector_sim = max(0, 1.0 - dist)
            
            # Normalize BM25 (rough heuristic)
            bm25_sim = min(bm25_scores[i] / 20.0, 1.0) if bm25_scores[i] > 0 else 0
            
            # Initial hybrid score (70% vector, 30% BM25)
            hybrid_score = (vector_sim * 0.7) + (bm25_sim * 0.3)
            
            items.append({
                "content": doc,
                "source_url": meta.get("source_url", "") if meta else "",
                "score": hybrid_score
            })

        if not items:
            return VectorSearchResult(found=False)

        # 3. Neural Re-ranking (Cross-Encoder)
        if self._reranker and len(items) > 1:
            pairs = [[query, item["content"]] for item in items]
            rerank_scores = self._reranker.predict(pairs)
            
            for i, score in enumerate(rerank_scores):
                # Cross-encoder scores can be outside 0-1, so we sigmoid them roughly
                # Or just use the raw score for sorting
                items[i]["rerank_score"] = float(score)
            
            # Sort by re-rank score
            items.sort(key=lambda x: x["rerank_score"], reverse=True)
        else:
            # Fallback to hybrid score
            items.sort(key=lambda x: x["score"], reverse=True)

        # Map to RetrievalResult
        final_results = [
            RetrievalResult(
                content=item["content"],
                source_url=item["source_url"],
                score=item.get("rerank_score", item["score"])
            )
            for item in items
        ]
        
        # Determine threshold (Cross-encoder scores are different, but we'll use a heuristic)
        threshold = min_score if min_score is not None else 0.2
        if self._reranker:
            # Cross-encoder threshold is typically lower or negative
            threshold = -3.0 if min_score is None else min_score
            
        found = any(r.score >= threshold for r in final_results)
        
        if return_all:
            return VectorSearchResult(found=found or len(final_results) > 0, results=final_results)
        
        # Filter to only high-quality results
        filtered = [r for r in final_results if r.score >= threshold]
        # Truncate to the original requested top_k
        filtered = filtered[:self._config.top_k]
        
        return VectorSearchResult(found=len(filtered) > 0, results=filtered)

    # ------------------------------------------------------------------
    # Update Vector DB (Content + Source URL)
    # ------------------------------------------------------------------
    def add_documents(
        self,
        contents: list[str],
        source_urls: list[str],
    ) -> None:
        """Index new documents into the vector store."""
        if not contents:
            return

        ids: list[str] = []
        for content in contents:
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            ids.append(f"doc-{content_hash}-{uuid.uuid4().hex[:8]}")

        now = time.time()
        self._collection.add(
            documents=contents,
            metadatas=[{"source_url": url, "indexed_at": now} for url in source_urls],
            ids=ids,
        )
