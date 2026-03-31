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
from dataclasses import dataclass, field

import chromadb

from active_rag.config import Config


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
    """Thin wrapper around ChromaDB for the Active RAG pipeline."""

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
        self._collection = self._client.get_or_create_collection(**create_kwargs)

    # ------------------------------------------------------------------
    # Check Vector Memory / RAG
    # ------------------------------------------------------------------
    def search(
        self, query: str, min_score: float | None = None, return_all: bool = False,
        max_age_seconds: float | None = None,
    ) -> VectorSearchResult:
        """Search the vector store for documents relevant to *query*.

        Returns a ``VectorSearchResult`` indicating whether data was found
        and, if so, the matching documents with scores.
        
        Args:
            query: The search query
            min_score: Minimum similarity score (default: config threshold or 0.2)
            return_all: If True, return all results regardless of score
        """
        if self._collection.count() == 0:
            return VectorSearchResult(found=False)

        results = self._collection.query(
            query_texts=[query],
            n_results=min(self._config.top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        items: list[RetrievalResult] = []
        if results and results["documents"]:
            documents = results["documents"][0]
            metadatas = (results["metadatas"] or [[]])[0]
            distances = (results["distances"] or [[]])[0]
            now = time.time()
            for doc, meta, dist in zip(documents, metadatas, distances):
                # Filter by age if max_age_seconds is set
                if max_age_seconds is not None and meta:
                    indexed_at = meta.get("indexed_at")
                    if indexed_at is None or (now - float(indexed_at)) > max_age_seconds:
                        continue  # Skip stale or un-timestamped documents

                # ChromaDB default uses L2 distance. For the default embedding
                # model (all-MiniLM-L6-v2), L2 distances typically range 0-2.
                # Convert to similarity: closer to 0 = more similar.
                # Using normalized formula: 1 / (1 + dist)
                similarity = 1.0 / (1.0 + dist)
                items.append(
                    RetrievalResult(
                        content=doc,
                        source_url=meta.get("source_url", "") if meta else "",
                        score=similarity,
                    )
                )

        # Sort by score descending
        items.sort(key=lambda r: r.score, reverse=True)
        
        # Determine threshold
        threshold = min_score if min_score is not None else 0.2
        
        # Consider data "found" if there is at least one result with a
        # reasonable similarity score.
        found = any(r.score >= threshold for r in items)
        
        if return_all:
            # Return all results, let caller decide what to use
            return VectorSearchResult(found=found or len(items) > 0, results=items)
        
        # Filter to only high-quality results
        filtered = [r for r in items if r.score >= threshold]
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
