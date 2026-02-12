"""Vector Memory / RAG – ChromaDB-backed vector store.

Handles:
* Checking whether relevant data exists (Check Vector Memory / RAG)
* Retrieving context & citations (Retrieve Context & Citations)
* Updating the vector DB with new content (Update Vector DB)
"""

from __future__ import annotations

import hashlib
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
    def search(self, query: str) -> VectorSearchResult:
        """Search the vector store for documents relevant to *query*.

        Returns a ``VectorSearchResult`` indicating whether data was found
        and, if so, the matching documents with scores.
        """
        if self._collection.count() == 0:
            return VectorSearchResult(found=False)

        results = self._collection.query(
            query_texts=[query],
            n_results=min(self._config.top_k, self._collection.count()),
        )

        items: list[RetrievalResult] = []
        if results and results["documents"]:
            documents = results["documents"][0]
            metadatas = (results["metadatas"] or [[]])[0]
            distances = (results["distances"] or [[]])[0]
            for doc, meta, dist in zip(documents, metadatas, distances):
                items.append(
                    RetrievalResult(
                        content=doc,
                        source_url=meta.get("source_url", "") if meta else "",
                        score=1.0 - dist,  # convert distance → similarity
                    )
                )

        # Consider data "found" if there is at least one result with a
        # reasonable similarity score.
        found = any(r.score > 0.3 for r in items)
        return VectorSearchResult(found=found, results=items if found else [])

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

        self._collection.add(
            documents=contents,
            metadatas=[{"source_url": url} for url in source_urls],
            ids=ids,
        )
