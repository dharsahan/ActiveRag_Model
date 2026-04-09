"""Ingestion router — /api/v1/ingest endpoints."""

from __future__ import annotations

import os
import logging
from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from active_rag.dependencies import ResourceManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ingest", tags=["Ingestion"])


# --- Models ---

class TextIngestRequest(BaseModel):
    content: str
    source: str = "api_ingest"
    title: Optional[str] = None


class BatchIngestItem(BaseModel):
    content: str
    source: str = "batch_ingest"
    title: Optional[str] = None


class BatchIngestRequest(BaseModel):
    documents: List[BatchIngestItem]


class UrlIngestRequest(BaseModel):
    url: str
    max_pages: int = 1


def register(r: APIRouter, resources: ResourceManager):
    """Register ingestion endpoints on the given router."""

    @r.post("/upload")
    async def upload_file(file: UploadFile = File(...)):
        """Upload a file (PDF, TXT, MD, DOCX) for ingestion."""
        temp_dir = Path("data/uploads")
        temp_dir.mkdir(parents=True, exist_ok=True)
        file_path = temp_dir / file.filename

        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        try:
            docs = resources.document_loader.load(str(file_path))
            chunk_ids = []
            for d in docs:
                cids = resources.vector_store.add_documents([d.content], [d.source])
                chunk_ids.extend(cids)

            os.remove(file_path)

            return {
                "status": "success",
                "filename": file.filename,
                "chunks": len(chunk_ids),
                "ids": chunk_ids[:10],
            }
        except Exception as e:
            if file_path.exists():
                os.remove(file_path)
            raise HTTPException(status_code=400, detail=str(e))

    @r.post("/text")
    async def ingest_text(req: TextIngestRequest):
        """Ingest raw text content."""
        try:
            chunk_ids = resources.vector_store.add_documents(
                [req.content], [req.source]
            )
            return {"status": "success", "chunks": len(chunk_ids), "ids": chunk_ids}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @r.post("/batch")
    async def ingest_batch(req: BatchIngestRequest):
        """Bulk-ingest multiple documents at once.

        Ideal for loading FAQ entries, legal records, or knowledge base articles.
        """
        if not req.documents:
            raise HTTPException(status_code=400, detail="No documents provided")

        contents = [d.content for d in req.documents]
        sources = [d.source for d in req.documents]

        try:
            chunk_ids = resources.vector_store.add_documents(contents, sources)
            return {
                "status": "success",
                "total_documents": len(req.documents),
                "chunks_created": len(chunk_ids),
                "ids": chunk_ids[:20],
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @r.post("/url")
    async def ingest_url(req: UrlIngestRequest):
        """Scrape a URL and ingest its content into the knowledge base."""
        try:
            from active_rag.web_search import WebSearcher

            searcher = WebSearcher(resources.config)
            pages = searcher.scrape_urls([req.url])

            if not pages:
                raise HTTPException(status_code=400, detail="Failed to scrape URL")

            chunk_ids = []
            for page in pages[:req.max_pages]:
                if page.content and len(page.content.strip()) > 50:
                    cids = resources.vector_store.add_documents(
                        [page.content], [page.url]
                    )
                    chunk_ids.extend(cids)

            return {
                "status": "success",
                "url": req.url,
                "pages_scraped": len(pages),
                "chunks_created": len(chunk_ids),
                "ids": chunk_ids[:10],
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
