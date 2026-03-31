# ActiveRag_Model — New Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend ActiveRag_Model from a CLI-only chatbot into a production-grade RAG platform with document ingestion, a web API, smarter chunking, answer quality validation, persistent conversations, and token usage tracking.

**Architecture:** The plan preserves the existing modular design (`active_rag/` package) and adds features as new modules with clear interfaces. Each feature is independently implementable — pick any task and build it without needing the others.

**Tech Stack:** Python 3.10+, FastAPI, ChromaDB, OpenAI-compatible APIs, Rich CLI, pytest

---

## Current State Summary

| Module | LOC | Responsibility |
|---|---|---|
| `pipeline.py` | 362 | Main orchestrator (sync + streaming) |
| `answer_generator.py` | 212 | LLM answer generation (direct + RAG) |
| `web_search.py` | 188 | DuckDuckGo + parallel scraping |
| `vector_store.py` | 151 | ChromaDB wrapper |
| `console.py` | 144 | Rich CLI output |
| `memory.py` | 87 | In-memory conversation history |
| `confidence_checker.py` | 85 | LLM self-evaluation |
| `cache.py` | 85 | Disk-based response cache |
| `config.py` | 60 | Env-based configuration |

---

## Feature 1: Document Ingestion (PDF, TXT, Markdown)

**Why:** Users can only add knowledge via web search. They should be able to upload their own documents (research papers, notes, manuals) directly into the vector store.

### Task 1.1: Document Parser Module

**Files:**
- Create: `active_rag/document_loader.py`
- Create: `tests/test_document_loader.py`
- Modify: `requirements.txt` (add `PyPDF2>=3.0.0`, `python-docx>=1.0.0`)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_document_loader.py
"""Tests for the document loader module."""
import os
import tempfile

from active_rag.document_loader import DocumentLoader, LoadedDocument


def test_load_txt_file():
    """Loading a .txt file returns its content."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Hello world. This is a test document with enough content to pass validation.")
        path = f.name
    try:
        loader = DocumentLoader()
        docs = loader.load(path)
        assert len(docs) >= 1
        assert "Hello world" in docs[0].content
        assert docs[0].source == path
    finally:
        os.unlink(path)


def test_load_markdown_file():
    """Loading a .md file strips markdown syntax."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("# Title\n\nThis is **bold** text with enough content for the loader to accept it.")
        path = f.name
    try:
        loader = DocumentLoader()
        docs = loader.load(path)
        assert len(docs) >= 1
        assert "Title" in docs[0].content
    finally:
        os.unlink(path)


def test_unsupported_extension_raises():
    """Unsupported file types raise ValueError."""
    import pytest
    loader = DocumentLoader()
    with pytest.raises(ValueError, match="Unsupported"):
        loader.load("file.xyz")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_document_loader.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the document loader**

```python
# active_rag/document_loader.py
"""Document ingestion for local files (TXT, MD, PDF, DOCX)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


@dataclass
class LoadedDocument:
    """A document loaded from a local file."""
    content: str
    source: str
    title: str = ""
    word_count: int = 0


class DocumentLoader:
    """Loads documents from local files for vector store ingestion."""

    def load(self, path: str) -> list[LoadedDocument]:
        """Load a document from *path* and return parsed content."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = p.suffix.lower()
        if ext not in _SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
            )

        if ext == ".txt":
            return self._load_text(p)
        elif ext == ".md":
            return self._load_markdown(p)
        elif ext == ".pdf":
            return self._load_pdf(p)
        elif ext == ".docx":
            return self._load_docx(p)
        return []

    def _load_text(self, path: Path) -> list[LoadedDocument]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [LoadedDocument(
            content=text, source=str(path),
            title=path.stem, word_count=len(text.split()),
        )]

    def _load_markdown(self, path: Path) -> list[LoadedDocument]:
        import re
        text = path.read_text(encoding="utf-8", errors="ignore")
        # Strip markdown formatting
        text = re.sub(r"#{1,6}\s*", "", text)  # headers
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # bold
        text = re.sub(r"\*(.+?)\*", r"\1", text)  # italic
        text = re.sub(r"`(.+?)`", r"\1", text)  # inline code
        return [LoadedDocument(
            content=text, source=str(path),
            title=path.stem, word_count=len(text.split()),
        )]

    def _load_pdf(self, path: Path) -> list[LoadedDocument]:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(LoadedDocument(
                    content=text, source=f"{path}#page={i+1}",
                    title=f"{path.stem} (p{i+1})", word_count=len(text.split()),
                ))
        return pages

    def _load_docx(self, path: Path) -> list[LoadedDocument]:
        from docx import Document as DocxDocument
        doc = DocxDocument(str(path))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [LoadedDocument(
            content=text, source=str(path),
            title=path.stem, word_count=len(text.split()),
        )]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_document_loader.py -v`
Expected: PASS

- [ ] **Step 5: Add CLI command for ingestion**

Add to `main.py` argparse:

```python
parser.add_argument(
    "--ingest", "-i", nargs="+",
    help="Ingest local files into the vector store.",
)
```

Add handler in `main()`:

```python
if args.ingest:
    from active_rag.document_loader import DocumentLoader
    loader = DocumentLoader()
    for filepath in args.ingest:
        with status_spinner(f"Ingesting {filepath}..."):
            docs = loader.load(filepath)
            pipeline._vector_store.add_documents(
                contents=[d.content for d in docs],
                source_urls=[d.source for d in docs],
            )
        print_success(f"Ingested {len(docs)} chunk(s) from {filepath}")
    return
```

- [ ] **Step 6: Commit**

```bash
git add active_rag/document_loader.py tests/test_document_loader.py main.py requirements.txt
git commit -m "feat: add local document ingestion (PDF, TXT, MD, DOCX)"
```

---

## Feature 2: REST API with FastAPI

**Why:** The CLI is great for personal use, but a REST API enables web frontends, mobile apps, and integration with other tools.

### Task 2.1: FastAPI Server

**Files:**
- Create: `active_rag/api.py`
- Create: `tests/test_api.py`
- Modify: `requirements.txt` (add `fastapi>=0.100.0`, `uvicorn>=0.20.0`)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_api.py
"""Tests for the REST API."""
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from active_rag.api import create_app


@patch("active_rag.api.ActiveRAGPipeline")
def test_query_endpoint(mock_pipeline_cls):
    """POST /query returns an answer."""
    mock_pipeline = MagicMock()
    mock_result = MagicMock()
    mock_result.answer.text = "Test answer"
    mock_result.answer.citations = ["https://example.com"]
    mock_result.path = "direct"
    mock_result.confidence.confidence = 0.9
    mock_result.confidence.reasoning = "Known fact"
    mock_result.web_pages_indexed = 0
    mock_result.from_cache = False
    mock_pipeline.run.return_value = mock_result
    mock_pipeline_cls.return_value = mock_pipeline

    app = create_app()
    client = TestClient(app)

    response = client.post("/query", json={"query": "What is Python?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Test answer"
    assert data["path"] == "direct"


@patch("active_rag.api.ActiveRAGPipeline")
def test_health_endpoint(mock_pipeline_cls):
    """GET /health returns ok."""
    app = create_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_api.py -v`
Expected: FAIL

- [ ] **Step 3: Implement the API server**

```python
# active_rag/api.py
"""FastAPI REST API for the Active RAG pipeline."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from active_rag.config import Config
from active_rag.pipeline import ActiveRAGPipeline


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    path: str
    confidence: float | None = None
    reasoning: str | None = None
    web_pages_indexed: int = 0
    from_cache: bool = False


def create_app(config: Config | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Active RAG API",
        description="Refined Retrieval-Augmented Generation",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    pipeline = ActiveRAGPipeline(config or Config())

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest):
        result = pipeline.run(req.query)
        return QueryResponse(
            answer=result.answer.text,
            citations=result.answer.citations,
            path=result.path,
            confidence=result.confidence.confidence if result.confidence else None,
            reasoning=result.confidence.reasoning if result.confidence else None,
            web_pages_indexed=result.web_pages_indexed,
            from_cache=result.from_cache,
        )

    @app.post("/clear-memory")
    def clear_memory():
        pipeline.clear_memory()
        return {"status": "cleared"}

    @app.post("/clear-cache")
    def clear_cache():
        pipeline.clear_cache()
        return {"status": "cleared"}

    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
```

- [ ] **Step 4: Run tests**

Run: `venv/bin/python -m pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 5: Add CLI flag to start API server**

Add to `main.py`:
```python
parser.add_argument("--serve", action="store_true", help="Start the REST API server.")

# In main():
if args.serve:
    import uvicorn
    from active_rag.api import create_app
    print_info("Starting API server at http://0.0.0.0:8000")
    uvicorn.run(create_app(config), host="0.0.0.0", port=8000)
    return
```

- [ ] **Step 6: Commit**

```bash
git add active_rag/api.py tests/test_api.py main.py requirements.txt
git commit -m "feat: add FastAPI REST API with /query, /health endpoints"
```

---

## Feature 3: Semantic Document Chunking

**Why:** Currently web pages are truncated at 8KB with no semantic awareness. Large documents lose context. Proper chunking improves retrieval quality dramatically.

### Task 3.1: Text Chunker Module

**Files:**
- Create: `active_rag/chunker.py`
- Create: `tests/test_chunker.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_chunker.py
"""Tests for the semantic text chunker."""

from active_rag.chunker import TextChunker


def test_short_text_single_chunk():
    """Text shorter than chunk_size stays as one chunk."""
    chunker = TextChunker(chunk_size=500, overlap=50)
    chunks = chunker.chunk("Short text.")
    assert len(chunks) == 1
    assert chunks[0] == "Short text."


def test_long_text_splits_at_paragraph_boundaries():
    """Long text splits at paragraph boundaries."""
    text = ("Paragraph one. " * 50 + "\n\n" + "Paragraph two. " * 50)
    chunker = TextChunker(chunk_size=200, overlap=30)
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
    # Chunks should not cut mid-sentence
    for chunk in chunks:
        assert not chunk.startswith(" ")


def test_overlap_creates_shared_content():
    """Adjacent chunks share overlapping content."""
    text = " ".join(f"Sentence {i}." for i in range(100))
    chunker = TextChunker(chunk_size=200, overlap=50)
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_chunker.py -v`
Expected: FAIL

- [ ] **Step 3: Implement the chunker**

```python
# active_rag/chunker.py
"""Semantic text chunker with paragraph-aware splitting and overlap."""

from __future__ import annotations

import re


class TextChunker:
    """Splits text into overlapping, semantically-aware chunks."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 100) -> None:
        self._chunk_size = chunk_size
        self._overlap = overlap

    def chunk(self, text: str) -> list[str]:
        """Split *text* into chunks, preferring paragraph/sentence boundaries."""
        if len(text) <= self._chunk_size:
            return [text]

        # Split into paragraphs first
        paragraphs = re.split(r"\n{2,}", text)
        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 1 <= self._chunk_size:
                current = f"{current}\n\n{para}".strip() if current else para
            else:
                if current:
                    chunks.append(current)
                # If paragraph itself is too long, split by sentences
                if len(para) > self._chunk_size:
                    chunks.extend(self._split_by_sentences(para))
                else:
                    current = para
                    continue
                current = ""

        if current.strip():
            chunks.append(current)

        # Add overlap between chunks
        if self._overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)

        return chunks

    def _split_by_sentences(self, text: str) -> list[str]:
        """Split text by sentence boundaries when paragraphs are too long."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) + 1 <= self._chunk_size:
                current = f"{current} {sent}".strip() if current else sent
            else:
                if current:
                    chunks.append(current)
                current = sent
        if current.strip():
            chunks.append(current)
        return chunks

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlapping content between adjacent chunks."""
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-self._overlap:]
            # Don't cut mid-word
            space_idx = prev_tail.find(" ")
            if space_idx > 0:
                prev_tail = prev_tail[space_idx + 1:]
            result.append(f"{prev_tail} {chunks[i]}")
        return result
```

- [ ] **Step 4: Run tests**

Run: `venv/bin/python -m pytest tests/test_chunker.py -v`
Expected: PASS

- [ ] **Step 5: Integrate with web_search.py and document_loader.py**

In `web_search.py`, replace the 8KB truncation:

```python
from active_rag.chunker import TextChunker

# In scrape():
# Replace: text = text[:8000]
# With:
chunker = TextChunker(chunk_size=2000, overlap=200)
chunks = chunker.chunk(text)
text = chunks[0] if chunks else text  # Use first chunk for search results
```

- [ ] **Step 6: Commit**

```bash
git add active_rag/chunker.py tests/test_chunker.py active_rag/web_search.py
git commit -m "feat: add semantic text chunker with paragraph-aware splitting"
```

---

## Feature 4: Answer Quality Self-Evaluation

**Why:** The system checks confidence *before* answering but never validates the *quality* of the answer it produces. A post-generation quality check catches hallucinations and incomplete answers.

### Task 4.1: Answer Evaluator Module

**Files:**
- Create: `active_rag/evaluator.py`
- Create: `tests/test_evaluator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_evaluator.py
"""Tests for the answer quality evaluator."""
import json
from unittest.mock import MagicMock, patch

from active_rag.config import Config
from active_rag.evaluator import AnswerEvaluator


def _mock_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@patch("active_rag.evaluator.OpenAI")
def test_evaluate_good_answer(mock_openai):
    """A good answer scores high quality."""
    client = MagicMock()
    mock_openai.return_value = client
    client.chat.completions.create.return_value = _mock_response(
        json.dumps({"quality": 0.9, "issues": [], "suggestion": ""})
    )
    evaluator = AnswerEvaluator(Config())
    result = evaluator.evaluate("What is Python?", "Python is a programming language.")
    assert result.quality >= 0.8
    assert result.is_acceptable is True


@patch("active_rag.evaluator.OpenAI")
def test_evaluate_bad_answer(mock_openai):
    """A bad answer scores low quality."""
    client = MagicMock()
    mock_openai.return_value = client
    client.chat.completions.create.return_value = _mock_response(
        json.dumps({"quality": 0.2, "issues": ["off-topic"], "suggestion": "Address the question"})
    )
    evaluator = AnswerEvaluator(Config())
    result = evaluator.evaluate("What is Python?", "The sky is blue.")
    assert result.quality < 0.5
    assert result.is_acceptable is False
```

- [ ] **Step 2: Run tests → FAIL**

- [ ] **Step 3: Implement evaluator**

```python
# active_rag/evaluator.py
"""Post-generation answer quality evaluation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from openai import OpenAI

from active_rag.config import Config

logger = logging.getLogger(__name__)

_EVAL_SYSTEM_PROMPT = (
    "You are an answer quality evaluator. Given a question and an answer, "
    "evaluate the answer quality.\n\n"
    "Respond with ONLY a JSON object:\n"
    '{"quality": <float 0.0-1.0>, "issues": [<list of issues>], '
    '"suggestion": "<improvement suggestion or empty string>"}\n\n'
    "Score 1.0 = perfect answer. Score 0.0 = completely wrong/irrelevant."
)


@dataclass
class EvaluationResult:
    """Result of answer quality evaluation."""
    quality: float
    issues: list[str]
    suggestion: str
    is_acceptable: bool  # quality >= 0.5


class AnswerEvaluator:
    """Evaluates generated answers for quality and relevance."""

    def __init__(self, config: Config) -> None:
        self._client = OpenAI(
            base_url=config.ollama_base_url,
            api_key=config.api_key,
        )
        self._config = config

    def evaluate(self, query: str, answer: str) -> EvaluationResult:
        """Evaluate the *answer* for the given *query*."""
        response = self._client.chat.completions.create(
            model=self._config.model_name,
            messages=[
                {"role": "system", "content": _EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {query}\n\nAnswer: {answer}"},
            ],
            temperature=0.0,
        )
        content = response.choices[0].message.content or "{}"
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {"quality": 0.5, "issues": ["Failed to parse evaluation"], "suggestion": ""}

        quality = float(data.get("quality", 0.5))
        return EvaluationResult(
            quality=quality,
            issues=data.get("issues", []),
            suggestion=data.get("suggestion", ""),
            is_acceptable=quality >= 0.5,
        )
```

- [ ] **Step 4: Run tests → PASS**
- [ ] **Step 5: Commit**

```bash
git add active_rag/evaluator.py tests/test_evaluator.py
git commit -m "feat: add post-generation answer quality evaluator"
```

---

## Feature 5: Persistent Conversation History

**Why:** Conversation memory is lost when the CLI exits. Users should be able to resume conversations across sessions.

### Task 5.1: SQLite-Based Conversation Store

**Files:**
- Create: `active_rag/conversation_store.py`
- Create: `tests/test_conversation_store.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_conversation_store.py
"""Tests for persistent conversation storage."""
import tempfile
import os

from active_rag.conversation_store import ConversationStore


def test_save_and_load_conversation():
    """Messages persist across store instances."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        store = ConversationStore(db_path)
        conv_id = store.create_conversation("Test Chat")
        store.add_message(conv_id, "user", "Hello")
        store.add_message(conv_id, "assistant", "Hi there!")

        # New instance should see the same messages
        store2 = ConversationStore(db_path)
        messages = store2.get_messages(conv_id)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["content"] == "Hi there!"
    finally:
        os.unlink(db_path)


def test_list_conversations():
    """Can list all conversations."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        store = ConversationStore(db_path)
        store.create_conversation("Chat 1")
        store.create_conversation("Chat 2")
        convs = store.list_conversations()
        assert len(convs) == 2
    finally:
        os.unlink(db_path)
```

- [ ] **Step 2: Run tests → FAIL**

- [ ] **Step 3: Implement conversation store**

```python
# active_rag/conversation_store.py
"""SQLite-based persistent conversation storage."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path


class ConversationStore:
    """Persists conversations to SQLite for cross-session memory."""

    def __init__(self, db_path: str = "conversations.db") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );
        """)

    def create_conversation(self, title: str = "New Chat") -> str:
        conv_id = uuid.uuid4().hex[:12]
        now = datetime.now().isoformat()
        self._conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conv_id, title, now, now),
        )
        self._conn.commit()
        return conv_id

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        now = datetime.now().isoformat()
        self._conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, now),
        )
        self._conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )
        self._conn.commit()

    def get_messages(self, conversation_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_conversations(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_conversation(self, conversation_id: str) -> None:
        self._conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        self._conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        self._conn.commit()
```

- [ ] **Step 4: Run tests → PASS**
- [ ] **Step 5: Commit**

```bash
git add active_rag/conversation_store.py tests/test_conversation_store.py
git commit -m "feat: add SQLite-based persistent conversation history"
```

---

## Feature 6: Token Usage Tracking & Cost Estimation

**Why:** Users have no visibility into API costs. Every LLM call should track token usage and estimate cost.

### Task 6.1: Token Tracker Module

**Files:**
- Create: `active_rag/token_tracker.py`
- Create: `tests/test_token_tracker.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_token_tracker.py
"""Tests for token usage tracking."""

from active_rag.token_tracker import TokenTracker


def test_track_usage():
    """Token tracker accumulates usage across calls."""
    tracker = TokenTracker()
    tracker.record(prompt_tokens=100, completion_tokens=50, model="gpt-3.5")
    tracker.record(prompt_tokens=200, completion_tokens=100, model="gpt-3.5")

    stats = tracker.stats()
    assert stats["total_prompt_tokens"] == 300
    assert stats["total_completion_tokens"] == 150
    assert stats["total_tokens"] == 450
    assert stats["call_count"] == 2


def test_cost_estimation():
    """Tracker estimates cost based on model pricing."""
    tracker = TokenTracker()
    tracker.record(prompt_tokens=1000, completion_tokens=500, model="gpt-3.5-turbo")
    stats = tracker.stats()
    assert stats["estimated_cost_usd"] >= 0


def test_reset():
    """Reset clears all tracked data."""
    tracker = TokenTracker()
    tracker.record(prompt_tokens=100, completion_tokens=50, model="test")
    tracker.reset()
    assert tracker.stats()["total_tokens"] == 0
```

- [ ] **Step 2: Run tests → FAIL**

- [ ] **Step 3: Implement token tracker**

```python
# active_rag/token_tracker.py
"""Token usage tracking and cost estimation."""

from __future__ import annotations

from dataclasses import dataclass, field


# Approximate pricing per 1K tokens (input/output)
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "gpt-4": (0.03, 0.06),
    "gpt-4-turbo": (0.01, 0.03),
    "llama3.2": (0.0, 0.0),  # Local/free
    "stepfun-ai/step-3.5-flash": (0.0001, 0.0003),
}
_DEFAULT_PRICING = (0.001, 0.002)


@dataclass
class UsageRecord:
    prompt_tokens: int
    completion_tokens: int
    model: str


class TokenTracker:
    """Tracks token usage and estimates cost across pipeline calls."""

    def __init__(self) -> None:
        self._records: list[UsageRecord] = []

    def record(self, prompt_tokens: int, completion_tokens: int, model: str) -> None:
        self._records.append(UsageRecord(prompt_tokens, completion_tokens, model))

    def stats(self) -> dict:
        total_prompt = sum(r.prompt_tokens for r in self._records)
        total_completion = sum(r.completion_tokens for r in self._records)
        total_cost = 0.0
        for r in self._records:
            input_price, output_price = _MODEL_PRICING.get(r.model, _DEFAULT_PRICING)
            total_cost += (r.prompt_tokens / 1000) * input_price
            total_cost += (r.completion_tokens / 1000) * output_price

        return {
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "call_count": len(self._records),
            "estimated_cost_usd": round(total_cost, 6),
        }

    def reset(self) -> None:
        self._records.clear()
```

- [ ] **Step 4: Run tests → PASS**
- [ ] **Step 5: Commit**

```bash
git add active_rag/token_tracker.py tests/test_token_tracker.py
git commit -m "feat: add token usage tracking with cost estimation"
```

---

## Feature 7: Multi-Provider LLM Support

**Why:** Currently hardcoded to one provider. Users should be able to switch between NVIDIA, OpenAI, Anthropic, or local Ollama from config.

### Task 7.1: Provider Abstraction

**Files:**
- Create: `active_rag/providers.py`
- Create: `tests/test_providers.py`
- Modify: `active_rag/config.py` (add `LLM_PROVIDER` env var)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_providers.py
"""Tests for the LLM provider abstraction."""

from active_rag.providers import get_provider_config


def test_nvidia_provider():
    config = get_provider_config("nvidia")
    assert "nvidia.com" in config["base_url"]


def test_ollama_provider():
    config = get_provider_config("ollama")
    assert "localhost" in config["base_url"]
    assert config["api_key"] == "ollama"


def test_openai_provider():
    config = get_provider_config("openai")
    assert "openai.com" in config["base_url"]


def test_unknown_provider_raises():
    import pytest
    with pytest.raises(ValueError):
        get_provider_config("unknown_provider")
```

- [ ] **Step 2: Run tests → FAIL**

- [ ] **Step 3: Implement provider config**

```python
# active_rag/providers.py
"""Multi-provider LLM configuration."""

from __future__ import annotations

_PROVIDERS: dict[str, dict[str, str]] = {
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "default_model": "stepfun-ai/step-3.5-flash",
        "api_key_env": "NVIDIA_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3.2",
        "api_key": "ollama",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4-turbo",
        "api_key_env": "OPENAI_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.1-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "default_model": "meta-llama/Llama-3-70b-chat-hf",
        "api_key_env": "TOGETHER_API_KEY",
    },
}


def get_provider_config(provider: str) -> dict[str, str]:
    """Return base URL, default model, and API key info for *provider*."""
    if provider not in _PROVIDERS:
        available = ", ".join(sorted(_PROVIDERS.keys()))
        raise ValueError(f"Unknown provider: {provider}. Available: {available}")
    return _PROVIDERS[provider]


def list_providers() -> list[str]:
    """Return all available provider names."""
    return sorted(_PROVIDERS.keys())
```

- [ ] **Step 4: Run tests → PASS**
- [ ] **Step 5: Commit**

```bash
git add active_rag/providers.py tests/test_providers.py
git commit -m "feat: add multi-provider LLM support (NVIDIA, OpenAI, Ollama, Groq, Together)"
```

---

## Feature 8: Vector Store Management CLI

**Why:** Users can't inspect, search, or clean up the vector store. They need tools to manage their knowledge base.

### Task 8.1: Vector Store CLI Commands

**Files:**
- Modify: `main.py` (add `--db-stats`, `--db-search`, `--db-clear` flags)

- [ ] **Step 1: Add CLI arguments**

```python
# Add to argparse in main.py:
db_group = parser.add_argument_group("Vector Store Management")
db_group.add_argument("--db-stats", action="store_true", help="Show vector store statistics.")
db_group.add_argument("--db-search", type=str, help="Search the vector store for a query.")
db_group.add_argument("--db-clear", action="store_true", help="Clear all documents from vector store.")
db_group.add_argument("--db-export", type=str, help="Export vector store contents to JSON file.")
```

- [ ] **Step 2: Implement handlers**

```python
# In main():
if args.db_stats:
    from active_rag.vector_store import VectorStore
    store = VectorStore(config)
    count = store._collection.count()
    print_info(f"Documents: {count}")
    print_info(f"Collection: {config.collection_name}")
    print_info(f"Persist dir: {config.chroma_persist_dir}")
    return

if args.db_search:
    from active_rag.vector_store import VectorStore
    store = VectorStore(config)
    result = store.search(args.db_search)
    if result.found:
        for r in result.results:
            console.print(f"\n[bold]{r.source_url}[/bold] (score: {r.score:.2f})")
            console.print(f"[dim]{r.content[:200]}...[/dim]")
    else:
        print_warning("No matching documents found.")
    return

if args.db_clear:
    from active_rag.vector_store import VectorStore
    store = VectorStore(config)
    store._client.delete_collection(config.collection_name)
    print_success("Vector store cleared!")
    return

if args.db_export:
    import json
    from active_rag.vector_store import VectorStore
    store = VectorStore(config)
    all_data = store._collection.get(include=["documents", "metadatas"])
    export = []
    for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
        export.append({"content": doc, "metadata": meta})
    with open(args.db_export, "w") as f:
        json.dump(export, f, indent=2)
    print_success(f"Exported {len(export)} documents to {args.db_export}")
    return
```

- [ ] **Step 3: Test manually**

```bash
venv/bin/python main.py --db-stats
venv/bin/python main.py --db-search "Python programming"
venv/bin/python main.py --db-export backup.json
```

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: add vector store management CLI (stats, search, clear, export)"
```

---

## Feature Priority Matrix

| # | Feature | Impact | Effort | Priority |
|---|---|---|---|---|
| 1 | Document Ingestion | 🔴 High | 🟡 Medium | ⭐ P0 |
| 2 | REST API (FastAPI) | 🔴 High | 🟡 Medium | ⭐ P0 |
| 3 | Semantic Chunking | 🔴 High | 🟢 Low | ⭐ P0 |
| 4 | Answer Quality Eval | 🟡 Medium | 🟢 Low | P1 |
| 5 | Persistent Conversations | 🟡 Medium | 🟢 Low | P1 |
| 6 | Token Tracking | 🟡 Medium | 🟢 Low | P1 |
| 7 | Multi-Provider LLM | 🟡 Medium | 🟢 Low | P2 |
| 8 | Vector Store CLI | 🟢 Low | 🟢 Low | P2 |

> **Recommended order:** Start with Features 3 → 1 → 2 (chunking improves ingestion, ingestion improves the API), then add quality/tracking features 4-6 in any order.

---

## Verification Plan

### Automated Tests

Each feature includes its own test file. Run all tests:
```bash
venv/bin/python -m pytest tests/ -v
```

### Manual Verification

1. **Document Ingestion:** `python main.py --ingest docs/sample.pdf` then query about the content
2. **REST API:** `python main.py --serve` then `curl -X POST http://localhost:8000/query -d '{"query":"hello"}'`
3. **Chunking:** Ingest a large document, verify multiple chunks stored in `--db-stats`
4. **Answer Eval:** Enable verbose mode and observe quality scores in output
5. **Conversations:** Start CLI, chat, exit, restart — verify history is preserved
6. **Token Tracking:** Use `/tokens` CLI command to see usage stats
7. **Providers:** `LLM_PROVIDER=ollama python main.py "test"` vs `LLM_PROVIDER=nvidia python main.py "test"`
8. **DB Management:** `python main.py --db-stats`, `--db-search`, `--db-export`
