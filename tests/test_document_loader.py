"""Tests for the document loader module."""

import os
import tempfile

import pytest

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
        # Bold markers should be stripped
        assert "**" not in docs[0].content
    finally:
        os.unlink(path)


def test_unsupported_extension_raises():
    """Unsupported file types raise ValueError."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
        f.write("content")
        path = f.name
    try:
        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load(path)
    finally:
        os.unlink(path)


def test_file_not_found_raises():
    """Non-existent files raise FileNotFoundError."""
    loader = DocumentLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("/nonexistent/file.txt")


def test_word_count_is_populated():
    """LoadedDocument word_count reflects actual content."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("one two three four five")
        path = f.name
    try:
        loader = DocumentLoader()
        docs = loader.load(path)
        assert docs[0].word_count == 5
    finally:
        os.unlink(path)
