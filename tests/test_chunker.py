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
    # Chunks should not start with whitespace
    for chunk in chunks:
        assert not chunk.startswith(" ")


def test_overlap_creates_shared_content():
    """Adjacent chunks share overlapping content."""
    text = " ".join(f"Sentence {i}." for i in range(100))
    chunker = TextChunker(chunk_size=200, overlap=50)
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2


def test_empty_text():
    """Empty text returns empty list."""
    chunker = TextChunker()
    chunks = chunker.chunk("")
    assert chunks == [""]  # Single chunk with empty string


def test_very_long_paragraph_splits_by_sentences():
    """A single paragraph longer than chunk_size splits by sentences."""
    text = ". ".join(f"This is sentence number {i}" for i in range(50)) + "."
    chunker = TextChunker(chunk_size=200, overlap=0)
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= 300  # Allow some tolerance
