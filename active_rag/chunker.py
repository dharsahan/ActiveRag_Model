"""Semantic text chunker with paragraph-aware splitting and overlap."""

from __future__ import annotations

import re


class TextChunker:
    """Splits text into overlapping, semantically-aware chunks.

    Prefers paragraph and sentence boundaries over hard character cuts.
    """

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
                    current = ""
                else:
                    current = para

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
            result.append(f"{prev_tail} {chunks[i]}".strip())
        return result
