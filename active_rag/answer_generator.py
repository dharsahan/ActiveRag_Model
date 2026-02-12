"""Answer generation module.

Provides two generation paths that mirror the architecture diagram:
1. Generate Answer Directly  – high-confidence, no retrieval needed.
2. Generate Answer with Citations – uses retrieved context & sources.
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from active_rag.config import Config
from active_rag.vector_store import RetrievalResult


@dataclass
class Answer:
    """Final answer returned to the user."""

    text: str
    citations: list[str]
    source: str  # "direct" | "rag"


class AnswerGenerator:
    """Generates answers via the LLM, optionally enriched with context."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = OpenAI(api_key=config.openai_api_key)

    # ------------------------------------------------------------------
    # Generate Answer Directly  (High Confidence path)
    # ------------------------------------------------------------------
    def generate_direct(self, query: str) -> Answer:
        """Answer *query* directly without any retrieval context."""
        response = self._client.chat.completions.create(
            model=self._config.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer the user's "
                        "question directly and concisely."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.3,
        )
        text = response.choices[0].message.content or ""
        return Answer(text=text, citations=[], source="direct")

    # ------------------------------------------------------------------
    # Generate Answer with Citations  (RAG path)
    # ------------------------------------------------------------------
    def generate_with_citations(
        self,
        query: str,
        context_results: list[RetrievalResult],
    ) -> Answer:
        """Answer *query* using the provided retrieval *context_results*.

        The answer includes citations referencing the source URLs.
        """
        context_block = "\n\n".join(
            f"[Source: {r.source_url}]\n{r.content}" for r in context_results
        )

        response = self._client.chat.completions.create(
            model=self._config.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Answer the user's "
                        "question using ONLY the provided context. "
                        "Include citations in your answer by referencing "
                        "the source URLs provided with each context block."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context_block}\n\n"
                        f"Question: {query}"
                    ),
                },
            ],
            temperature=0.3,
        )
        text = response.choices[0].message.content or ""
        citations = list(
            dict.fromkeys(r.source_url for r in context_results if r.source_url)
        )
        return Answer(text=text, citations=citations, source="rag")
