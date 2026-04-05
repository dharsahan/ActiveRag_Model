"""Answer generation module with streaming and retry support.

Provides two generation paths that mirror the architecture diagram:
1. Generate Answer Directly  – high-confidence, no retrieval needed.
2. Generate Answer with Citations – uses retrieved context & sources.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Generator

from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from openai import APIConnectionError, RateLimitError, APITimeoutError

from active_rag.config import Config
from active_rag.vector_store import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class Answer:
    """Final answer returned to the user."""

    text: str
    citations: list[str]
    source: str  # "direct" | "rag"


# Retry decorator for API calls
api_retry = retry(
    retry=retry_if_exception_type((APIConnectionError, RateLimitError, APITimeoutError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=lambda retry_state: logger.warning(
        "API call failed, retrying in %s seconds...",
        retry_state.next_action.sleep
    ),
)


class AnswerGenerator:
    """Generates answers via the LLM, optionally enriched with context."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = OpenAI(
            base_url=config.ollama_base_url,
            api_key=config.api_key,
        )

    # ------------------------------------------------------------------
    # Generate Answer Directly  (High Confidence path)
    # ------------------------------------------------------------------
    @api_retry
    def generate_direct(self, query: str, conversation_context: str = "") -> Answer:
        """Answer *query* directly without any retrieval context."""
        system_content = (
            "You are a helpful assistant. Answer the user's question directly and concisely. "
            "Always format your responses clearly using markdown:\n"
            "- Use bullet points for lists\n"
            "- Use **bold** for emphasis\n"
            "- Use headers (## Header) for sections when appropriate\n"
            "- Use proper paragraphs with line breaks\n"
            "- Keep responses well-structured and readable\n"
            "Provide clear, well-formatted answers."
        )

        if conversation_context:
            system_content += f"\n\nConversation context:\n{conversation_context}"

        response = self._client.chat.completions.create(
            model=self._config.model_name,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
        )
        if response.choices:
            text = response.choices[0].message.content or ""
        else:
            text = ""
        text = self._post_process_response(text)
        return Answer(text=text, citations=[], source="direct")

    def generate_direct_stream(
        self, query: str, conversation_context: str = ""
    ) -> Generator[str, None, Answer]:
        """Stream answer tokens for *query* directly without retrieval."""
        system_content = (
            "You are a helpful assistant. Answer the user's question directly and concisely. "
            "Always format your responses clearly using markdown:\n"
            "- Use bullet points for lists\n"
            "- Use **bold** for emphasis\n"
            "- Use headers (## Header) for sections when appropriate\n"
            "- Use proper paragraphs with line breaks\n"
            "- Keep responses well-structured and readable\n"
            "Provide clear, well-formatted answers."
        )

        if conversation_context:
            system_content += f"\n\nConversation context:\n{conversation_context}"

        response = self._client.chat.completions.create(
            model=self._config.model_name,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            stream=True,
        )

        full_text = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_text += token
                yield token

        full_text = self._post_process_response(full_text)
        return Answer(text=full_text, citations=[], source="direct")

    # ------------------------------------------------------------------
    # Generate Answer with Citations  (RAG path)
    # ------------------------------------------------------------------
    @api_retry
    def generate_with_citations(
        self,
        query: str,
        context_results: list[RetrievalResult],
        conversation_context: str = "",
    ) -> Answer:
        """Answer *query* using the provided retrieval *context_results*.

        The answer includes citations referencing the source URLs.
        """
        context_block = "\n\n".join(
            f"[Source: {r.source_url}]\n{r.content}" for r in context_results
        )

        system_content = (
            "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
            "Always format your responses clearly using markdown:\n"
            "- Use bullet points for lists\n"
            "- Use **bold** for emphasis\n"
            "- Use headers (## Header) for sections when appropriate\n"
            "- Use proper paragraphs with line breaks\n"
            "- Keep responses well-structured and readable\n"
            "Include citations in your answer by referencing the source URLs provided with each context block."
        )

        if conversation_context:
            system_content += f"\n\nConversation context:\n{conversation_context}"

        response = self._client.chat.completions.create(
            model=self._config.model_name,
            messages=[
                {"role": "system", "content": system_content},
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
        if response.choices:
            text = response.choices[0].message.content or ""
        else:
            text = ""
        text = self._post_process_response(text)
        citations = list(
            dict.fromkeys(r.source_url for r in context_results if r.source_url)
        )
        return Answer(text=text, citations=citations, source="rag")

    def generate_with_citations_stream(
        self,
        query: str,
        context_results: list[RetrievalResult],
        conversation_context: str = "",
    ) -> Generator[str, None, Answer]:
        """Stream answer tokens for *query* with citations."""
        context_block = "\n\n".join(
            f"[Source: {r.source_url}]\n{r.content}" for r in context_results
        )

        system_content = (
            "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
            "Always format your responses clearly using markdown:\n"
            "- Use bullet points for lists\n"
            "- Use **bold** for emphasis\n"
            "- Use headers (## Header) for sections when appropriate\n"
            "- Use proper paragraphs with line breaks\n"
            "- Keep responses well-structured and readable\n"
            "Include citations in your answer by referencing the source URLs provided with each context block."
        )

        if conversation_context:
            system_content += f"\n\nConversation context:\n{conversation_context}"

        response = self._client.chat.completions.create(
            model=self._config.model_name,
            messages=[
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context_block}\n\n"
                        f"Question: {query}"
                    ),
                },
            ],
            temperature=0.3,
            stream=True,
        )

        full_text = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_text += token
                yield token

        full_text = self._post_process_response(full_text)
        citations = list(
            dict.fromkeys(r.source_url for r in context_results if r.source_url)
        )
        return Answer(text=full_text, citations=citations, source="rag")

    def _post_process_response(self, text: str) -> str:
        """Post-process response text to ensure good formatting."""
        if not text:
            return text

        # Fix common formatting issues
        text = text.strip()

        # Ensure proper spacing around headers
        text = re.sub(r'(?<!^)(\n#{1,6}\s)', r'\n\n\1', text, flags=re.MULTILINE)
        text = re.sub(r'(#{1,6}.*?)(\n)(?!\n)', r'\1\n\n', text)

        # Fix bullet points formatting
        text = re.sub(r'\n([*-])\s*([^\n]+)', r'\n\1 \2', text)

        # Fix numbered lists
        text = re.sub(r'\n(\d+\.)\s*([^\n]+)', r'\n\1 \2', text)

        # Clean up multiple newlines but preserve intentional breaks
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Ensure proper spacing after periods in lists
        text = re.sub(r'\.([A-Z])', r'. \1', text)

        return text.strip()
