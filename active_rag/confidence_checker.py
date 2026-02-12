"""AI Model Confidence Check.

Evaluates whether the LLM can answer a user query with high confidence,
or whether there is a risk of hallucination / "don't know" scenario.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI

from active_rag.config import Config

_CONFIDENCE_SYSTEM_PROMPT = (
    "You are a confidence evaluator. Given a user question, estimate how "
    "confident you are that you can provide a correct, factual answer "
    "WITHOUT any external retrieval.\n\n"
    "Respond with ONLY a JSON object in the following format:\n"
    '{"confidence": <float between 0.0 and 1.0>, '
    '"reasoning": "<brief explanation>"}\n\n'
    "A confidence of 1.0 means you are absolutely certain you know the "
    "correct answer. A confidence of 0.0 means you have no idea."
)


@dataclass
class ConfidenceResult:
    """Result of a confidence check."""

    confidence: float
    reasoning: str
    is_high_confidence: bool


class ConfidenceChecker:
    """Checks whether the LLM can answer a query with high confidence."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = OpenAI(api_key=config.openai_api_key)

    def check(self, query: str) -> ConfidenceResult:
        """Return a confidence assessment for the given *query*."""
        response = self._client.chat.completions.create(
            model=self._config.model_name,
            messages=[
                {"role": "system", "content": _CONFIDENCE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
        )
        content = response.choices[0].message.content or "{}"
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {"confidence": 0.0, "reasoning": "Failed to parse response"}

        confidence = float(data.get("confidence", 0.0))
        reasoning = str(data.get("reasoning", ""))
        return ConfidenceResult(
            confidence=confidence,
            reasoning=reasoning,
            is_high_confidence=confidence >= self._config.confidence_threshold,
        )
