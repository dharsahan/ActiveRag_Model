"""Post-generation answer quality evaluation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from openai import OpenAI
from openai import APIConnectionError, RateLimitError, APITimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

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


# Retry decorator for API calls
api_retry = retry(
    retry=retry_if_exception_type((APIConnectionError, RateLimitError, APITimeoutError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=lambda retry_state: logger.warning(
        "Evaluation API call failed, retrying in %s seconds...",
        retry_state.next_action.sleep
    ),
)


class AnswerEvaluator:
    """Evaluates generated answers for quality and relevance."""

    def __init__(self, config: Config) -> None:
        self._client = OpenAI(
            base_url=config.ollama_base_url,
            api_key=config.api_key,
        )
        self._config = config

    @api_retry
    def evaluate(self, query: str, answer: str) -> EvaluationResult:
        """Evaluate the *answer* for the given *query*."""
        response = self._client.chat.completions.create(
            model=self._config.model_name,
            messages=[
                {"role": "system", "content": _EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {query}\n\nAnswer: {answer}"},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {"quality": 0.5, "issues": ["Failed to parse evaluation JSON"], "suggestion": ""}

        quality = float(data.get("quality", 0.5))
        return EvaluationResult(
            quality=quality,
            issues=data.get("issues", []),
            suggestion=data.get("suggestion", ""),
            is_acceptable=quality >= 0.5,
        )
