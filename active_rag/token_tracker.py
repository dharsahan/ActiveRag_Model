"""Token usage tracking and cost estimation."""

from __future__ import annotations

from dataclasses import dataclass


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
