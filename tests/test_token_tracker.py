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
    assert stats["estimated_cost_usd"] > 0


def test_reset():
    """Reset clears all tracked data."""
    tracker = TokenTracker()
    tracker.record(prompt_tokens=100, completion_tokens=50, model="test")
    tracker.reset()
    assert tracker.stats()["total_tokens"] == 0
