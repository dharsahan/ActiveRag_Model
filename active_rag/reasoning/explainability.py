"""Explainability system for hybrid RAG answers.

Converts graph reasoning results into human-readable explanations
with confidence breakdowns and source attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from active_rag.reasoning.reasoning_engine import ReasoningResult, ReasoningPath
from active_rag.routing.result_combiner import CombinedResult


@dataclass
class ExplanationResult:
    """Structured explanation attached to a pipeline answer."""
    reasoning_text: str           # Human-readable reasoning chain
    confidence_explanation: str   # Why we're this confident
    path_visualization: str       # ASCII path diagram
    source_breakdown: dict        # {"vector": N, "graph": M}
    strategy_used: str            # "vector", "graph", "hybrid"
    top_paths: List[str]          # Top reasoning path strings


class ExplainabilityFormatter:
    """Formats reasoning results into human-readable explanations."""

    def format_reasoning(
        self,
        reasoning: Optional[ReasoningResult] = None,
        combined: Optional[CombinedResult] = None,
        strategy: str = "unknown",
    ) -> ExplanationResult:
        """Build a structured explanation from reasoning and retrieval results.

        Args:
            reasoning: Output from ReasoningEngine (may be None for vector-only)
            combined: Output from ResultCombiner
            strategy: Retrieval strategy used ("vector", "graph", "hybrid")

        Returns:
            ExplanationResult with all explanation components
        """
        reasoning_text = self._format_reasoning_chain(reasoning)
        confidence_explanation = self._format_confidence(reasoning, combined, strategy)
        path_visualization = self._format_path_diagram(reasoning)
        source_breakdown = self._format_source_breakdown(combined)
        top_paths = self._extract_top_paths(reasoning)

        return ExplanationResult(
            reasoning_text=reasoning_text,
            confidence_explanation=confidence_explanation,
            path_visualization=path_visualization,
            source_breakdown=source_breakdown,
            strategy_used=strategy,
            top_paths=top_paths,
        )

    def _format_reasoning_chain(self, reasoning: Optional[ReasoningResult]) -> str:
        """Convert reasoning paths into a readable chain."""
        if not reasoning or not reasoning.ranked_paths:
            return "Answer derived from semantic similarity search (no graph reasoning applied)."

        lines = [f"**Reasoning ({len(reasoning.ranked_paths)} path(s) found):**"]
        for i, path in enumerate(reasoning.ranked_paths[:3], 1):
            score_pct = f"{path.score * 100:.0f}%"
            lines.append(f"  {i}. {path.reasoning_text} (relevance: {score_pct})")

        if reasoning.supporting_entities:
            names = [e.get("name", "?") for e in reasoning.supporting_entities[:5]]
            lines.append(f"\n**Supporting entities:** {', '.join(names)}")

        return "\n".join(lines)

    def _format_confidence(
        self,
        reasoning: Optional[ReasoningResult],
        combined: Optional[CombinedResult],
        strategy: str,
    ) -> str:
        """Explain why we're confident (or not) in the answer."""
        parts = [f"**Strategy:** {strategy}"]

        if reasoning:
            conf_pct = f"{reasoning.confidence * 100:.0f}%"
            parts.append(f"**Graph confidence:** {conf_pct}")

            if reasoning.confidence >= 0.7:
                parts.append("Strong graph support — multiple relevant paths found.")
            elif reasoning.confidence >= 0.4:
                parts.append("Moderate graph support — some paths found but limited coverage.")
            else:
                parts.append("Weak graph support — few or no relevant paths in the knowledge graph.")

        if combined:
            parts.append(
                f"**Sources used:** {combined.vector_count} vector chunk(s), "
                f"{combined.graph_count} graph result(s)"
            )

        if strategy == "vector":
            parts.append("Answer based on semantic similarity search only.")
        elif strategy == "hybrid":
            parts.append("Answer combines vector similarity and graph reasoning.")

        return "\n".join(parts)

    def _format_path_diagram(self, reasoning: Optional[ReasoningResult]) -> str:
        """Build an ASCII path visualization."""
        if not reasoning or not reasoning.ranked_paths:
            return "No graph paths to visualize."

        lines = []
        for i, path in enumerate(reasoning.ranked_paths[:3], 1):
            if not path.nodes:
                continue

            # Build ASCII diagram
            diagram_parts = []
            for j, node in enumerate(path.nodes):
                name = node.get("name", node.get("id", "?"))
                diagram_parts.append(f"[{name}]")
                if j < len(path.relationships):
                    rel = path.relationships[j]
                    diagram_parts.append(f" --{rel}--> ")

            lines.append(f"Path {i}: {''.join(diagram_parts)}")

        return "\n".join(lines) if lines else "No graph paths to visualize."

    def _format_source_breakdown(self, combined: Optional[CombinedResult]) -> dict:
        """Break down how many results came from each source."""
        if not combined:
            return {"vector": 0, "graph": 0}
        return {
            "vector": combined.vector_count,
            "graph": combined.graph_count,
        }

    def _extract_top_paths(self, reasoning: Optional[ReasoningResult]) -> List[str]:
        """Extract top reasoning path strings."""
        if not reasoning:
            return []
        return [p.reasoning_text for p in reasoning.ranked_paths[:5] if p.reasoning_text]
