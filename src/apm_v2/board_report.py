"""Combined executive board report for APM v2."""
from __future__ import annotations

from typing import Any


class BoardReport:
    def build(
        self,
        *,
        profile_id: str,
        executive_score: dict[str, Any],
        board: dict[str, Any],
        effectiveness: dict[str, Any],
        learning: dict[str, Any],
        memory_snapshot: dict[str, Any],
        feedback: dict[str, Any],
        roadmap_learning: dict[str, Any],
        outcomes: list[dict[str, Any]],
        memories: list[dict[str, Any]],
        lessons: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "profile_id": profile_id,
            "executive_score": executive_score.get("executive_score"),
            "executive_score_v2": executive_score.get("executive_score_v2"),
            "executive_score_components": executive_score.get("components"),
            "board": board,
            "board_consensus": board.get("board_consensus"),
            "decision_accuracy": effectiveness.get("decision_accuracy"),
            "effectiveness": effectiveness,
            "executive_memory": memories,
            "executive_lessons": lessons,
            "decision_outcomes": outcomes,
            "memory_snapshot": memory_snapshot,
            "best_decisions": memory_snapshot.get("best_decisions"),
            "worst_decisions": memory_snapshot.get("worst_decisions"),
            "improvement_opportunities": learning.get("improvement_opportunities"),
            "recommendation_feedback": feedback,
            "roadmap_learning": roadmap_learning,
            "learned_roadmap": roadmap_learning.get("learned_roadmap"),
        }
