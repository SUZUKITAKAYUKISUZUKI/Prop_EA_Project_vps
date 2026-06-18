"""Memory snapshot packaging for timeline display."""
from __future__ import annotations

from typing import Any


class MemorySnapshotEngine:
    def build(
        self,
        *,
        memories: list[dict[str, Any]],
        lessons: list[dict[str, Any]],
        outcomes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        sorted_outcomes = sorted(outcomes, key=lambda o: float(o.get("success_score") or 0), reverse=True)
        return {
            "timeline": memories,
            "best_decisions": sorted_outcomes[:3],
            "worst_decisions": list(reversed(sorted_outcomes[-3:])) if sorted_outcomes else [],
            "memory_count": len(memories),
            "lesson_count": len(lessons),
            "outcome_count": len(outcomes),
        }
