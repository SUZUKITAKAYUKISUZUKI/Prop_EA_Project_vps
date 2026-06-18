"""Institutional memory synthesis."""
from __future__ import annotations

from typing import Any
from uuid import uuid4


class ExecutiveMemoryEngine:
    def evaluate(
        self,
        *,
        outcomes: list[dict[str, Any]],
        lessons: list[dict[str, Any]],
        effectiveness: dict[str, Any],
    ) -> list[dict[str, Any]]:
        memories: list[dict[str, Any]] = []
        for lesson in lessons:
            memories.append(
                {
                    "memory_id": str(uuid4()),
                    "category": lesson.get("lesson_type") or "lesson",
                    "title": self._title(lesson),
                    "summary": lesson.get("description"),
                    "success_rate": self._category_success_rate(lesson, effectiveness),
                    "confidence": float(lesson.get("confidence") or 70),
                    "lesson": lesson.get("description"),
                }
            )

        for category, accuracy in (effectiveness.get("effectiveness_by_category") or {}).items():
            if not accuracy:
                continue
            memories.append(
                {
                    "memory_id": str(uuid4()),
                    "category": category,
                    "title": category.replace("_", " ").title(),
                    "summary": f"Historical {category} accuracy: {accuracy:.1f}%",
                    "success_rate": accuracy,
                    "confidence": min(100.0, accuracy + 5.0),
                    "lesson": f"Executive {category} track record informs future governance.",
                }
            )
        return memories

    def _title(self, lesson: dict[str, Any]) -> str:
        return str(lesson.get("lesson_type") or "lesson").replace("_", " ").title()

    def _category_success_rate(self, lesson: dict[str, Any], effectiveness: dict[str, Any]) -> float:
        lesson_type = str(lesson.get("lesson_type") or "")
        mapping = {
            "promotion_caution": "promotion_accuracy",
            "portfolio_fit_threshold": "promotion_accuracy",
            "best_practice": "allocation_accuracy",
        }
        key = mapping.get(lesson_type)
        if key:
            return float((effectiveness.get("effectiveness_by_category") or {}).get(key) or 70)
        return float(lesson.get("impact_score") or 70)
