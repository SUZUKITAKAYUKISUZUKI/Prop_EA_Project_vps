"""Confidence history management for CACE v1.5."""
from __future__ import annotations

from typing import Any

from src.cace.confidence_repository import ConfidenceRepository
from src.cace.confidence_trend_engine import ConfidenceTrendEngine


class ConfidenceHistoryEngine:
    def __init__(
        self,
        *,
        repo: ConfidenceRepository | None = None,
        trend: ConfidenceTrendEngine | None = None,
    ) -> None:
        self._repo = repo or ConfidenceRepository(owns_connection=False)
        self._trend = trend or ConfidenceTrendEngine()

    def list_history(self, *, profile_id: str, limit: int = 90) -> list[dict[str, Any]]:
        return self._repo.list_confidence_history(profile_id=profile_id, limit=limit)

    def record(
        self,
        *,
        profile_id: str,
        confidence: float,
        category: str,
        trend: str,
        trend_strength: float,
        snapshot_json: dict[str, Any],
    ) -> int:
        return self._repo.save_confidence_history(
            profile_id=profile_id,
            confidence=confidence,
            category=category,
            trend=trend,
            trend_strength=trend_strength,
            snapshot_json=snapshot_json,
        )

    def build_timeline(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        timeline = []
        for row in history:
            timeline.append(
                {
                    "timestamp": row.get("timestamp"),
                    "confidence": row.get("confidence"),
                    "category": row.get("category"),
                    "trend": row.get("trend"),
                    "trend_strength": row.get("trend_strength"),
                }
            )
        return timeline

    def stability_score(self, history: list[dict[str, Any]]) -> float:
        if len(history) < 2:
            return 75.0
        scores = [float(r.get("confidence") or 0) for r in history[:30]]
        spread = max(scores) - min(scores)
        return round(max(0.0, min(100.0, 100.0 - spread)), 1)
