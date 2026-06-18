"""Query helpers for governor decision history."""
from __future__ import annotations

from typing import Any

from src.ai_governor.decision_repository import DecisionRepository


class DecisionHistory:
    def __init__(self, repo: DecisionRepository | None = None, *, owns_connection: bool = False) -> None:
        self._repo = repo or DecisionRepository(owns_connection=owns_connection)
        self._owns = owns_connection or repo is None

    def close(self) -> None:
        if self._owns:
            self._repo.close()

    def recent_decisions(
        self,
        *,
        profile_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return self._repo.list_decisions(profile_id=profile_id, limit=limit)

    def timeline(
        self,
        *,
        profile_id: str | None = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        rows = self.recent_decisions(profile_id=profile_id, limit=limit)
        return [
            {
                "timestamp": row.get("timestamp"),
                "decision_type": row.get("decision_type"),
                "decision": row.get("decision"),
                "confidence": row.get("confidence"),
                "profile_id": row.get("profile_id"),
                "reason": row.get("reason_json"),
            }
            for row in rows
        ]

    def open_recommendations(self, *, limit: int = 50) -> list[dict[str, Any]]:
        return self._repo.list_recommendations(status="OPEN", limit=limit)

    def health_history(self, *, limit: int = 20) -> list[dict[str, Any]]:
        return self._repo.list_health_snapshots(limit=limit)
