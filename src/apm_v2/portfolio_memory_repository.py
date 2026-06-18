"""Persistence for APM v2 portfolio memory."""
from __future__ import annotations

import json
from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class PortfolioMemoryRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns = owns_connection or db is None

    def close(self) -> None:
        if self._owns:
            self._db.close()

    def save_outcomes(self, *, profile_id: str, outcomes: list[dict[str, Any]]) -> None:
        for outcome in outcomes:
            self._db.portfolio.execute(
                """
                INSERT INTO executive_decision_outcomes (
                    decision_id, profile_id, decision_type, predicted_benefit, actual_benefit,
                    predicted_risk, actual_risk, success_score, outcome_class, evaluation_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    outcome.get("decision_id"),
                    profile_id,
                    outcome.get("decision_type"),
                    outcome.get("predicted_benefit"),
                    outcome.get("actual_benefit"),
                    outcome.get("predicted_risk"),
                    outcome.get("actual_risk"),
                    outcome.get("success_score"),
                    outcome.get("outcome_class"),
                    outcome.get("evaluation_date"),
                ),
            )
        self._db.portfolio.commit()

    def save_memories(self, *, profile_id: str, memories: list[dict[str, Any]]) -> None:
        ts = utc_now_iso()
        for memory in memories:
            self._db.portfolio.execute(
                """
                INSERT INTO executive_memory (
                    memory_id, profile_id, category, title, summary,
                    success_rate, confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.get("memory_id"),
                    profile_id,
                    memory.get("category"),
                    memory.get("title"),
                    memory.get("summary"),
                    memory.get("success_rate"),
                    memory.get("confidence"),
                    ts,
                ),
            )
        self._db.portfolio.commit()

    def save_lessons(self, *, profile_id: str, lessons: list[dict[str, Any]]) -> None:
        ts = utc_now_iso()
        for idx, lesson in enumerate(lessons):
            self._db.portfolio.execute(
                """
                INSERT INTO executive_lessons (
                    lesson_id, profile_id, source_module, lesson_type,
                    description, impact_score, confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lesson.get("lesson_id") or f"{profile_id}-lesson-{idx}",
                    profile_id,
                    lesson.get("source_module"),
                    lesson.get("lesson_type"),
                    lesson.get("description"),
                    lesson.get("impact_score"),
                    lesson.get("confidence"),
                    ts,
                ),
            )
        self._db.portfolio.commit()

    def load_outcomes(self, *, profile_id: str, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT * FROM executive_decision_outcomes
            WHERE profile_id=?
            ORDER BY evaluation_date DESC, id DESC
            LIMIT ?
            """,
            (profile_id, limit),
        )
        return [dict(row) for row in rows or []]

    def load_memories(self, *, profile_id: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT * FROM executive_memory
            WHERE profile_id=?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (profile_id, limit),
        )
        return [dict(row) for row in rows or []]

    def load_lessons(self, *, profile_id: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT * FROM executive_lessons
            WHERE profile_id=?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (profile_id, limit),
        )
        return [dict(row) for row in rows or []]

    def load_executed_decisions(self, *, profile_id: str, limit: int = 100) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT * FROM apm_execution_queue
            WHERE profile_id=? AND status IN ('EXECUTED', 'APPROVED')
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (profile_id, limit),
        )
        results = []
        for row in rows or []:
            item = dict(row)
            if item.get("payload_json"):
                try:
                    payload = json.loads(item["payload_json"])
                    payload.update({k: v for k, v in item.items() if k != "payload_json"})
                    results.append(payload)
                    continue
                except (TypeError, json.JSONDecodeError):
                    pass
            results.append(item)
        return results
