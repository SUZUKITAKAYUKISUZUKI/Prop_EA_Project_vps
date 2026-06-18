"""SQLite persistence for AI Governor Engine."""
from __future__ import annotations

from typing import Any

from src.ai_governor.decision_engine import GovernorDecision
from src.ai_governor.explainability_engine import ExplainabilityEngine
from src.ai_governor.health_monitor import PortfolioHealthSnapshot
from src.ai_governor.recommendation_engine import GovernorRecommendation
from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class DecisionRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns = owns_connection or db is None
        self._explain = ExplainabilityEngine()

    def close(self) -> None:
        if self._owns:
            self._db.close()

    def save_decision(
        self,
        decision: GovernorDecision,
        *,
        created_by: str = "age_engine",
    ) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO governor_decisions (
                timestamp, decision_type, decision, confidence, reason_json,
                profile, state, profile_id, source_state, executed, created_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                decision.decision_type,
                decision.decision,
                decision.confidence,
                self._explain.serialize(decision.reason_json),
                decision.profile_id,
                decision.source_state,
                decision.profile_id,
                decision.source_state,
                1 if decision.executed else 0,
                created_by,
            ),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def save_decisions(
        self,
        decisions: list[GovernorDecision],
        *,
        created_by: str = "age_engine",
    ) -> list[int]:
        return [self.save_decision(d, created_by=created_by) for d in decisions]

    def save_recommendation(self, rec: GovernorRecommendation) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO governor_recommendations (
                timestamp, action, category, priority, recommendation, confidence,
                expected_benefit, expected_risk, reason_json, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                rec.action,
                rec.category,
                rec.priority,
                rec.reason,
                rec.confidence,
                rec.expected_benefit,
                rec.expected_risk,
                self._explain.serialize(rec.reason_json),
                rec.status,
            ),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def save_recommendations(self, recommendations: list[GovernorRecommendation]) -> list[int]:
        return [self.save_recommendation(r) for r in recommendations]

    def save_health_snapshot(self, snapshot: PortfolioHealthSnapshot, *, profile_id: str) -> None:
        self._db.portfolio.execute(
            """
            INSERT INTO governor_health (
                timestamp, health_score, health_status, state, profile,
                profile_id, risk_level, risk_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                snapshot.health_score,
                snapshot.health_status,
                snapshot.state,
                profile_id,
                profile_id,
                snapshot.risk_level,
                snapshot.risk_score,
            ),
        )
        self._db.portfolio.commit()

    def list_decisions(self, *, profile_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if profile_id:
            rows = self._db.query(
                """
                SELECT * FROM governor_decisions
                WHERE profile_id = ? OR profile = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (profile_id, profile_id, limit),
            )
        else:
            rows = self._db.query(
                """
                SELECT * FROM governor_decisions
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            )
        return [_row_to_decision(dict(row)) for row in rows]

    def list_recommendations(
        self,
        *,
        status: str | None = "OPEN",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if status:
            rows = self._db.query(
                """
                SELECT * FROM governor_recommendations
                WHERE status = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (status, limit),
            )
        else:
            rows = self._db.query(
                """
                SELECT * FROM governor_recommendations
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            )
        return [_row_to_recommendation(dict(row)) for row in rows]

    def list_health_snapshots(self, *, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT * FROM governor_health
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in rows]


def _row_to_decision(row: dict[str, Any]) -> dict[str, Any]:
    explain = ExplainabilityEngine()
    row["reason_json"] = explain.deserialize(row.get("reason_json"))
    row.setdefault("profile", row.get("profile_id"))
    row.setdefault("state", row.get("source_state"))
    return row


def _row_to_recommendation(row: dict[str, Any]) -> dict[str, Any]:
    explain = ExplainabilityEngine()
    row["reason_json"] = explain.deserialize(row.get("reason_json"))
    row.setdefault("action", row.get("decision_type") or row.get("category"))
    return row
