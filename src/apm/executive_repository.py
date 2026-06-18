"""Persistence for APM v1 executive layer."""
from __future__ import annotations

import json
from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class ExecutiveRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns = owns_connection or db is None

    def close(self) -> None:
        if self._owns:
            self._db.close()

    def save_execution_queue(self, *, profile_id: str, actions: list[dict[str, Any]]) -> None:
        ts = utc_now_iso()
        for action in actions:
            self._db.portfolio.execute(
                """
                INSERT INTO apm_execution_queue (
                    action_id, timestamp, profile_id, action_type, strategy,
                    confidence, expected_benefit_pct, expected_risk_pct,
                    status, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    action.get("action_id"),
                    ts,
                    profile_id,
                    action.get("action_type"),
                    action.get("strategy"),
                    action.get("confidence"),
                    action.get("expected_benefit_pct"),
                    action.get("expected_risk_pct"),
                    action.get("status"),
                    json.dumps(action, ensure_ascii=False),
                ),
            )
        self._db.portfolio.commit()

    def update_action_status(self, *, action_id: str, status: str, reason: str = "") -> None:
        self._db.portfolio.execute(
            """
            UPDATE apm_execution_queue
            SET status=?, rejection_reason=?
            WHERE action_id=?
            """,
            (status, reason or None, action_id),
        )
        self._db.portfolio.commit()

    def load_execution_queue(self, *, profile_id: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT * FROM apm_execution_queue
            WHERE profile_id=?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (profile_id, limit),
        )
        return [self._row_to_action(dict(row)) for row in rows or []]

    def load_action(self, *, action_id: str) -> dict[str, Any] | None:
        row = self._db.query(
            "SELECT * FROM apm_execution_queue WHERE action_id=?",
            (action_id,),
            one=True,
        )
        return self._row_to_action(dict(row)) if row else None

    def save_executive_report(self, *, profile_id: str, report: dict[str, Any]) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO apm_executive_reports (
                timestamp, profile_id, executive_score, executive_category, payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                profile_id,
                report.get("executive_score"),
                report.get("executive_category"),
                json.dumps(report, ensure_ascii=False),
            ),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def save_roadmap(self, *, profile_id: str, roadmap: list[dict[str, Any]]) -> None:
        ts = utc_now_iso()
        for item in roadmap:
            self._db.portfolio.execute(
                """
                INSERT INTO apm_roadmaps (
                    timestamp, profile_id, horizon, action_type, strategy,
                    description, confidence, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    profile_id,
                    item.get("horizon"),
                    item.get("action_type"),
                    item.get("strategy"),
                    item.get("description"),
                    item.get("confidence"),
                    item.get("status"),
                ),
            )
        self._db.portfolio.commit()

    def save_opportunities(self, *, profile_id: str, opportunities: list[dict[str, Any]]) -> None:
        ts = utc_now_iso()
        for opp in opportunities:
            self._db.portfolio.execute(
                """
                INSERT INTO apm_opportunities (
                    timestamp, profile_id, strategy, portfolio_fit, lifecycle_score,
                    current_allocation_pct, recommended_allocation_pct, message, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    profile_id,
                    opp.get("strategy"),
                    opp.get("portfolio_fit"),
                    opp.get("lifecycle_score"),
                    opp.get("current_allocation_pct"),
                    opp.get("recommended_allocation_pct"),
                    opp.get("message"),
                    json.dumps(opp, ensure_ascii=False),
                ),
            )
        self._db.portfolio.commit()

    def save_risk_alerts(self, *, profile_id: str, alerts: list[dict[str, Any]]) -> None:
        ts = utc_now_iso()
        for alert in alerts:
            self._db.portfolio.execute(
                """
                INSERT INTO apm_risk_alerts (
                    timestamp, profile_id, strategy, risk_score,
                    dd_contribution_pct, health_impact, message, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    profile_id,
                    alert.get("strategy"),
                    alert.get("risk_score"),
                    alert.get("dd_contribution_pct"),
                    alert.get("health_impact"),
                    alert.get("message"),
                    json.dumps(alert, ensure_ascii=False),
                ),
            )
        self._db.portfolio.commit()

    def load_latest_roadmap(self, *, profile_id: str) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT horizon, action_type, strategy, description, confidence, status, timestamp
            FROM apm_roadmaps
            WHERE profile_id=?
            ORDER BY timestamp DESC, id DESC
            LIMIT 20
            """,
            (profile_id,),
        )
        return [dict(row) for row in rows or []]

    def load_opportunities(self, *, profile_id: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._db.query(
            "SELECT * FROM apm_opportunities WHERE profile_id=? ORDER BY timestamp DESC LIMIT ?",
            (profile_id, limit),
        )
        return [dict(row) for row in rows or []]

    def load_risk_alerts(self, *, profile_id: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._db.query(
            "SELECT * FROM apm_risk_alerts WHERE profile_id=? ORDER BY timestamp DESC LIMIT ?",
            (profile_id, limit),
        )
        return [dict(row) for row in rows or []]

    def _row_to_action(self, row: dict[str, Any]) -> dict[str, Any]:
        if row.get("payload_json"):
            try:
                payload = json.loads(row["payload_json"])
                payload.update({k: v for k, v in row.items() if k != "payload_json"})
                return payload
            except (TypeError, json.JSONDecodeError):
                pass
        return row
