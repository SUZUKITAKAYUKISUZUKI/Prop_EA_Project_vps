"""Persistence for CIO Intelligence Layer v1."""
from __future__ import annotations

import json
from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class IntelligenceRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns = owns_connection or db is None

    def close(self) -> None:
        if self._owns:
            self._db.close()

    def save_snapshot(self, *, profile_id: str, report: dict[str, Any]) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO cio_intelligence_snapshots (
                timestamp, profile_id, executive_score, investment_state_json, payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                profile_id,
                report.get("executive_score"),
                json.dumps(report.get("investment_state") or [], ensure_ascii=False),
                json.dumps(report, ensure_ascii=False),
            ),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def save_executive_score(
        self,
        *,
        profile_id: str,
        executive_score: float,
        opportunity_score: float,
        risk_score: float,
        confidence_score: float,
        capital_efficiency: float,
    ) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO executive_investment_scores (
                timestamp, profile_id, executive_score, opportunity_score,
                risk_score, confidence_score, capital_efficiency
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                profile_id,
                executive_score,
                opportunity_score,
                risk_score,
                confidence_score,
                capital_efficiency,
            ),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def load_latest_snapshot(self, *, profile_id: str) -> dict[str, Any] | None:
        row = self._db.query(
            """
            SELECT * FROM cio_intelligence_snapshots
            WHERE profile_id=?
            ORDER BY timestamp DESC, id DESC
            LIMIT 1
            """,
            (profile_id,),
            one=True,
        )
        if not row:
            return None
        item = dict(row)
        if item.get("payload_json"):
            try:
                return json.loads(item["payload_json"])
            except (TypeError, json.JSONDecodeError):
                pass
        return item

    def load_score_history(self, *, profile_id: str, limit: int = 30) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT * FROM executive_investment_scores
            WHERE profile_id=?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (profile_id, limit),
        )
        return [dict(row) for row in rows or []]
