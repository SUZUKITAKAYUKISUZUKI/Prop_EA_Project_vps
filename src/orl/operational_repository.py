"""Persistence for Operational Readiness Layer v1."""
from __future__ import annotations

import json
from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class OperationalRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns = owns_connection or db is None

    def close(self) -> None:
        if self._owns:
            self._db.close()

    def save_readiness(self, *, profile_id: str, report: dict[str, Any]) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO operational_readiness (
                timestamp, profile_id, readiness_score, readiness_status, payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                profile_id,
                report.get("readiness_score"),
                report.get("readiness_status"),
                json.dumps(report, ensure_ascii=False),
            ),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def save_audit_log(self, *, profile_id: str, category: str, message: str, severity: str = "info") -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO operational_audit_log (
                timestamp, profile_id, category, severity, message
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (utc_now_iso(), profile_id, category, severity, message),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def save_health_snapshot(
        self,
        *,
        profile_id: str,
        system_health: float,
        readiness_score: float,
    ) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO system_health_history (
                timestamp, profile_id, system_health, readiness_score
            ) VALUES (?, ?, ?, ?)
            """,
            (utc_now_iso(), profile_id, system_health, readiness_score),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def load_latest_readiness(self, *, profile_id: str) -> dict[str, Any] | None:
        row = self._db.query(
            """
            SELECT * FROM operational_readiness
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

    def load_audit_log(self, *, profile_id: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT * FROM operational_audit_log
            WHERE profile_id=?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (profile_id, limit),
        )
        return [dict(row) for row in rows or []]

    def load_health_history(self, *, profile_id: str, limit: int = 30) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT * FROM system_health_history
            WHERE profile_id=?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (profile_id, limit),
        )
        return [dict(row) for row in rows or []]

    def historical_stability_score(self, *, profile_id: str) -> float:
        rows = self.load_health_history(profile_id=profile_id, limit=10)
        if len(rows) < 2:
            return 85.0
        scores = [float(r.get("readiness_score") or 0) for r in rows]
        if not scores:
            return 85.0
        avg = sum(scores) / len(scores)
        variance = sum((s - avg) ** 2 for s in scores) / len(scores)
        stability = max(0.0, 100.0 - variance)
        return round(min(100.0, stability), 2)
