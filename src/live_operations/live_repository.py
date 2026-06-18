"""Persistence for RC2 Live Operations Layer."""
from __future__ import annotations

import json
from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class LiveRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns = owns_connection or db is None

    def close(self) -> None:
        if self._owns:
            self._db.close()

    def save_briefing(self, *, profile_id: str, briefing: dict[str, Any]) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO daily_briefings (
                timestamp, profile_id, briefing_date, operational_score,
                live_readiness, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                profile_id,
                briefing.get("date"),
                briefing.get("operational_score"),
                briefing.get("readiness"),
                json.dumps(briefing, ensure_ascii=False),
            ),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def save_digest(self, *, profile_id: str, digest: dict[str, Any]) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO daily_digests (
                timestamp, profile_id, digest_date, payload_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                profile_id,
                (digest.get("digest_summary") or {}).get("date") or digest.get("date"),
                json.dumps(digest, ensure_ascii=False),
            ),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def save_alert(self, *, profile_id: str, level: str, message: str, payload: dict[str, Any]) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO operational_alerts (
                timestamp, profile_id, level, message, payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (utc_now_iso(), profile_id, level, message, json.dumps(payload, ensure_ascii=False)),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def save_anomaly(self, *, profile_id: str, anomaly: dict[str, Any]) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO anomaly_history (
                timestamp, profile_id, anomaly_type, severity, payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                profile_id,
                anomaly.get("type"),
                anomaly.get("severity"),
                json.dumps(anomaly, ensure_ascii=False),
            ),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def save_operations_history(
        self,
        *,
        profile_id: str,
        operational_score: float,
        live_readiness: float,
        user_action_load: int,
        report: dict[str, Any],
    ) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO live_operations_history (
                timestamp, profile_id, operational_score, live_readiness,
                user_action_load, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                profile_id,
                operational_score,
                live_readiness,
                user_action_load,
                json.dumps(report, ensure_ascii=False),
            ),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def load_latest_briefing(self, *, profile_id: str) -> dict[str, Any] | None:
        row = self._db.query(
            """
            SELECT * FROM daily_briefings
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

    def load_latest_digest(self, *, profile_id: str) -> dict[str, Any] | None:
        row = self._db.query(
            """
            SELECT * FROM daily_digests
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

    def load_alerts(self, *, profile_id: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT * FROM operational_alerts
            WHERE profile_id=?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (profile_id, limit),
        )
        return [dict(row) for row in rows or []]

    def load_briefing_history(self, *, profile_id: str, limit: int = 14) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT operational_score, live_readiness, briefing_date, timestamp
            FROM daily_briefings
            WHERE profile_id=?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (profile_id, limit),
        )
        return [dict(row) for row in rows or []]

    def historical_reliability_score(self, *, profile_id: str) -> float:
        rows = self.load_briefing_history(profile_id=profile_id, limit=10)
        if len(rows) < 2:
            return 88.0
        scores = [float(r.get("operational_score") or r.get("live_readiness") or 0) for r in rows]
        avg = sum(scores) / len(scores)
        variance = sum((s - avg) ** 2 for s in scores) / len(scores)
        return round(min(100.0, max(0.0, avg - variance * 0.5)), 2)
