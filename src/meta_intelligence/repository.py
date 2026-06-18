"""Persistence for Meta Intelligence Engine v1."""
from __future__ import annotations

import json
from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class MetaIntelligenceRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns = owns_connection or db is None

    def close(self) -> None:
        if self._owns:
            self._db.close()

    def save_trust_scores(self, *, profile_id: str, trust_scores: dict[str, dict[str, Any]]) -> None:
        ts = utc_now_iso()
        for module, data in trust_scores.items():
            self._db.portfolio.execute(
                """
                INSERT INTO module_trust_history (
                    timestamp, profile_id, module, trust_score, category
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (ts, profile_id, module, data.get("trust_score"), data.get("category")),
            )
        self._db.portfolio.commit()

    def save_rankings(self, *, profile_id: str, rankings: list[dict[str, Any]]) -> None:
        ts = utc_now_iso()
        for row in rankings:
            self._db.portfolio.execute(
                """
                INSERT INTO module_rankings (
                    timestamp, profile_id, module, rank, trust_score
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (ts, profile_id, row.get("module"), row.get("rank"), row.get("trust_score")),
            )
        self._db.portfolio.commit()

    def save_drift_alerts(self, *, profile_id: str, alerts: list[dict[str, Any]]) -> None:
        ts = utc_now_iso()
        for alert in alerts:
            self._db.portfolio.execute(
                """
                INSERT INTO module_drift_alerts (
                    timestamp, profile_id, module, previous_score, current_score, delta, alert_code
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    profile_id,
                    alert.get("module"),
                    alert.get("previous_score"),
                    alert.get("current_score"),
                    alert.get("delta"),
                    alert.get("alert_code"),
                ),
            )
        self._db.portfolio.commit()

    def save_improvement_notes(self, *, profile_id: str, notes: list[dict[str, Any]]) -> None:
        ts = utc_now_iso()
        for note in notes:
            self._db.portfolio.execute(
                """
                INSERT INTO mie_self_improvement_notes (
                    timestamp, profile_id, module, issue, recommendation, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    profile_id,
                    note.get("module"),
                    note.get("issue"),
                    note.get("recommendation"),
                    json.dumps(note, ensure_ascii=False),
                ),
            )
        self._db.portfolio.commit()

    def load_trust_history(self, *, profile_id: str, limit: int = 500) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT * FROM module_trust_history
            WHERE profile_id=?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (profile_id, limit),
        )
        return [dict(row) for row in rows or []]

    def load_latest_rankings(self, *, profile_id: str) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT module, rank, trust_score, timestamp
            FROM module_rankings mr
            INNER JOIN (
                SELECT MAX(timestamp) AS max_ts
                FROM module_rankings
                WHERE profile_id=?
            ) latest ON mr.timestamp = latest.max_ts
            WHERE mr.profile_id=?
            ORDER BY rank ASC
            """,
            (profile_id, profile_id),
        )
        return [dict(row) for row in rows or []]

    def load_drift_alerts(self, *, profile_id: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT * FROM module_drift_alerts
            WHERE profile_id=?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (profile_id, limit),
        )
        return [dict(row) for row in rows or []]

    def load_improvement_notes(self, *, profile_id: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT module, issue, recommendation, payload_json, timestamp
            FROM mie_self_improvement_notes
            WHERE profile_id=?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (profile_id, limit),
        )
        notes: list[dict[str, Any]] = []
        for row in rows or []:
            item = dict(row)
            if item.get("payload_json"):
                try:
                    item = {**json.loads(item["payload_json"]), **item}
                except (TypeError, json.JSONDecodeError):
                    pass
            notes.append(item)
        return notes
