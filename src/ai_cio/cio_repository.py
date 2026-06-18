"""Persistence for AI CIO v1."""
from __future__ import annotations

import json
from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class CioRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns = owns_connection or db is None

    def close(self) -> None:
        if self._owns:
            self._db.close()

    def save_report(self, *, profile_id: str, report: dict[str, Any]) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO cio_reports (
                timestamp, profile_id, cio_score, cio_opinion, payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                profile_id,
                report.get("cio_score"),
                report.get("cio_opinion"),
                json.dumps(report, ensure_ascii=False),
            ),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def save_opinion(self, *, profile_id: str, opinion: str, cio_score: float) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO cio_opinions (timestamp, profile_id, cio_opinion, cio_score)
            VALUES (?, ?, ?, ?)
            """,
            (utc_now_iso(), profile_id, opinion, cio_score),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def save_recommendations(self, *, profile_id: str, recommendations: list[dict[str, Any]]) -> int:
        count = 0
        ts = utc_now_iso()
        for rec in recommendations:
            self._db.portfolio.execute(
                """
                INSERT INTO cio_recommendations (
                    timestamp, profile_id, category, priority, action,
                    description, confidence, requires_approval, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    profile_id,
                    rec.get("category"),
                    rec.get("priority"),
                    rec.get("action"),
                    rec.get("description"),
                    rec.get("confidence"),
                    1 if rec.get("requires_approval", True) else 0,
                    json.dumps(rec, ensure_ascii=False),
                ),
            )
            count += 1
        self._db.portfolio.commit()
        return count

    def load_latest_report(self, *, profile_id: str) -> dict[str, Any] | None:
        row = self._db.query(
            """
            SELECT * FROM cio_reports
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

    def load_opinion_history(self, *, profile_id: str, limit: int = 30) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT * FROM cio_opinions
            WHERE profile_id=?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (profile_id, limit),
        )
        return [dict(row) for row in rows or []]

    def load_recommendation_history(self, *, profile_id: str, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT * FROM cio_recommendations
            WHERE profile_id=?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (profile_id, limit),
        )
        return [dict(row) for row in rows or []]
