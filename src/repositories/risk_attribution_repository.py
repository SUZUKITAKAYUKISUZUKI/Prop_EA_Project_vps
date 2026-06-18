"""SQLite persistence for Portfolio Risk Attribution reports."""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class RiskAttributionRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns_connection = owns_connection or db is None

    def close(self) -> None:
        if self._owns_connection:
            self._db.close()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._db.portfolio

    def save_report(
        self,
        *,
        report_id: str,
        source_run_id: str | None,
        profile_id: str,
        total_r: float,
        total_dd: float,
        pf: float,
        win_rate: float,
        report_json: dict[str, Any],
    ) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO risk_attribution_reports (
                report_id, source_run_id, profile_id, generated_at,
                total_r, total_dd, pf, win_rate, report_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report_id,
                source_run_id,
                profile_id,
                utc_now_iso(),
                total_r,
                total_dd,
                pf,
                win_rate,
                json.dumps(report_json, ensure_ascii=False),
            ),
        )
        self.conn.commit()

    def get_report(self, report_id: str) -> dict[str, Any] | None:
        row = self._db.query(
            "SELECT * FROM risk_attribution_reports WHERE report_id=?",
            (report_id,),
            one=True,
        )
        if row is None:
            return None
        return self._hydrate(dict(row))

    def get_latest_report(self, *, profile_id: str | None = None) -> dict[str, Any] | None:
        if profile_id:
            row = self._db.query(
                """
                SELECT * FROM risk_attribution_reports
                WHERE profile_id=?
                ORDER BY generated_at DESC
                LIMIT 1
                """,
                (profile_id,),
                one=True,
            )
        else:
            row = self._db.query(
                """
                SELECT * FROM risk_attribution_reports
                ORDER BY generated_at DESC
                LIMIT 1
                """,
                one=True,
            )
        if row is None:
            return None
        return self._hydrate(dict(row))

    def list_reports(self, *, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT report_id, source_run_id, profile_id, generated_at,
                   total_r, total_dd, pf, win_rate
            FROM risk_attribution_reports
            ORDER BY generated_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(r) for r in rows]

    def get_cache(self, cache_key: str) -> dict[str, Any] | None:
        row = self._db.query(
            "SELECT cache_value FROM risk_attribution_cache WHERE cache_key=?",
            (cache_key,),
            one=True,
        )
        if row is None:
            return None
        try:
            return json.loads(str(row["cache_value"]))
        except json.JSONDecodeError:
            return None

    def set_cache(self, cache_key: str, value: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO risk_attribution_cache (cache_key, cache_value, updated_at)
            VALUES (?, ?, ?)
            """,
            (cache_key, json.dumps(value, ensure_ascii=False), utc_now_iso()),
        )
        self.conn.commit()

    @staticmethod
    def _hydrate(row: dict[str, Any]) -> dict[str, Any]:
        payload = dict(row)
        raw = payload.pop("report_json", "{}")
        try:
            payload["report"] = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            payload["report"] = {}
        return payload
