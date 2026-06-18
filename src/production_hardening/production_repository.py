"""Persistence for Portfolio OS RC1 production hardening."""
from __future__ import annotations

import json
from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class ProductionRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns = owns_connection or db is None

    def close(self) -> None:
        if self._owns:
            self._db.close()

    def save_readiness(self, *, profile_id: str, report: dict[str, Any]) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO production_readiness (
                timestamp, profile_id, production_readiness, resilience_score,
                production_status, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                profile_id,
                report.get("production_readiness_score"),
                report.get("resilience_score"),
                report.get("production_status"),
                json.dumps(report, ensure_ascii=False),
            ),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def save_validation_history(self, *, profile_id: str, report: dict[str, Any]) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO production_validation_history (
                timestamp, profile_id, end_to_end_score, chain_health, payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                profile_id,
                (report.get("end_to_end") or {}).get("end_to_end_score"),
                (report.get("recommendation_chain") or {}).get("recommendation_chain_health"),
                json.dumps(report.get("validation_results") or {}, ensure_ascii=False),
            ),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def save_benchmark_history(self, *, profile_id: str, benchmarks: dict[str, Any]) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO production_benchmark_history (
                timestamp, profile_id, benchmark_score, payload_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                profile_id,
                benchmarks.get("benchmark_score"),
                json.dumps(benchmarks.get("benchmarks") or {}, ensure_ascii=False),
            ),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def save_failure(self, *, profile_id: str, scenario: str, recovered: bool, payload: dict[str, Any]) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO production_failures (
                timestamp, profile_id, scenario, recovered, payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                profile_id,
                scenario,
                1 if recovered else 0,
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def save_resilience_history(
        self,
        *,
        profile_id: str,
        resilience_score: float,
        failure_recovery: float,
    ) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO production_resilience_history (
                timestamp, profile_id, resilience_score, failure_recovery
            ) VALUES (?, ?, ?, ?)
            """,
            (utc_now_iso(), profile_id, resilience_score, failure_recovery),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def load_latest_readiness(self, *, profile_id: str) -> dict[str, Any] | None:
        row = self._db.query(
            """
            SELECT * FROM production_readiness
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

    def load_benchmark_history(self, *, profile_id: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT * FROM production_benchmark_history
            WHERE profile_id=?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (profile_id, limit),
        )
        return [dict(row) for row in rows or []]
