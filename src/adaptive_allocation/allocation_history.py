"""Adaptive allocation history repository."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class AllocationHistoryRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns = owns_connection or db is None

    def close(self) -> None:
        if self._owns:
            self._db.close()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._db.portfolio

    def log_change(
        self,
        *,
        profile_id: str,
        strategy: str,
        old_weight: float,
        new_weight: float,
        reason: str,
        risk_score: float | None = None,
        profit_score: float | None = None,
        health_score: float | None = None,
        timestamp: str | None = None,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO adaptive_allocation_history (
                timestamp, profile_id, strategy, old_weight, new_weight,
                reason, risk_score, profit_score, health_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp or utc_now_iso(),
                profile_id,
                strategy,
                old_weight,
                new_weight,
                reason,
                risk_score,
                profit_score,
                health_score,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def last_rebalance_timestamp(self, profile_id: str) -> str | None:
        row = self._db.query(
            """
            SELECT timestamp FROM adaptive_allocation_history
            WHERE profile_id=?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (profile_id,),
            one=True,
        )
        return str(row["timestamp"]) if row else None

    def days_since_last_rebalance(self, profile_id: str) -> float | None:
        ts = self.last_rebalance_timestamp(profile_id)
        if not ts:
            return None
        try:
            normalized = ts.replace("Z", "+00:00")
            last = datetime.fromisoformat(normalized)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return (now - last).total_seconds() / 86400.0
        except ValueError:
            return None

    def recent_changes(self, *, profile_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if profile_id:
            rows = self._db.query(
                """
                SELECT * FROM adaptive_allocation_history
                WHERE profile_id=?
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (profile_id, limit),
            )
        else:
            rows = self._db.query(
                """
                SELECT * FROM adaptive_allocation_history
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            )
        return [dict(r) for r in rows]
