"""SQLite repository for account_state_history."""
from __future__ import annotations

import sqlite3
from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class StateHistoryRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns = owns_connection or db is None

    def close(self) -> None:
        if self._owns:
            self._db.close()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._db.portfolio

    def insert_snapshot(
        self,
        *,
        state: str,
        profile: str,
        equity: float | None = None,
        balance: float | None = None,
        drawdown_pct: float | None = None,
        risk_budget_remaining: float | None = None,
        challenge_progress: float | None = None,
        source: str = "auto_switch",
        timestamp: str | None = None,
    ) -> int:
        ts = timestamp or utc_now_iso()
        cur = self.conn.execute(
            """
            INSERT INTO account_state_history (
                timestamp, state, profile, equity, balance,
                drawdown_pct, risk_budget_remaining, challenge_progress, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                state,
                profile,
                equity,
                balance,
                drawdown_pct,
                risk_budget_remaining,
                challenge_progress,
                source,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_latest(self) -> dict[str, Any] | None:
        row = self._db.query(
            """
            SELECT * FROM account_state_history
            ORDER BY timestamp DESC, id DESC
            LIMIT 1
            """,
            one=True,
        )
        return dict(row) if row else None

    def list_history(
        self,
        *,
        limit: int = 5000,
        start: str | None = None,
        end: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["1=1"]
        params: list[Any] = []
        if start:
            clauses.append("timestamp>=?")
            params.append(start)
        if end:
            clauses.append("timestamp<=?")
            params.append(end)
        params.append(limit)
        rows = self._db.query(
            f"""
            SELECT * FROM account_state_history
            WHERE {' AND '.join(clauses)}
            ORDER BY timestamp ASC, id ASC
            LIMIT ?
            """,
            tuple(params),
        )
        return [dict(r) for r in rows]

    def list_recent(self, *, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT * FROM account_state_history
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(r) for r in reversed(rows)]
