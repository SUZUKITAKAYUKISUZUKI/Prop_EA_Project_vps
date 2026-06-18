"""Auto Switch Engine — profile repository for switch event log."""
from __future__ import annotations

import sqlite3
from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class AutoSwitchRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns = owns_connection or db is None

    def close(self) -> None:
        if self._owns:
            self._db.close()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._db.portfolio

    def log_switch(
        self,
        *,
        old_profile: str | None,
        new_profile: str,
        account_state: str,
        reason: str,
        equity: float | None = None,
        dd: float | None = None,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO profile_switch_log (
                timestamp, old_profile, new_profile, account_state, reason, equity, dd
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                old_profile,
                new_profile,
                account_state,
                reason,
                equity,
                dd,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def recent_switches(self, *, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT id, timestamp, old_profile, new_profile, account_state, reason, equity, dd
            FROM profile_switch_log
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(r) for r in rows]
