"""Strategy version genealogy for SLM v3."""
from __future__ import annotations

from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class GenealogyRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns = owns_connection or db is None

    def close(self) -> None:
        if self._owns:
            self._db.close()

    def add_version(
        self,
        strategy_id: str,
        strategy_version: str,
        *,
        parent_strategy_id: str | None = None,
    ) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO strategy_genealogy (strategy_id, parent_strategy_id, strategy_version, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (strategy_id, parent_strategy_id or strategy_id, strategy_version, utc_now_iso()),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def list_genealogy(self, strategy_id: str | None = None) -> list[dict[str, Any]]:
        if strategy_id:
            rows = self._db.query(
                """
                SELECT * FROM strategy_genealogy
                WHERE strategy_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (strategy_id,),
            )
        else:
            rows = self._db.query(
                "SELECT * FROM strategy_genealogy ORDER BY strategy_id ASC, created_at ASC, id ASC"
            )
        return [dict(row) for row in rows]

    def build_tree(self, strategy_id: str | None = None) -> dict[str, Any]:
        rows = self.list_genealogy(strategy_id)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            sid = str(row["strategy_id"])
            grouped.setdefault(sid, []).append(
                {
                    "version": row.get("strategy_version"),
                    "parent_strategy_id": row.get("parent_strategy_id"),
                    "created_at": row.get("created_at"),
                }
            )
        if strategy_id:
            return {"strategy": strategy_id, "versions": grouped.get(strategy_id, [])}
        return {"strategies": grouped}
