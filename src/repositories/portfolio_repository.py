"""Portfolio / WFT / MC / risk attribution repository."""
from __future__ import annotations

import json
from typing import Any

from src.database.db_manager import DatabaseManager
from src.repositories.base import create_default_db_manager
from src.repositories.run_repository import RunRepository


class PortfolioRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns_connection = owns_connection or db is None
        self._runs = RunRepository(self._db)

    def close(self) -> None:
        if self._owns_connection:
            self._db.close()

    def _resolve_run_id(self, *, run_id: int | None, source_path: str | None) -> int | None:
        if run_id is not None:
            return run_id
        if source_path:
            return self._runs.resolve_run_id(source_path=source_path)
        return None

    def get_portfolio_result(
        self,
        *,
        run_id: int | None = None,
        source_path: str | None = None,
        rank: int = 1,
    ) -> dict[str, Any] | None:
        resolved = self._resolve_run_id(run_id=run_id, source_path=source_path)
        sql = "SELECT * FROM portfolio_results WHERE 1=1"
        params: list[Any] = []
        if resolved is not None:
            sql += " AND run_id=?"
            params.append(resolved)
        sql += " AND rank=? ORDER BY id DESC LIMIT 1"
        params.append(rank)
        row = self._db.query(sql, tuple(params), one=True)
        if not row:
            return None
        item = dict(row)
        if item.get("allocation_json"):
            try:
                item["allocation"] = json.loads(item["allocation_json"])
            except json.JSONDecodeError:
                item["allocation"] = {}
        return item

    def get_latest_allocation(
        self,
        *,
        run_id: int | None = None,
        source_path: str | None = None,
    ) -> dict[str, float]:
        result = self.get_portfolio_result(run_id=run_id, source_path=source_path, rank=1)
        if not result:
            return {}
        return dict(result.get("allocation") or {})

    def get_wft_results(self, run_id: int | None = None, source_path: str | None = None) -> list[dict[str, Any]]:
        resolved = self._resolve_run_id(run_id=run_id, source_path=source_path) or self._runs.get_latest_run_id("wft")
        if resolved is None:
            return []
        rows = self._db.query(
            "SELECT * FROM wft_results WHERE run_id=? ORDER BY window_id ASC",
            (resolved,),
        )
        return [dict(r) for r in rows]

    def get_mc_results(self, run_id: int | None = None, source_path: str | None = None) -> list[dict[str, Any]]:
        resolved = self._resolve_run_id(run_id=run_id, source_path=source_path) or self._runs.get_latest_run_id("mc")
        if resolved is None:
            return []
        rows = self._db.query(
            "SELECT * FROM mc_results WHERE run_id=? ORDER BY id ASC",
            (resolved,),
        )
        return [dict(r) for r in rows]

    def get_bt_summary(
        self,
        run_id: int | None = None,
        source_path: str | None = None,
        label: str | None = None,
    ) -> dict[str, Any] | None:
        resolved = self._resolve_run_id(run_id=run_id, source_path=source_path)
        if resolved is None and source_path:
            resolved = self._runs.resolve_run_id(source_path=source_path)
        if resolved is None:
            resolved = self._runs.get_latest_run_id("trade")
        if resolved is None:
            return None
        sql = "SELECT * FROM bt_summary WHERE run_id=?"
        params: list[Any] = [resolved]
        if label:
            sql += " AND label=?"
            params.append(label)
        sql += " ORDER BY summary_id DESC LIMIT 1"
        row = self._db.query(sql, tuple(params), one=True)
        return dict(row) if row else None

    def get_risk_attribution(
        self,
        run_id: int | None = None,
        source_path: str | None = None,
    ) -> list[dict[str, Any]]:
        resolved = self._resolve_run_id(run_id=run_id, source_path=source_path) or self._runs.get_latest_run_id("risk")
        if resolved is None:
            return []
        rows = self._db.query(
            "SELECT * FROM risk_attribution WHERE run_id=? ORDER BY contribution_r DESC",
            (resolved,),
        )
        return [dict(r) for r in rows]
