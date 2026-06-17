"""Feature vector repository."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.database.db_manager import DatabaseManager
from src.repositories.base import create_default_db_manager
from src.repositories.run_repository import RunRepository


class FeatureRepository:
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
        return self._runs.get_latest_run_id("feature")

    def get_features(
        self,
        *,
        run_id: int | None = None,
        source_path: str | None = None,
        strategy: str | None = None,
        trade_id: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        resolved = self._resolve_run_id(run_id=run_id, source_path=source_path)
        sql = "SELECT feature_id, trade_id, run_id, strategy, feature_json, source_key FROM features WHERE 1=1"
        params: list[Any] = []
        if resolved is not None:
            sql += " AND run_id=?"
            params.append(resolved)
        if strategy:
            sql += " AND strategy=?"
            params.append(strategy)
        if trade_id is not None:
            sql += " AND trade_id=?"
            params.append(trade_id)
        sql += " ORDER BY feature_id ASC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = self._db.query(sql, tuple(params))
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["features"] = json.loads(item.pop("feature_json"))
            except json.JSONDecodeError:
                item["features"] = {}
            out.append(item)
        return out

    def get_feature_vector(self, feature_id: int) -> dict[str, Any] | None:
        row = self._db.query(
            "SELECT feature_id, trade_id, run_id, strategy, feature_json, source_key FROM features WHERE feature_id=?",
            (feature_id,),
            one=True,
        )
        if not row:
            return None
        item = dict(row)
        try:
            item["features"] = json.loads(item.pop("feature_json"))
        except json.JSONDecodeError:
            item["features"] = {}
        return item

    def get_bayes_dataset(
        self,
        *,
        run_id: int | None = None,
        source_path: str | None = None,
        strategy: str | None = None,
    ) -> pd.DataFrame:
        records = self.get_features(run_id=run_id, source_path=source_path, strategy=strategy)
        if not records:
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        for rec in records:
            payload = dict(rec.get("features") or {})
            payload["feature_id"] = rec["feature_id"]
            payload["trade_id"] = rec.get("trade_id")
            payload["run_id"] = rec["run_id"]
            payload["strategy"] = rec.get("strategy")
            rows.append(payload)
        return pd.DataFrame(rows)

    def count_features(self, run_id: int | None = None) -> int:
        if run_id is None:
            row = self._db.query("SELECT COUNT(*) AS c FROM features", one=True)
        else:
            row = self._db.query("SELECT COUNT(*) AS c FROM features WHERE run_id=?", (run_id,), one=True)
        return int(row["c"]) if row else 0
