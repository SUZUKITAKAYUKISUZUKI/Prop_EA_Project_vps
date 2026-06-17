"""Run metadata repository."""
from __future__ import annotations

import json
from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class RunRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns_connection = owns_connection or db is None

    def close(self) -> None:
        if self._owns_connection:
            self._db.close()

    def create_run(
        self,
        run_type: str,
        *,
        strategy: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> int:
        return self._db.insert_run(
            run_type,
            strategy=strategy,
            description=description,
            parameters=parameters,
            created_at=utc_now_iso(),
        )

    def get_run(self, run_id: int) -> dict[str, Any] | None:
        row = self._db.query(
            """
            SELECT run_id, run_type, strategy, created_at, description, parameters_json
            FROM runs WHERE run_id=?
            """,
            (run_id,),
            one=True,
        )
        return dict(row) if row else None

    def get_runs(
        self,
        *,
        run_type: str | None = None,
        strategy: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        sql = """
            SELECT run_id, run_type, strategy, created_at, description, parameters_json
            FROM runs
            WHERE 1=1
        """
        params: list[Any] = []
        if run_type:
            sql += " AND run_type=?"
            params.append(run_type)
        if strategy:
            sql += " AND strategy=?"
            params.append(strategy)
        sql += " ORDER BY run_id DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        rows = self._db.query(sql, tuple(params))
        return [dict(r) for r in rows]

    def resolve_run_id(
        self,
        *,
        run_id: int | None = None,
        source_path: str | None = None,
        description: str | None = None,
    ) -> int | None:
        if run_id is not None:
            return run_id
        if source_path:
            rel = source_path.replace("\\", "/")
            row = self._db.query(
                "SELECT run_id FROM import_registry WHERE source_path=? LIMIT 1",
                (rel,),
                one=True,
            )
            if row:
                return int(row["run_id"])
            row = self._db.query(
                "SELECT run_id FROM runs WHERE description=? ORDER BY run_id DESC LIMIT 1",
                (rel,),
                one=True,
            )
            if row:
                return int(row["run_id"])
            basename = rel.rsplit("/", 1)[-1]
            rows = self._db.query(
                "SELECT run_id, source_path FROM import_registry WHERE source_path LIKE ?",
                (f"%/{basename}",),
            )
            if len(rows) == 1:
                return int(rows[0]["run_id"])
        if description:
            row = self._db.query(
                "SELECT run_id FROM runs WHERE description=? ORDER BY run_id DESC LIMIT 1",
                (description,),
                one=True,
            )
            if row:
                return int(row["run_id"])
        return None

    def get_latest_run_id(self, run_type: str | None = None) -> int | None:
        sql = "SELECT run_id FROM runs"
        params: tuple[Any, ...] = ()
        if run_type:
            sql += " WHERE run_type=?"
            params = (run_type,)
        sql += " ORDER BY run_id DESC LIMIT 1"
        row = self._db.query(sql, params, one=True)
        return int(row["run_id"]) if row else None

    def parse_parameters(self, run: dict[str, Any]) -> dict[str, Any]:
        raw = run.get("parameters_json")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
