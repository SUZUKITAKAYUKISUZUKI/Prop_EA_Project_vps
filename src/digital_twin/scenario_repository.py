"""Persist PDTS scenario run results."""
from __future__ import annotations

import json
from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class ScenarioRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns = owns_connection or db is None

    def close(self) -> None:
        if self._owns:
            self._db.close()

    def save_run(
        self,
        *,
        profile_id: str,
        scenario_name: str,
        metrics: dict[str, Any],
        allocation: dict[str, float],
        created_by: str = "pdts",
        monte_carlo: dict[str, Any] | None = None,
    ) -> int:
        rec_score = metrics.get("score")
        rec_label = metrics.get("recommendation")
        cur = self._db.portfolio.execute(
            """
            INSERT INTO scenario_runs (
                timestamp, profile_id, scenario_name,
                pass_rate, avg_pass_days, pf, total_r, max_dd, sharpe, health_score,
                allocation_json, created_by,
                recommendation_score, recommendation,
                win_rate, recovery_factor, ulcer_index, risk_score,
                prob_recovery, prob_ruin, monte_carlo_json, metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                profile_id,
                scenario_name,
                metrics.get("pass_rate"),
                metrics.get("avg_pass_days"),
                metrics.get("pf"),
                metrics.get("total_r"),
                metrics.get("max_dd"),
                metrics.get("sharpe"),
                metrics.get("health_score"),
                json.dumps(allocation),
                created_by,
                rec_score,
                rec_label,
                metrics.get("win_rate"),
                metrics.get("recovery_factor"),
                metrics.get("ulcer_index"),
                metrics.get("risk_score"),
                metrics.get("prob_recovery"),
                metrics.get("prob_ruin"),
                json.dumps(monte_carlo or {}),
                json.dumps(metrics.get("metrics") or metrics),
            ),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def list_runs(
        self,
        *,
        profile_id: str | None = None,
        scenario_name: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if profile_id:
            clauses.append("profile_id = ?")
            params.append(profile_id)
        if scenario_name:
            clauses.append("scenario_name = ?")
            params.append(scenario_name)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._db.query(
            f"""
            SELECT * FROM scenario_runs
            {where}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        return [self._row_to_dict(row) for row in rows]

    def get_latest(
        self,
        profile_id: str,
        scenario_name: str,
    ) -> dict[str, Any] | None:
        row = self._db.query(
            """
            SELECT * FROM scenario_runs
            WHERE profile_id = ? AND scenario_name = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (profile_id, scenario_name),
            one=True,
        )
        return self._row_to_dict(row) if row else None

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        data = dict(row)
        for key in ("allocation_json", "monte_carlo_json", "metrics_json"):
            if data.get(key):
                try:
                    data[key] = json.loads(data[key])
                except (TypeError, json.JSONDecodeError):
                    pass
        return data
