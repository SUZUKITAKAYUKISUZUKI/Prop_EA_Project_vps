"""Strategic scenario persistence for AGE v4."""
from __future__ import annotations

import json
from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class StrategicRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns = owns_connection or db is None

    def close(self) -> None:
        if self._owns:
            self._db.close()

    def save_scenario(
        self,
        *,
        profile_id: str,
        horizon_label: str,
        scenario_json: dict[str, Any],
        confidence: float,
    ) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO governor_future_scenarios (
                timestamp, profile_id, horizon_label, scenario_json, confidence
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (utc_now_iso(), profile_id, horizon_label, _json(scenario_json), confidence),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def save_branches(self, scenario_id: int, branches: list[dict[str, Any]]) -> list[int]:
        ids: list[int] = []
        for branch in branches:
            cur = self._db.portfolio.execute(
                """
                INSERT INTO governor_future_branches (
                    scenario_id, timestamp, branch_id, action_type, action_label,
                    metrics_json, strategic_score, rank_category
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scenario_id,
                    utc_now_iso(),
                    str(branch.get("branch_id") or ""),
                    str(branch.get("action_type") or ""),
                    str(branch.get("action_label") or ""),
                    _json(branch.get("metrics_by_horizon") or {}),
                    float(branch.get("strategic_score") or 0),
                    str(branch.get("rank_category") or "REJECT"),
                ),
            )
            ids.append(int(cur.lastrowid))
        self._db.portfolio.commit()
        return ids

    def save_rankings(
        self,
        scenario_id: int,
        *,
        rankings_json: dict[str, Any],
        best_branch_id: str,
        best_action: str,
        confidence: float,
    ) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO governor_future_rankings (
                scenario_id, timestamp, rankings_json, best_branch_id, best_action, confidence
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                scenario_id,
                utc_now_iso(),
                _json(rankings_json),
                best_branch_id,
                best_action,
                confidence,
            ),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def list_scenarios(self, *, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._db.query(
            "SELECT * FROM governor_future_scenarios ORDER BY timestamp DESC, id DESC LIMIT ?",
            (limit,),
        )
        return [_parse_scenario(dict(r)) for r in rows]

    def list_rankings(self, *, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._db.query(
            "SELECT * FROM governor_future_rankings ORDER BY timestamp DESC, id DESC LIMIT ?",
            (limit,),
        )
        return [_parse_ranking(dict(r)) for r in rows]

    def list_branches(self, scenario_id: int) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT * FROM governor_future_branches
            WHERE scenario_id = ?
            ORDER BY strategic_score DESC
            """,
            (scenario_id,),
        )
        return [_parse_branch(dict(r)) for r in rows]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _parse_scenario(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("scenario_json"):
        try:
            row["scenario_json"] = json.loads(row["scenario_json"])
        except (TypeError, json.JSONDecodeError):
            pass
    return row


def _parse_branch(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("metrics_json"):
        try:
            row["metrics_json"] = json.loads(row["metrics_json"])
        except (TypeError, json.JSONDecodeError):
            pass
    return row


def _parse_ranking(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("rankings_json"):
        try:
            row["rankings_json"] = json.loads(row["rankings_json"])
        except (TypeError, json.JSONDecodeError):
            pass
    return row
