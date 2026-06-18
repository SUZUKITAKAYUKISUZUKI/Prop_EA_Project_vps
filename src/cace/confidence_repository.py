"""Confidence persistence for CACE."""
from __future__ import annotations

import json
from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class ConfidenceRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns = owns_connection or db is None

    def close(self) -> None:
        if self._owns:
            self._db.close()

    def save_allocation_confidence(
        self,
        *,
        profile_id: str,
        allocation_json: dict[str, Any],
        confidence: float,
        category: str,
        expected_r: float,
        expected_pf: float,
        expected_dd: float,
        reason_json: list[str],
        breakdown: dict[str, float] | None = None,
    ) -> int:
        bd = breakdown or {}
        cur = self._db.portfolio.execute(
            """
            INSERT INTO allocation_confidence (
                timestamp, profile_id, allocation_json, confidence, category,
                expected_r, expected_pf, expected_dd, reason_json,
                historical_reliability, mc_stability, forecast_stability,
                portfolio_health, lifecycle_quality, breakdown_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                profile_id,
                _json(allocation_json),
                confidence,
                category,
                expected_r,
                expected_pf,
                expected_dd,
                _json(reason_json),
                bd.get("historical_reliability"),
                bd.get("monte_carlo_stability"),
                bd.get("forecast_stability"),
                bd.get("portfolio_health"),
                bd.get("lifecycle_quality"),
                _json(bd) if bd else None,
            ),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def save_strategy_confidence(
        self,
        *,
        strategy: str,
        confidence: float,
        portfolio_fit: float,
        lifecycle_stage: str,
        reason_json: list[str],
        breakdown: dict[str, float] | None = None,
    ) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO strategy_confidence (
                timestamp, strategy, confidence, portfolio_fit, lifecycle_stage,
                reason_json, breakdown_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                strategy,
                confidence,
                portfolio_fit,
                lifecycle_stage,
                _json(reason_json),
                _json(breakdown) if breakdown else None,
            ),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def save_confidence_history(
        self,
        *,
        profile_id: str,
        confidence: float,
        category: str,
        snapshot_json: dict[str, Any],
        trend: str | None = None,
        trend_strength: float | None = None,
    ) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO confidence_history (
                timestamp, profile_id, confidence, category, snapshot_json, trend, trend_strength
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                profile_id,
                confidence,
                category,
                _json(snapshot_json),
                trend,
                trend_strength,
            ),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def list_allocation_confidence(self, *, profile_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        if profile_id:
            rows = self._db.query(
                """
                SELECT * FROM allocation_confidence
                WHERE profile_id=?
                ORDER BY timestamp DESC, id DESC LIMIT ?
                """,
                (profile_id, limit),
            )
        else:
            rows = self._db.query(
                "SELECT * FROM allocation_confidence ORDER BY timestamp DESC, id DESC LIMIT ?",
                (limit,),
            )
        return [_parse_allocation(dict(r)) for r in rows]

    def list_strategy_confidence(self, *, strategy: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if strategy:
            rows = self._db.query(
                """
                SELECT * FROM strategy_confidence
                WHERE strategy=?
                ORDER BY timestamp DESC, id DESC LIMIT ?
                """,
                (strategy, limit),
            )
        else:
            rows = self._db.query(
                "SELECT * FROM strategy_confidence ORDER BY timestamp DESC, id DESC LIMIT ?",
                (limit,),
            )
        return [_parse_strategy(dict(r)) for r in rows]

    def list_confidence_history(self, *, profile_id: str | None = None, limit: int = 30) -> list[dict[str, Any]]:
        if profile_id:
            rows = self._db.query(
                """
                SELECT * FROM confidence_history
                WHERE profile_id=?
                ORDER BY timestamp DESC, id DESC LIMIT ?
                """,
                (profile_id, limit),
            )
        else:
            rows = self._db.query(
                "SELECT * FROM confidence_history ORDER BY timestamp DESC, id DESC LIMIT ?",
                (limit,),
            )
        return [_parse_history(dict(r)) for r in rows]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _parse_allocation(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("allocation_json", "reason_json", "breakdown_json"):
        if row.get(key):
            try:
                row[key] = json.loads(row[key])
            except (TypeError, json.JSONDecodeError):
                pass
    return row


def _parse_strategy(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("reason_json", "breakdown_json"):
        if row.get(key):
            try:
                row[key] = json.loads(row[key])
            except (TypeError, json.JSONDecodeError):
                pass
    return row


def _parse_history(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("snapshot_json"):
        try:
            row["snapshot_json"] = json.loads(row["snapshot_json"])
        except (TypeError, json.JSONDecodeError):
            pass
    return row
