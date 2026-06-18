"""Persistence for CACE v1.6 decay and consensus history."""
from __future__ import annotations

import json
from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class ConfidenceV16Repository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns = owns_connection or db is None

    def close(self) -> None:
        if self._owns:
            self._db.close()

    def save_decay(
        self,
        *,
        profile_id: str,
        durability_score: float,
        durability_category: str,
        half_life: int,
        forecast_json: dict[str, Any],
    ) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO confidence_decay_history (
                timestamp, profile_id, durability_score, durability_category,
                half_life, forecast_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                profile_id,
                durability_score,
                durability_category,
                half_life,
                json.dumps(forecast_json, ensure_ascii=False),
            ),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def save_consensus(
        self,
        *,
        profile_id: str,
        recommended_action: str,
        consensus_score: float,
        consensus_category: str,
        agreement_ratio: float,
        agree_count: int,
        total_modules: int,
    ) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO confidence_consensus_history (
                timestamp, profile_id, recommended_action, consensus_score,
                consensus_category, agreement_ratio, agree_count, total_modules
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                profile_id,
                recommended_action,
                consensus_score,
                consensus_category,
                agreement_ratio,
                agree_count,
                total_modules,
            ),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def load_latest_decay(self, *, profile_id: str) -> dict[str, Any] | None:
        row = self._db.query(
            """
            SELECT * FROM confidence_decay_history
            WHERE profile_id=?
            ORDER BY timestamp DESC, id DESC
            LIMIT 1
            """,
            (profile_id,),
            one=True,
        )
        if not row:
            return None
        item = dict(row)
        if item.get("forecast_json"):
            try:
                item["forecast_json"] = json.loads(item["forecast_json"])
            except (TypeError, json.JSONDecodeError):
                pass
        return item

    def load_latest_consensus(self, *, profile_id: str) -> dict[str, Any] | None:
        row = self._db.query(
            """
            SELECT * FROM confidence_consensus_history
            WHERE profile_id=?
            ORDER BY timestamp DESC, id DESC
            LIMIT 1
            """,
            (profile_id,),
            one=True,
        )
        return dict(row) if row else None
