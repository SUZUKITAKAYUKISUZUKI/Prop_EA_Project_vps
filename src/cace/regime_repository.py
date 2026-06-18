"""Regime persistence and cache for CACE v1.5."""
from __future__ import annotations

import json
from typing import Any

from src.cace.confidence_cache import ConfidenceCache
from src.cace.confidence_v15_config import CACHE_V15_REGIME
from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class RegimeRepository:
    def __init__(
        self,
        db: DatabaseManager | None = None,
        *,
        cache: ConfidenceCache | None = None,
        owns_connection: bool = False,
    ) -> None:
        self._db = db or create_default_db_manager()
        self._owns = owns_connection or db is None
        self._cache = cache or ConfidenceCache()

    def close(self) -> None:
        if self._owns:
            self._db.close()

    def save_regime_snapshot(self, *, profile_id: str, regime_payload: dict[str, Any]) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO confidence_regime_snapshots (
                timestamp, profile_id, regime, confidence_modifier, metrics_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                profile_id,
                str(regime_payload.get("regime") or "TRANSITIONAL"),
                float(regime_payload.get("confidence_modifier") or 0),
                json.dumps(regime_payload.get("metrics") or {}, ensure_ascii=False),
            ),
        )
        self._db.portfolio.commit()
        self._cache.set(CACHE_V15_REGIME.format(profile_id=profile_id), regime_payload)
        return int(cur.lastrowid)

    def get_cached_regime(self, profile_id: str) -> dict[str, Any] | None:
        return self._cache.get_if_fresh(CACHE_V15_REGIME.format(profile_id=profile_id))

    def list_recent(self, *, profile_id: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT * FROM confidence_regime_snapshots
            WHERE profile_id=?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (profile_id, limit),
        )
        result = []
        for row in rows:
            item = dict(row)
            if item.get("metrics_json"):
                try:
                    item["metrics_json"] = json.loads(item["metrics_json"])
                except (TypeError, json.JSONDecodeError):
                    pass
            result.append(item)
        return result
