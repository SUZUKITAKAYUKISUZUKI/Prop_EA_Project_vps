"""Analytics result cache repository."""
from __future__ import annotations

import json
from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class CacheRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns_connection = owns_connection or db is None

    def close(self) -> None:
        if self._owns_connection:
            self._db.close()

    def get(self, cache_key: str) -> Any | None:
        row = self._db.query(
            "SELECT cache_value FROM analytics_cache WHERE cache_key=?",
            (cache_key,),
            one=True,
        )
        if not row:
            return None
        try:
            return json.loads(row["cache_value"])
        except json.JSONDecodeError:
            return row["cache_value"]

    def set(self, cache_key: str, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
        self._db.portfolio.execute(
            """
            INSERT INTO analytics_cache (cache_key, cache_value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                cache_value=excluded.cache_value,
                updated_at=excluded.updated_at
            """,
            (cache_key, payload, utc_now_iso()),
        )
        self._db.portfolio.commit()

    def delete(self, cache_key: str) -> None:
        self._db.portfolio.execute("DELETE FROM analytics_cache WHERE cache_key=?", (cache_key,))
        self._db.portfolio.commit()

    def get_or_compute(self, cache_key: str, compute_fn) -> Any:
        cached = self.get(cache_key)
        if cached is not None:
            return cached
        value = compute_fn()
        self.set(cache_key, value)
        return value
