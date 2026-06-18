"""TTL-aware cache helpers for CACE."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from src.cace.confidence_factors import CACHE_TTL_SECONDS
from src.repositories.cache_repository import CacheRepository


class ConfidenceCache:
    def __init__(
        self,
        cache: CacheRepository | None = None,
        *,
        ttl_seconds: int = CACHE_TTL_SECONDS,
    ) -> None:
        self._cache = cache or CacheRepository(owns_connection=False)
        self._ttl = ttl_seconds

    def get_if_fresh(self, cache_key: str) -> Any | None:
        row = self._cache._db.query(
            "SELECT cache_value, updated_at FROM analytics_cache WHERE cache_key=?",
            (cache_key,),
            one=True,
        )
        if not row:
            return None
        row_dict = dict(row)
        updated_at = str(row_dict.get("updated_at") or "")
        if not self._is_fresh(updated_at):
            return None
        try:
            return json.loads(row_dict["cache_value"])
        except (TypeError, json.JSONDecodeError, KeyError):
            return row_dict.get("cache_value")

    def set(self, cache_key: str, value: Any) -> None:
        self._cache.set(cache_key, value)

    def delete(self, cache_key: str) -> None:
        self._cache.delete(cache_key)

    def _is_fresh(self, updated_at: str) -> bool:
        if not updated_at:
            return False
        try:
            normalized = updated_at.replace("Z", "+00:00")
            ts = datetime.fromisoformat(normalized)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            return age <= self._ttl
        except ValueError:
            return False
