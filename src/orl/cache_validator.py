"""Cache health validation for ORL v1."""
from __future__ import annotations

from typing import Any

from src.cace.confidence_cache import ConfidenceCache
from src.repositories.base import create_default_db_manager


class CacheValidator:
    def evaluate(self, *, profile_id: str) -> dict[str, Any]:
        issues: list[str] = []
        db = create_default_db_manager()
        try:
            row = db.query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='analytics_cache'",
                (),
                one=True,
            )
            if not row:
                issues.append("analytics_cache table missing")
                return {"cache_health": 0.0, "issues": issues, "healthy": False}

            cache = ConfidenceCache()
            test_key = f"orl:healthcheck:{profile_id}"
            cache.set(test_key, {"ok": True})
            cached = cache.get_if_fresh(test_key)
            if not cached:
                issues.append("Cache write/read cycle failed")

            stale_rows = db.query(
                "SELECT COUNT(*) AS cnt FROM analytics_cache",
                (),
                one=True,
            )
            entry_count = int(dict(stale_rows or {}).get("cnt") or 0)
            score = 100.0 if not issues else 50.0
            if entry_count == 0 and not issues:
                score = 90.0

            return {
                "cache_health": score,
                "cache_entries": entry_count,
                "issues": issues,
                "healthy": score >= 85,
            }
        finally:
            db.close()
