"""Database health validation for ORL v1."""
from __future__ import annotations

from typing import Any

from src.orl.config import REQUIRED_DB_TABLES
from src.repositories.base import create_default_db_manager


class DatabaseValidator:
    def __init__(self, db: Any | None = None) -> None:
        self._db = db

    def evaluate(self, *, profile_id: str) -> dict[str, Any]:
        issues: list[str] = []
        tables_ok = 0
        db = self._db or create_default_db_manager()
        owns = self._db is None

        try:
            for table in REQUIRED_DB_TABLES:
                row = db.query(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                    one=True,
                )
                if row:
                    tables_ok += 1
                else:
                    issues.append(f"Missing table: {table}")

            score = round((tables_ok / len(REQUIRED_DB_TABLES)) * 100, 2) if REQUIRED_DB_TABLES else 100.0
            if issues:
                score = min(score, 70.0)

            return {
                "database_health": score,
                "tables_checked": len(REQUIRED_DB_TABLES),
                "tables_present": tables_ok,
                "issues": issues,
                "healthy": score >= 85 and not issues,
            }
        finally:
            if owns:
                db.close()
