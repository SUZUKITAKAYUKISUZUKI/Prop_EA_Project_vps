"""Database integrity checks for RC1."""
from __future__ import annotations

from typing import Any

from src.production_hardening.config import REQUIRED_DB_TABLES
from src.repositories.base import create_default_db_manager


class DatabaseIntegrityChecker:
    def evaluate(self, *, profile_id: str) -> dict[str, Any]:
        issues: list[str] = []
        present = 0
        db = create_default_db_manager()
        try:
            for table in REQUIRED_DB_TABLES:
                row = db.query(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                    one=True,
                )
                if row:
                    present += 1
                else:
                    issues.append(f"Missing table: {table}")

            integrity = db.query("PRAGMA integrity_check", (), one=True)
            pragma_ok = str(dict(integrity or {}).get("integrity_check") or "").lower() == "ok"
            if not pragma_ok:
                issues.append(f"SQLite integrity_check failed: {integrity}")

            score = round((present / len(REQUIRED_DB_TABLES)) * 100, 2)
            if not pragma_ok:
                score = min(score, 60.0)
            if issues:
                score = min(score, max(0.0, 100.0 - len(issues) * 8))

            return {
                "data_integrity": score,
                "database_health": score,
                "tables_present": present,
                "tables_checked": len(REQUIRED_DB_TABLES),
                "sqlite_integrity": pragma_ok,
                "issues": issues,
                "healthy": score >= 85 and pragma_ok,
            }
        finally:
            db.close()
