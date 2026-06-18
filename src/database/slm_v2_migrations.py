"""Strategy Lifecycle Manager v2 — portfolio fit schema upgrades."""
from __future__ import annotations

import sqlite3

from src.database.schema_migrations import _add_column, _set_schema_meta

SLM_V2_SCHEMA_VERSION = "2.0"


def apply_slm_v2_migrations(conn: sqlite3.Connection) -> None:
    _add_column(conn, "strategy_registry", "portfolio_fit_score REAL")
    _add_column(conn, "strategy_lifecycle_history", "portfolio_fit_score REAL")
    _set_schema_meta(conn, "slm_schema_version", SLM_V2_SCHEMA_VERSION)
