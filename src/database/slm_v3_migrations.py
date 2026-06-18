"""Strategy Lifecycle Manager v3 — governance schema upgrades."""
from __future__ import annotations

import sqlite3

from src.database.schema_migrations import _add_column, _set_schema_meta, _table_exists

SLM_V3_SCHEMA_VERSION = "3.0"

GENEALOGY_DDL = """
CREATE TABLE IF NOT EXISTS strategy_genealogy (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id TEXT NOT NULL,
    parent_strategy_id TEXT,
    strategy_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_strategy_genealogy_strategy
    ON strategy_genealogy(strategy_id);
CREATE INDEX IF NOT EXISTS idx_strategy_genealogy_parent
    ON strategy_genealogy(parent_strategy_id);
"""


def apply_slm_v3_migrations(conn: sqlite3.Connection) -> None:
    registry_cols = (
        "diversification_score REAL",
        "recovery_score REAL",
        "challenge_score REAL",
        "stability_contribution_score REAL",
        "dd_reduction_score REAL",
        "core_strategy INTEGER DEFAULT 0",
    )
    history_cols = (
        ("strategy_version", "strategy_version TEXT"),
        ("diversification_score", "diversification_score REAL"),
        ("recovery_score", "recovery_score REAL"),
        ("challenge_score", "challenge_score REAL"),
        ("stability_contribution_score", "stability_contribution_score REAL"),
        ("dd_reduction_score", "dd_reduction_score REAL"),
    )

    for col in registry_cols:
        _add_column(conn, "strategy_registry", col)

    for _, col in history_cols:
        _add_column(conn, "strategy_lifecycle_history", col)

    if not _table_exists(conn, "strategy_genealogy"):
        conn.executescript(GENEALOGY_DDL)

    conn.execute(
        """
        UPDATE strategy_registry
        SET core_strategy = 1, current_stage = 'CORE'
        WHERE strategy_id = 'LSFC' AND current_stage = 'PRODUCTION'
        """
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO strategy_genealogy (strategy_id, parent_strategy_id, strategy_version, created_at)
        SELECT strategy_id, strategy_id, strategy_version, created_at
        FROM strategy_registry
        """
    )
    _set_schema_meta(conn, "slm_schema_version", SLM_V3_SCHEMA_VERSION)
