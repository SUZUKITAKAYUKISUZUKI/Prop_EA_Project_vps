"""Portfolio Adaptive Allocation Engine — SQLite schema."""
from __future__ import annotations

import sqlite3

PAAE_SCHEMA_VERSION = "1.0"

PAAE_DDL = """
CREATE TABLE IF NOT EXISTS adaptive_allocation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    strategy TEXT NOT NULL,
    old_weight REAL,
    new_weight REAL,
    reason TEXT,
    risk_score REAL,
    profit_score REAL,
    health_score REAL
);

CREATE INDEX IF NOT EXISTS idx_adaptive_alloc_history_ts
    ON adaptive_allocation_history(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_adaptive_alloc_history_profile
    ON adaptive_allocation_history(profile_id);
CREATE INDEX IF NOT EXISTS idx_adaptive_alloc_history_strategy
    ON adaptive_allocation_history(strategy);
"""


def apply_paae_migrations(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='adaptive_allocation_history'"
    ).fetchone()
    if row is not None:
        return
    conn.executescript(PAAE_DDL)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO schema_meta (key, value, updated_at)
        VALUES (?, ?, datetime('now'))
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """,
        ("paae_schema_version", PAAE_SCHEMA_VERSION),
    )
