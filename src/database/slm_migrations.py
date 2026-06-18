"""Strategy Lifecycle Manager — SQLite schema."""
from __future__ import annotations

import sqlite3

from src.database.profile_migrations import DASHBOARD_STRATEGY_CODES

SLM_SCHEMA_VERSION = "1.0"

SLM_DDL = """
CREATE TABLE IF NOT EXISTS strategy_registry (
    strategy_id TEXT PRIMARY KEY,
    strategy_name TEXT NOT NULL,
    current_stage TEXT NOT NULL DEFAULT 'INCUBATION',
    strategy_version TEXT DEFAULT '1.0',
    created_at TEXT NOT NULL,
    promoted_at TEXT,
    demoted_at TEXT,
    retired_at TEXT,
    score REAL DEFAULT 0.0,
    active INTEGER NOT NULL DEFAULT 1,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS strategy_lifecycle_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    old_stage TEXT,
    new_stage TEXT NOT NULL,
    reason TEXT,
    score REAL,
    pf REAL,
    pass_rate REAL,
    max_dd REAL,
    oos_pf REAL
);

CREATE INDEX IF NOT EXISTS idx_strategy_registry_stage
    ON strategy_registry(current_stage);
CREATE INDEX IF NOT EXISTS idx_strategy_lifecycle_history_ts
    ON strategy_lifecycle_history(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_strategy_lifecycle_history_strategy
    ON strategy_lifecycle_history(strategy_id);
"""

DEFAULT_STAGES = {
    "LSFC": "PRODUCTION",
    "DBBS": "PRODUCTION",
    "DiNapoli": "PRODUCTION",
    "VAMR": "CANDIDATE",
    "SMRS": "RECOVERY",
}


def apply_slm_migrations(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='strategy_registry'"
    ).fetchone()
    if row is not None:
        return
    conn.executescript(SLM_DDL)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    for code in DASHBOARD_STRATEGY_CODES:
        stage = DEFAULT_STAGES.get(code, "INCUBATION")
        conn.execute(
            """
            INSERT INTO strategy_registry (
                strategy_id, strategy_name, current_stage, strategy_version,
                created_at, score, active, notes
            ) VALUES (?, ?, ?, '1.0', datetime('now'), 70.0, 1, ?)
            """,
            (code, code, stage, "seeded_by_slm_migration"),
        )
    conn.execute(
        """
        INSERT INTO schema_meta (key, value, updated_at)
        VALUES (?, ?, datetime('now'))
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """,
        ("slm_schema_version", SLM_SCHEMA_VERSION),
    )
