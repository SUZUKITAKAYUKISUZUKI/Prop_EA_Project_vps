"""State Analytics Engine v1 — SQLite schema."""
from __future__ import annotations

import sqlite3

STATE_ANALYTICS_SCHEMA_VERSION = "1.0"

STATE_ANALYTICS_DDL = """
CREATE TABLE IF NOT EXISTS account_state_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    state TEXT NOT NULL,
    profile TEXT NOT NULL,
    equity REAL,
    balance REAL,
    drawdown_pct REAL,
    risk_budget_remaining REAL,
    challenge_progress REAL,
    source TEXT DEFAULT 'auto_switch'
);

CREATE INDEX IF NOT EXISTS idx_account_state_history_ts
    ON account_state_history(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_account_state_history_state
    ON account_state_history(state);
CREATE INDEX IF NOT EXISTS idx_account_state_history_profile
    ON account_state_history(profile);
"""


def apply_state_analytics_migrations(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='account_state_history'"
    ).fetchone()
    if row is not None:
        return
    conn.executescript(STATE_ANALYTICS_DDL)
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
        ("state_analytics_schema_version", STATE_ANALYTICS_SCHEMA_VERSION),
    )
