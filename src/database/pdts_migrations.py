"""Portfolio Digital Twin & Scenario Simulator — SQLite schema."""
from __future__ import annotations

import sqlite3

PDTS_SCHEMA_VERSION = "1.0"

PDTS_DDL = """
CREATE TABLE IF NOT EXISTS scenario_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    scenario_name TEXT NOT NULL,
    pass_rate REAL,
    avg_pass_days REAL,
    pf REAL,
    total_r REAL,
    max_dd REAL,
    sharpe REAL,
    health_score REAL,
    allocation_json TEXT,
    created_by TEXT DEFAULT 'pdts',
    recommendation_score REAL,
    recommendation TEXT,
    win_rate REAL,
    recovery_factor REAL,
    ulcer_index REAL,
    risk_score REAL,
    prob_recovery REAL,
    prob_ruin REAL,
    monte_carlo_json TEXT,
    metrics_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_scenario_runs_ts
    ON scenario_runs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_scenario_runs_profile
    ON scenario_runs(profile_id);
CREATE INDEX IF NOT EXISTS idx_scenario_runs_name
    ON scenario_runs(scenario_name);
"""


def apply_pdts_migrations(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='scenario_runs'"
    ).fetchone()
    if row is not None:
        return
    conn.executescript(PDTS_DDL)
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
        ("pdts_schema_version", PDTS_SCHEMA_VERSION),
    )
