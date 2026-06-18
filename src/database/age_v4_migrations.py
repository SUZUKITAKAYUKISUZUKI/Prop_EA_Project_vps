"""AGE v4 — Strategic Decision Simulator schema."""
from __future__ import annotations

import sqlite3

from src.database.schema_migrations import _set_schema_meta, _table_exists

AGE_V4_SCHEMA_VERSION = "4.0"

AGE_V4_DDL = """
CREATE TABLE IF NOT EXISTS governor_future_scenarios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT,
    horizon_label TEXT,
    scenario_json TEXT,
    confidence REAL
);

CREATE INDEX IF NOT EXISTS idx_governor_future_scenarios_ts
    ON governor_future_scenarios(timestamp DESC);

CREATE TABLE IF NOT EXISTS governor_future_branches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario_id INTEGER,
    timestamp TEXT NOT NULL,
    branch_id TEXT,
    action_type TEXT,
    action_label TEXT,
    metrics_json TEXT,
    strategic_score REAL,
    rank_category TEXT,
    FOREIGN KEY (scenario_id) REFERENCES governor_future_scenarios(id)
);

CREATE INDEX IF NOT EXISTS idx_governor_future_branches_scenario
    ON governor_future_branches(scenario_id);

CREATE TABLE IF NOT EXISTS governor_future_rankings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario_id INTEGER,
    timestamp TEXT NOT NULL,
    rankings_json TEXT,
    best_branch_id TEXT,
    best_action TEXT,
    confidence REAL,
    FOREIGN KEY (scenario_id) REFERENCES governor_future_scenarios(id)
);

CREATE INDEX IF NOT EXISTS idx_governor_future_rankings_ts
    ON governor_future_rankings(timestamp DESC);
"""


def apply_age_v4_migrations(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "governor_future_scenarios"):
        conn.executescript(AGE_V4_DDL)
    _set_schema_meta(conn, "age_v4_schema_version", AGE_V4_SCHEMA_VERSION)
