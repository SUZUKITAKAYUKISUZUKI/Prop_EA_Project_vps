"""CACE v1.7 schema — decision accuracy and calibration history."""
from __future__ import annotations

import sqlite3

from src.database.schema_migrations import _add_column, _set_schema_meta, _table_exists

CACE_V17_SCHEMA_VERSION = "1.7.0"

CACE_V17_DDL = """
CREATE TABLE IF NOT EXISTS decision_accuracy_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    module TEXT,
    decision_type TEXT,
    confidence REAL,
    predicted_benefit REAL,
    actual_benefit REAL,
    predicted_dd REAL,
    actual_dd REAL,
    prediction_error REAL,
    accuracy_score REAL,
    evaluation_date TEXT,
    evaluated INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_decision_accuracy_profile
    ON decision_accuracy_history(profile_id);
CREATE INDEX IF NOT EXISTS idx_decision_accuracy_eval_date
    ON decision_accuracy_history(evaluation_date);

CREATE TABLE IF NOT EXISTS confidence_calibration_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    calibration_score REAL,
    calibration_category TEXT,
    decision_accuracy_score REAL,
    reliability_trend TEXT,
    payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_confidence_calibration_profile
    ON confidence_calibration_history(profile_id);

CREATE TABLE IF NOT EXISTS confidence_learning_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    note TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_confidence_learning_profile
    ON confidence_learning_notes(profile_id);
"""


def apply_cace_v17_migrations(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "decision_accuracy_history"):
        conn.executescript(CACE_V17_DDL)
    else:
        for ddl in (
            "profile_id TEXT",
            "module TEXT",
            "evaluated INTEGER DEFAULT 0",
        ):
            _add_column(conn, "decision_accuracy_history", ddl)
    _set_schema_meta(conn, "cace_v17_schema_version", CACE_V17_SCHEMA_VERSION)
