"""PRAE v2 — SQLite views for strategy/symbol/recovery/transition risk."""
from __future__ import annotations

import sqlite3

RISK_ATTRIBUTION_V2_SCHEMA_VERSION = "2.0"

RISK_ATTRIBUTION_V2_VIEWS = """
DROP VIEW IF EXISTS v_state_transition_risk;
DROP VIEW IF EXISTS v_recovery_events;
DROP VIEW IF EXISTS v_symbol_risk;
DROP VIEW IF EXISTS v_strategy_risk;

CREATE VIEW v_strategy_risk AS
SELECT
    COALESCE(strategy, 'UNKNOWN') AS strategy,
    COUNT(*) AS trade_count,
    SUM(COALESCE(r_multiple, 0.0)) AS total_r,
    AVG(CASE WHEN COALESCE(r_multiple, 0.0) > 0 THEN 1.0 ELSE 0.0 END) * 100.0 AS win_rate,
    SUM(CASE WHEN COALESCE(r_multiple, 0.0) > 0 THEN COALESCE(r_multiple, 0.0) ELSE 0 END) AS gross_profit,
    SUM(CASE WHEN COALESCE(r_multiple, 0.0) < 0 THEN ABS(COALESCE(r_multiple, 0.0)) ELSE 0 END) AS gross_loss
FROM trades
GROUP BY COALESCE(strategy, 'UNKNOWN');

CREATE VIEW v_symbol_risk AS
SELECT
    COALESCE(symbol, 'UNKNOWN') AS symbol,
    COUNT(*) AS trade_count,
    SUM(COALESCE(r_multiple, 0.0)) AS total_r,
    AVG(CASE WHEN COALESCE(r_multiple, 0.0) > 0 THEN 1.0 ELSE 0.0 END) * 100.0 AS win_rate,
    SUM(CASE WHEN COALESCE(r_multiple, 0.0) > 0 THEN COALESCE(r_multiple, 0.0) ELSE 0 END) AS gross_profit,
    SUM(CASE WHEN COALESCE(r_multiple, 0.0) < 0 THEN ABS(COALESCE(r_multiple, 0.0)) ELSE 0 END) AS gross_loss
FROM trades
GROUP BY COALESCE(symbol, 'UNKNOWN');

CREATE VIEW v_recovery_events AS
SELECT
    id,
    timestamp,
    state,
    profile,
    equity,
    balance,
    drawdown_pct,
    risk_budget_remaining,
    challenge_progress,
    source
FROM account_state_history
WHERE LOWER(state) = 'recovery';

CREATE VIEW v_state_transition_risk AS
SELECT
    h.id,
    h.timestamp,
    LAG(h.state) OVER (ORDER BY h.timestamp, h.id) AS from_state,
    h.state AS to_state,
    h.profile,
    h.equity,
    h.drawdown_pct,
    h.risk_budget_remaining
FROM account_state_history h;
"""


def apply_risk_attribution_v2_migrations(conn: sqlite3.Connection) -> None:
    if _view_exists(conn, "v_strategy_risk"):
        return
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    conn.executescript(RISK_ATTRIBUTION_V2_VIEWS)
    conn.execute(
        """
        INSERT INTO schema_meta (key, value, updated_at)
        VALUES (?, ?, datetime('now'))
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """,
        ("risk_attribution_v2_schema_version", RISK_ATTRIBUTION_V2_SCHEMA_VERSION),
    )


def _view_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='view' AND name=? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None
