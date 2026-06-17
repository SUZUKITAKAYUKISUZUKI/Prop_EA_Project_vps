"""SQLite schema definitions for portfolio_os.db and market_data.db."""
from __future__ import annotations

import sqlite3

PORTFOLIO_TABLES = (
    "runs",
    "trades",
    "features",
    "bt_summary",
    "wft_results",
    "mc_results",
    "portfolio_results",
    "risk_attribution",
    "import_registry",
    "analytics_cache",
    "trade_events",
    "import_state",
)

MARKET_TABLES = ("candles",)

PORTFOLIO_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type TEXT NOT NULL,
    strategy TEXT,
    created_at TEXT NOT NULL,
    description TEXT,
    parameters_json TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    strategy TEXT,
    symbol TEXT,
    direction TEXT,
    entry_time TEXT,
    exit_time TEXT,
    entry_price REAL,
    exit_price REAL,
    r_multiple REAL,
    profit REAL,
    result TEXT,
    source_trade_id TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS features (
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER,
    run_id INTEGER NOT NULL,
    strategy TEXT,
    feature_json TEXT NOT NULL,
    source_key TEXT,
    FOREIGN KEY (trade_id) REFERENCES trades(trade_id) ON DELETE SET NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS bt_summary (
    summary_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    pf REAL,
    wr REAL,
    total_r REAL,
    max_dd REAL,
    sharpe REAL,
    recovery REAL,
    label TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS wft_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    window_id INTEGER NOT NULL,
    oos_pf REAL,
    oos_r REAL,
    oos_dd REAL,
    pass_flag INTEGER,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS mc_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    pass_rate REAL,
    ror REAL,
    avg_pass_days REAL,
    max_dd REAL,
    label TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS portfolio_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    allocation_json TEXT,
    pf REAL,
    total_r REAL,
    max_dd REAL,
    pass_rate REAL,
    rank INTEGER,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS risk_attribution (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    strategy TEXT NOT NULL,
    contribution_r REAL,
    contribution_dd REAL,
    contribution_pf REAL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS import_registry (
    source_path TEXT PRIMARY KEY,
    run_id INTEGER,
    csv_kind TEXT,
    row_count INTEGER DEFAULT 0,
    imported_at TEXT NOT NULL,
    checksum TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS analytics_cache (
    cache_key TEXT PRIMARY KEY,
    cache_value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_dedup
    ON trades(run_id, source_trade_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_features_dedup
    ON features(run_id, source_key);

CREATE UNIQUE INDEX IF NOT EXISTS idx_wft_dedup
    ON wft_results(run_id, window_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_bt_summary_dedup
    ON bt_summary(run_id, label);

CREATE UNIQUE INDEX IF NOT EXISTS idx_mc_dedup
    ON mc_results(run_id, label);

CREATE UNIQUE INDEX IF NOT EXISTS idx_portfolio_dedup
    ON portfolio_results(run_id, rank);

CREATE UNIQUE INDEX IF NOT EXISTS idx_risk_attr_dedup
    ON risk_attribution(run_id, strategy);

CREATE INDEX IF NOT EXISTS idx_trades_run ON trades(run_id);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_features_run ON features(run_id);
CREATE INDEX IF NOT EXISTS idx_features_trade ON features(trade_id);
CREATE INDEX IF NOT EXISTS idx_features_strategy ON features(strategy);
CREATE INDEX IF NOT EXISTS idx_wft_run_window ON wft_results(run_id, window_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_run ON portfolio_results(run_id);
CREATE INDEX IF NOT EXISTS idx_analytics_cache_updated ON analytics_cache(updated_at);

CREATE TABLE IF NOT EXISTS trade_events (
    event_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    trade_id TEXT,
    strategy TEXT,
    symbol TEXT,
    payload_json TEXT NOT NULL,
    imported_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS import_state (
    filename TEXT PRIMARY KEY,
    last_offset INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trade_events_timestamp ON trade_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_trade_events_trade_id ON trade_events(trade_id);
CREATE INDEX IF NOT EXISTS idx_trade_events_strategy ON trade_events(strategy);
CREATE INDEX IF NOT EXISTS idx_trade_events_event_type ON trade_events(event_type);
"""

MARKET_DDL = """
CREATE TABLE IF NOT EXISTS candles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    dt TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_candles_sym_tf_dt
    ON candles(symbol, timeframe, dt);

CREATE INDEX IF NOT EXISTS idx_candles_lookup
    ON candles(symbol, timeframe, dt);
"""


def _apply_pragmas(conn: sqlite3.Connection, journal_mode: str, synchronous: str) -> None:
    conn.execute(f"PRAGMA journal_mode={journal_mode}")
    conn.execute(f"PRAGMA synchronous={synchronous}")
    conn.execute("PRAGMA foreign_keys=ON")


def create_portfolio_schema(
    conn: sqlite3.Connection,
    *,
    journal_mode: str = "WAL",
    synchronous: str = "NORMAL",
) -> None:
    _apply_pragmas(conn, journal_mode, synchronous)
    conn.executescript(PORTFOLIO_DDL)


def create_market_schema(
    conn: sqlite3.Connection,
    *,
    journal_mode: str = "WAL",
    synchronous: str = "NORMAL",
) -> None:
    _apply_pragmas(conn, journal_mode, synchronous)
    conn.executescript(MARKET_DDL)


def list_portfolio_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def list_market_tables(conn: sqlite3.Connection) -> list[str]:
    return list_portfolio_tables(conn)
