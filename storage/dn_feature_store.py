"""DiNapoli feature log — SQLite persistence (entry INSERT / exit UPDATE)."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DN_FEATURE_DB = PROJECT_ROOT / "storage" / "dn_feature_log.db"

DN_FEATURE_TABLE = "dn_feature_log"

# Entry-time columns (INSERT) + exit columns (UPDATE) + metadata
DN_FEATURE_COLUMNS: tuple[str, ...] = (
    "trade_id",
    "run_id",
    "wft_window",
    "is_oos",
    "symbol",
    "direction",
    "entry_time",
    "exit_time",
    "session",
    "weekday",
    "month",
    "is_gotobi",
    "hour",
    "minute",
    "setup_type",
    "fib_level",
    "fib_distance",
    "pullback_depth",
    "swing_size",
    "swing_duration",
    "trend_direction",
    "trend_strength",
    "ema20",
    "ema50",
    "ema200",
    "ema20_slope",
    "ema50_slope",
    "ema200_slope",
    "ema_alignment_score",
    "atr_m15",
    "atr_h1",
    "atr_h4",
    "volatility_regime",
    "rsi_m15",
    "rsi_h1",
    "momentum_score",
    "velocity",
    "acceleration",
    "distance_to_asia_high",
    "distance_to_asia_low",
    "distance_to_london_high",
    "distance_to_london_low",
    "distance_to_daily_high",
    "distance_to_daily_low",
    "spread",
    "tick_volume",
    "volume_zscore",
    "liquidity_score",
    "minutes_from_london_open",
    "minutes_from_ny_open",
    "minutes_to_major_news",
    "minutes_after_major_news",
    "llm_decision",
    "llm_confidence",
    "llm_reason",
    "decision_source",
    "ev_rank",
    "ev_bucket",
    "executed",
    "result_r",
    "win_loss",
    "holding_minutes",
    "max_favorable_excursion_r",
    "max_adverse_excursion_r",
)

ENTRY_COLUMNS: tuple[str, ...] = tuple(
    c
    for c in DN_FEATURE_COLUMNS
    if c
    not in (
        "exit_time",
        "result_r",
        "win_loss",
        "holding_minutes",
        "max_favorable_excursion_r",
        "max_adverse_excursion_r",
    )
)

EXIT_COLUMNS: tuple[str, ...] = (
    "exit_time",
    "result_r",
    "win_loss",
    "holding_minutes",
    "max_favorable_excursion_r",
    "max_adverse_excursion_r",
    "executed",
)


def _sql_type(col: str) -> str:
    if col in ("is_gotobi", "executed", "wft_window", "is_oos", "weekday", "month", "hour", "minute"):
        return "INTEGER"
    if col in (
        "fib_level",
        "fib_distance",
        "pullback_depth",
        "swing_size",
        "swing_duration",
        "trend_strength",
        "ema20",
        "ema50",
        "ema200",
        "ema20_slope",
        "ema50_slope",
        "ema200_slope",
        "ema_alignment_score",
        "atr_m15",
        "atr_h1",
        "atr_h4",
        "volatility_regime",
        "rsi_m15",
        "rsi_h1",
        "momentum_score",
        "velocity",
        "acceleration",
        "distance_to_asia_high",
        "distance_to_asia_low",
        "distance_to_london_high",
        "distance_to_london_low",
        "distance_to_daily_high",
        "distance_to_daily_low",
        "spread",
        "tick_volume",
        "volume_zscore",
        "liquidity_score",
        "minutes_from_london_open",
        "minutes_from_ny_open",
        "minutes_to_major_news",
        "minutes_after_major_news",
        "llm_confidence",
        "ev_rank",
        "result_r",
        "holding_minutes",
        "max_favorable_excursion_r",
        "max_adverse_excursion_r",
    ):
        return "REAL"
    return "TEXT"


CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {DN_FEATURE_TABLE} (
    {", ".join(f"{col} {_sql_type(col)}" for col in DN_FEATURE_COLUMNS)},
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (run_id, trade_id)
);
"""


class DnFeatureStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_DN_FEATURE_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def ensure_schema(self) -> None:
        conn = self.connect()
        conn.execute(CREATE_TABLE_SQL)
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({DN_FEATURE_TABLE})")}
        for col in DN_FEATURE_COLUMNS:
            if col not in existing:
                conn.execute(
                    f"ALTER TABLE {DN_FEATURE_TABLE} ADD COLUMN {col} {_sql_type(col)}"
                )
        conn.commit()

    def clear_run(self, run_id: str) -> None:
        conn = self.connect()
        conn.execute(f"DELETE FROM {DN_FEATURE_TABLE} WHERE run_id = ?", (run_id,))
        conn.commit()

    @staticmethod
    def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for col in DN_FEATURE_COLUMNS:
            val = row.get(col)
            if val is None or val == "":
                out[col] = None
            elif col in ("is_gotobi", "executed", "wft_window", "is_oos"):
                out[col] = int(bool(val)) if col != "wft_window" else int(val)
            elif col in ("weekday", "month", "hour", "minute", "llm_confidence", "holding_minutes"):
                try:
                    out[col] = int(val)
                except (TypeError, ValueError):
                    out[col] = None
            elif col in ENTRY_COLUMNS and col not in (
                "trade_id",
                "run_id",
                "symbol",
                "direction",
                "entry_time",
                "session",
                "setup_type",
                "trend_direction",
                "llm_decision",
                "llm_reason",
                "decision_source",
                "win_loss",
                "exit_time",
                "ev_bucket",
            ):
                try:
                    out[col] = float(val)
                except (TypeError, ValueError):
                    out[col] = None
            else:
                out[col] = str(val) if val is not None else None
        return out

    def insert_entry(self, row: dict[str, Any]) -> None:
        data = self._normalize_row(row)
        cols = list(ENTRY_COLUMNS)
        placeholders = ", ".join("?" for _ in cols)
        col_sql = ", ".join(cols)
        values = [data.get(c) for c in cols]
        conn = self.connect()
        conn.execute(
            f"INSERT OR REPLACE INTO {DN_FEATURE_TABLE} ({col_sql}, updated_at) "
            f"VALUES ({placeholders}, datetime('now'))",
            values,
        )
        conn.commit()

    def update_exit(self, *, run_id: str, trade_id: str, patch: dict[str, Any]) -> None:
        data = self._normalize_row({**patch, "run_id": run_id, "trade_id": trade_id})
        sets = [f"{col} = ?" for col in EXIT_COLUMNS if col in patch or col == "executed"]
        if not sets:
            return
        sets.append("updated_at = datetime('now')")
        values = [data.get(col) for col in EXIT_COLUMNS if col in patch or col == "executed"]
        conn = self.connect()
        conn.execute(
            f"UPDATE {DN_FEATURE_TABLE} SET {', '.join(sets)} WHERE run_id = ? AND trade_id = ?",
            (*values, run_id, trade_id),
        )
        conn.commit()

    def fetch_all(self, *, run_id: str | None = None) -> list[dict[str, Any]]:
        conn = self.connect()
        if run_id:
            cur = conn.execute(
                f"SELECT * FROM {DN_FEATURE_TABLE} WHERE run_id = ? ORDER BY entry_time",
                (run_id,),
            )
        else:
            cur = conn.execute(f"SELECT * FROM {DN_FEATURE_TABLE} ORDER BY entry_time")
        rows = []
        for row in cur.fetchall():
            d = {k: row[k] for k in row.keys() if k in DN_FEATURE_COLUMNS}
            rows.append(d)
        return rows

    def export_csv(
        self,
        output_path: Path,
        *,
        run_ids: Iterable[str] | None = None,
        is_oos: int | None = None,
    ) -> int:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self.connect()
        clauses: list[str] = []
        params: list[Any] = []
        if run_ids:
            run_list = list(run_ids)
            placeholders = ", ".join("?" for _ in run_list)
            clauses.append(f"run_id IN ({placeholders})")
            params.extend(run_list)
        if is_oos is not None:
            clauses.append("is_oos = ?")
            params.append(int(is_oos))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        cur = conn.execute(
            f"SELECT {', '.join(DN_FEATURE_COLUMNS)} FROM {DN_FEATURE_TABLE}{where} ORDER BY entry_time",
            tuple(params),
        )
        with output_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(DN_FEATURE_COLUMNS))
            writer.writeheader()
            count = 0
            for row in cur.fetchall():
                writer.writerow({col: row[col] for col in DN_FEATURE_COLUMNS})
                count += 1
        return count
