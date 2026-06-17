"""Forex Tester 6 CSV importer for market_data.db."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from src.database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

FILENAME_RE = re.compile(
    r"^(?P<symbol>[A-Za-z0-9]+)_(?P<timeframe>M\d+|H\d+|D\d+|W\d+|MN\d+)$",
    re.IGNORECASE,
)

COLUMN_ALIASES = {
    "date": {"date", "<date>", "datetime", "time", "dt", "timestamp"},
    "time": {"time", "<time>"},
    "open": {"open", "<open>", "o"},
    "high": {"high", "<high>", "h"},
    "low": {"low", "<low>", "l"},
    "close": {"close", "<close>", "c"},
    "volume": {"volume", "<volume>", "vol", "tickvolume", "tick volume"},
}


def parse_symbol_timeframe(path: Path) -> tuple[str, str]:
    match = FILENAME_RE.match(path.stem)
    if not match:
        raise ValueError(f"Cannot parse symbol/timeframe from filename: {path.name}")
    symbol = match.group("symbol").upper()
    timeframe = match.group("timeframe").upper()
    return symbol, timeframe


def _normalize_header(name: str) -> str:
    return str(name).strip().lower().replace(" ", "").replace("_", "")


def _map_columns(columns: list[str]) -> dict[str, str]:
    normalized = {_normalize_header(c): c for c in columns}
    mapped: dict[str, str] = {}
    for target, aliases in COLUMN_ALIASES.items():
        norm_aliases = {_normalize_header(a) for a in aliases}
        for norm, original in normalized.items():
            if norm in norm_aliases:
                mapped[target] = original
                break
    required = {"open", "high", "low", "close"}
    if not required <= mapped.keys():
        raise ValueError(f"Missing OHLC columns. Found: {columns}")
    if "date" not in mapped and "time" not in mapped:
        raise ValueError(f"Missing datetime columns. Found: {columns}")
    return mapped


def _row_datetime(row: pd.Series, colmap: dict[str, str]) -> str:
    if "date" in colmap and "time" in colmap and colmap["date"] != colmap["time"]:
        date_val = str(row[colmap["date"]]).strip()
        time_val = str(row[colmap["time"]]).strip()
        if "." in date_val and len(date_val.split(".")[0]) == 4:
            date_val = date_val.replace(".", "-")
        dt_text = f"{date_val} {time_val}"
    else:
        dt_text = str(row[colmap.get("date", colmap["time"])]).strip()
        dt_text = dt_text.replace(".", "-").replace("/", "-")
    ts = pd.to_datetime(dt_text, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid datetime: {dt_text}")
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _rows_from_chunk(
    chunk: pd.DataFrame,
    colmap: dict[str, str],
    symbol: str,
    timeframe: str,
) -> list[tuple]:
    rows: list[tuple] = []
    vol_col = colmap.get("volume")
    for _, row in chunk.iterrows():
        try:
            dt = _row_datetime(row, colmap)
            rows.append(
                (
                    symbol,
                    timeframe,
                    dt,
                    float(row[colmap["open"]]),
                    float(row[colmap["high"]]),
                    float(row[colmap["low"]]),
                    float(row[colmap["close"]]),
                    float(row[vol_col]) if vol_col and pd.notna(row.get(vol_col)) else None,
                )
            )
        except Exception:
            continue
    return rows


def import_forex_tester_csv(
    db: DatabaseManager,
    path: Path,
    *,
    chunk_size: int = 100_000,
    ignore_duplicates: bool = True,
) -> int:
    symbol, timeframe = parse_symbol_timeframe(path)
    header = pd.read_csv(path, nrows=0)
    colmap = _map_columns(list(header.columns))
    inserted = 0
    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        batch = _rows_from_chunk(chunk, colmap, symbol, timeframe)
        if batch:
            inserted += db.insert_candles_batch(batch, ignore_duplicates=ignore_duplicates)
    logger.info(
        "Imported market data %s (%s %s): %d rows accepted",
        path.name,
        symbol,
        timeframe,
        inserted,
    )
    return inserted


def discover_market_csv_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.csv")):
            if FILENAME_RE.match(path.stem):
                files.append(path)
    return files


def import_all_market_csvs(
    db: DatabaseManager,
    roots: list[Path],
    *,
    chunk_size: int = 100_000,
) -> dict[str, int | list[str]]:
    summary: dict[str, int | list[str]] = {
        "files": 0,
        "rows_inserted": 0,
        "failures": [],
    }
    failures: list[str] = summary["failures"]  # type: ignore[assignment]
    for path in discover_market_csv_files(roots):
        summary["files"] = int(summary["files"]) + 1
        try:
            rows = import_forex_tester_csv(db, path, chunk_size=chunk_size)
            summary["rows_inserted"] = int(summary["rows_inserted"]) + rows
        except Exception as exc:
            logger.exception("Market import failed for %s: %s", path, exc)
            failures.append(f"{path.as_posix()}: {exc}")
    return summary
