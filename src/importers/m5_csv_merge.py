"""Merge FT6-format M5 CSV exports into local backtest data files."""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

FT6_HEADER = ("<TICKER>", "<DTYYYYMMDD>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>")


def read_ft6_dataframe(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=["ticker", "date", "time", "open", "high", "low", "close", "vol"])
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header:
            return pd.DataFrame()
        cols = {name.strip("<>").upper(): idx for idx, name in enumerate(header)}
        for row in reader:
            if not row or len(row) < 8:
                continue
            rows.append(
                {
                    "ticker": row[cols["TICKER"]].strip(),
                    "date": row[cols["DTYYYYMMDD"]].strip(),
                    "time": row[cols["TIME"]].strip().zfill(4),
                    "open": float(row[cols["OPEN"]]),
                    "high": float(row[cols["HIGH"]]),
                    "low": float(row[cols["LOW"]]),
                    "close": float(row[cols["CLOSE"]]),
                    "vol": float(row[cols.get("VOL", cols.get("VOLUME", 7))]),
                }
            )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["date"] + df["time"], format="%Y%m%d%H%M")
    return df.sort_values("datetime").reset_index(drop=True)


def write_ft6_dataframe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(FT6_HEADER)
        for row in df.itertuples(index=False):
            dt = row.datetime
            writer.writerow(
                [
                    row.ticker,
                    dt.strftime("%Y%m%d"),
                    dt.strftime("%H%M"),
                    f"{row.open:.5f}",
                    f"{row.high:.5f}",
                    f"{row.low:.5f}",
                    f"{row.close:.5f}",
                    int(row.vol),
                ]
            )


def merge_ft6_csv(target: Path, incoming: Path, *, symbol: str) -> dict[str, int | str]:
    """Append new M5 bars from ``incoming`` into ``target`` (dedupe by datetime)."""
    inc = read_ft6_dataframe(incoming)
    if inc.empty:
        return {"merged": 0, "total": 0, "last_bar": ""}

    inc["ticker"] = symbol.upper()
    existing = read_ft6_dataframe(target) if target.is_file() else pd.DataFrame()
    if existing.empty:
        combined = inc
        added = len(inc)
    else:
        combined = pd.concat([existing, inc], ignore_index=True)
        combined = combined.drop_duplicates(subset=["datetime"], keep="last")
        combined = combined.sort_values("datetime").reset_index(drop=True)
        added = len(combined) - len(existing)

    write_ft6_dataframe(target, combined)
    last_bar = str(combined["datetime"].iloc[-1]) if len(combined) else ""
    logger.info(
        "Merged M5 %s: +%d bars (total=%d, last=%s) -> %s",
        symbol,
        added,
        len(combined),
        last_bar,
        target.name,
    )
    return {"merged": added, "total": len(combined), "last_bar": last_bar}
