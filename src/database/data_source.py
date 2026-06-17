"""Canonical data lineage labels for portfolio_os.db rows."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

DataSource = Literal["BACKTEST", "WFT_OOS", "LIVE", "FORWARD_TEST"]

DATA_SOURCES: tuple[DataSource, ...] = ("BACKTEST", "WFT_OOS", "LIVE", "FORWARD_TEST")

# Bump when portfolio_os DDL changes (new tables/columns/constraints).
PORTFOLIO_DB_SCHEMA_VERSION = 4

# Bump when L6 / feature snapshot JSON shape changes (htf_lot_multiplier, vp_location_score, …).
FEATURE_LOG_SCHEMA_VERSION = 1

SOURCE_CHECK_SQL = "CHECK (source IN ('BACKTEST','WFT_OOS','LIVE','FORWARD_TEST'))"


def normalize_source(value: str | None, *, default: DataSource = "BACKTEST") -> DataSource:
    if not value:
        return default
    upper = str(value).strip().upper()
    aliases = {
        "BT": "BACKTEST",
        "BACKTEST": "BACKTEST",
        "WFT": "WFT_OOS",
        "WFT_OOS": "WFT_OOS",
        "OOS": "WFT_OOS",
        "LIVE": "LIVE",
        "FORWARD": "FORWARD_TEST",
        "FORWARD_TEST": "FORWARD_TEST",
        "PAPER": "FORWARD_TEST",
    }
    resolved = aliases.get(upper)
    if resolved is None:
        raise ValueError(f"Invalid data source {value!r}; expected one of {DATA_SOURCES}")
    return resolved  # type: ignore[return-value]


def infer_source_from_run_type(run_type: str | None, description: str | None = None) -> DataSource:
    rt = (run_type or "").strip().lower()
    desc = (description or "").lower()
    if rt == "live":
        return "LIVE"
    if rt == "wft" or "wft" in desc or "walkforward" in desc or "walk_forward" in desc:
        return "WFT_OOS"
    if "forward" in desc or "paper" in desc:
        return "FORWARD_TEST"
    return "BACKTEST"


def infer_source_from_path(path: str | Path, *, csv_kind: str | None = None) -> DataSource:
    text = Path(path).as_posix().lower()
    kind = (csv_kind or "").lower()
    if kind == "wft" or re.search(r"(^|/)(wft|walkforward|walk_forward|oos)(/|_|\.|$)", text):
        return "WFT_OOS"
    if re.search(r"(^|/)(forward|paper|demo)(/|_|\.|$)", text):
        return "FORWARD_TEST"
    if kind == "live" or "/live/" in text or text.endswith("_live.csv"):
        return "LIVE"
    return "BACKTEST"


def resolve_feature_schema_version(payload: dict | None = None) -> int:
    if isinstance(payload, dict):
        raw = payload.get("schema_version")
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                pass
    return FEATURE_LOG_SCHEMA_VERSION
