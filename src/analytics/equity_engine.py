"""Equity curve generation from trade history."""
from __future__ import annotations

from typing import Any

import pandas as pd

import feature_engineering as fe

DEFAULT_START = float(fe.STARTING_EQUITY)
DEFAULT_RISK = 0.025


def build_equity_curve(
    trades: pd.DataFrame,
    *,
    starting_equity: float = DEFAULT_START,
    risk_pct: float = DEFAULT_RISK,
    r_column: str = "profit_r",
    lot_column: str = "lot_factor",
    time_column: str = "timestamp",
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["date", "equity", "drawdown", "peak"])

    work = trades.copy()
    work[time_column] = pd.to_datetime(work[time_column], errors="coerce")
    work = work.dropna(subset=[time_column]).sort_values(time_column)
    if "trade_result" in work.columns:
        work = work[work["trade_result"] != "NOT_EXECUTED"]

    equity = starting_equity
    rows: list[dict[str, Any]] = []
    peak = starting_equity

    for row in work.itertuples(index=False):
        eq_before = equity
        profit_r = float(getattr(row, r_column, 0.0) or 0.0)
        lot_factor = float(getattr(row, lot_column, 1.0) or 1.0)
        equity = eq_before * (1.0 + risk_pct * lot_factor * profit_r)
        peak = max(peak, equity)
        dd_pct = ((peak - equity) / peak * 100.0) if peak > 0 else 0.0
        ts = getattr(row, time_column)
        rows.append(
            {
                "date": pd.Timestamp(ts),
                "equity": equity,
                "drawdown": dd_pct,
                "peak": peak,
                "equity_before_trade": eq_before,
                "equity_after_trade": equity,
            }
        )

    curve = pd.DataFrame(rows)
    return curve.reset_index(drop=True)


def attach_equity_columns(
    trades: pd.DataFrame,
    *,
    starting_equity: float = DEFAULT_START,
    risk_pct: float = DEFAULT_RISK,
) -> pd.DataFrame:
    curve = build_equity_curve(trades, starting_equity=starting_equity, risk_pct=risk_pct)
    if curve.empty:
        out = trades.copy()
        out["equity_before_trade"] = starting_equity
        out["equity_after_trade"] = starting_equity
        return out
    out = trades.copy().reset_index(drop=True)
    out["equity_before_trade"] = curve["equity_before_trade"].values[: len(out)]
    out["equity_after_trade"] = curve["equity_after_trade"].values[: len(out)]
    return out
