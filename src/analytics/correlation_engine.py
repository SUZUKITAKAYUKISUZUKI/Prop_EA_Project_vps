"""Strategy and symbol correlation engines."""
from __future__ import annotations

import pandas as pd


def _monthly_r_matrix(trades: pd.DataFrame, group_col: str, r_col: str = "profit_r") -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    work = trades.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"])
    if "trade_result" in work.columns:
        work = work[work["trade_result"] != "NOT_EXECUTED"]
    work[r_col] = pd.to_numeric(work[r_col], errors="coerce").fillna(0.0)
    monthly = (
        work.assign(month=work["timestamp"].dt.to_period("M"))
        .groupby([group_col, "month"])[r_col]
        .sum()
        .unstack(level=0)
        .fillna(0.0)
    )
    monthly.index = monthly.index.astype(str)
    return monthly


def strategy_correlation(trades: pd.DataFrame, *, strategy_col: str = "setup_type") -> pd.DataFrame:
    matrix = _monthly_r_matrix(trades, strategy_col)
    if matrix.shape[1] < 2:
        return matrix.corr(method="pearson") if not matrix.empty else pd.DataFrame()
    return matrix.corr(method="pearson")


def symbol_correlation(trades: pd.DataFrame, *, symbol_col: str = "pair") -> pd.DataFrame:
    matrix = _monthly_r_matrix(trades, symbol_col)
    if matrix.shape[1] < 2:
        return matrix.corr(method="pearson") if not matrix.empty else pd.DataFrame()
    return matrix.corr(method="pearson")


def rolling_corr(
    series_a: pd.Series,
    series_b: pd.Series,
    *,
    window: int = 12,
) -> pd.Series:
    a = pd.to_numeric(series_a, errors="coerce").fillna(0.0)
    b = pd.to_numeric(series_b, errors="coerce").fillna(0.0)
    aligned = pd.concat([a, b], axis=1).dropna()
    if aligned.shape[0] < window:
        return pd.Series(dtype=float)
    return aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])
