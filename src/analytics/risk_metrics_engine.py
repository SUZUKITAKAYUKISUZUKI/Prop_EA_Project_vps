"""Risk metrics calculations on R-multiple series."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _r_series(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return arr


def profit_factor(r_values: pd.Series | np.ndarray) -> float:
    r = _r_series(r_values)
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def expectancy(r_values: pd.Series | np.ndarray) -> float:
    r = _r_series(r_values)
    return float(r.mean()) if r.size else 0.0


def max_dd(equity: pd.Series | np.ndarray) -> float:
    eq = pd.to_numeric(equity, errors="coerce").ffill().fillna(0.0).to_numpy(dtype=np.float64)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = np.divide(peak - eq, peak, out=np.zeros_like(eq), where=peak > 0) * 100.0
    return float(dd.max())


def recovery_factor(total_r: float, max_dd_pct: float) -> float:
    if max_dd_pct <= 0:
        return float("inf") if total_r > 0 else 0.0
    return float(total_r / max_dd_pct)


def ulcer_index(equity: pd.Series | np.ndarray) -> float:
    eq = pd.to_numeric(equity, errors="coerce").ffill().fillna(0.0).to_numpy(dtype=np.float64)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd_pct = np.divide(peak - eq, peak, out=np.zeros_like(eq), where=peak > 0) * 100.0
    return float(np.sqrt(np.mean(np.square(dd_pct))))


def cagr(equity: pd.Series, periods_per_year: float = 252.0) -> float:
    if equity.empty:
        return 0.0
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    if start <= 0 or end <= 0:
        return 0.0
    years = max(len(equity) / periods_per_year, 1e-9)
    return float((end / start) ** (1.0 / years) - 1.0)


def risk_of_ruin(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    *,
    risk_fraction: float = 0.025,
    ruin_threshold_r: float = 40.0,
) -> float:
    if avg_loss >= 0 or avg_win <= 0:
        return 1.0
    edge = win_rate * avg_win + (1.0 - win_rate) * avg_loss
    if edge <= 0:
        return 1.0
    units = ruin_threshold_r / max(abs(avg_loss), 1e-9)
    p = win_rate
    q = 1.0 - p
    ratio = q / p * abs(avg_loss / avg_win)
    if ratio >= 1.0:
        return 1.0
    return float(ratio ** units)


def losing_streak(results: pd.Series | np.ndarray) -> int:
    vals = pd.Series(results).astype(str).str.upper()
    streak = 0
    best = 0
    for val in vals:
        if val == "LOSS":
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def sharpe_ratio(r_values: pd.Series | np.ndarray, *, periods: float = 252.0) -> float:
    r = _r_series(r_values)
    if r.size < 2:
        return 0.0
    std = float(r.std(ddof=1))
    if std <= 0:
        return 0.0
    return float(r.mean() / std * np.sqrt(periods))
