"""
Shared reversal feature helpers (LGR and legacy WR analysis).

WR 本体は strategies/archive/ へ移動。LGR が利用する特徴量計算のみここに残す。
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
import pandas as pd

from strategies.cspa import resolve_cspa_session_type
from strategies.market_utils import compute_atr

SessionType = Literal["ASIA", "LONDON", "NY", "OFF_HOURS"]
TradeDirection = Literal["BUY", "SELL"]

VOLATILITY_LOOKBACK = int(os.getenv("REVERSAL_VOLATILITY_LOOKBACK", "120"))
ADR_LOOKBACK_DAYS = int(os.getenv("REVERSAL_ADR_LOOKBACK_DAYS", "14"))
ATR_PERIOD = int(os.getenv("REVERSAL_ATR_PERIOD", "14"))


def spread_percentile(spread_pips: float, *, ref_pips: float = 3.0) -> float:
    return round(min(max(spread_pips / ref_pips, 0.0), 1.0), 4)


def volatility_percentile(work: pd.DataFrame, bar_index: int, atr_series: pd.Series) -> float:
    start = max(ATR_PERIOD, bar_index - VOLATILITY_LOOKBACK + 1)
    if bar_index < start:
        return 50.0
    window = atr_series.iloc[start : bar_index + 1].dropna()
    if window.empty:
        return 50.0
    current = float(atr_series.iloc[bar_index])
    if not np.isfinite(current):
        return 50.0
    rank = (window <= current).sum()
    return round(100.0 * rank / len(window), 4)


def compute_adr_remaining(work: pd.DataFrame, bar_index: int, atr: float) -> float:
    if bar_index < 1 or atr <= 0:
        return 1.0
    day_norm = pd.to_datetime(work["datetime"]).dt.normalize()
    current_day = day_norm.iloc[bar_index]
    day_mask = day_norm == current_day
    day_high = float(work.loc[day_mask, "high"].max())
    day_low = float(work.loc[day_mask, "low"].min())
    adr_used = day_high - day_low

    unique_days = day_norm.unique()
    day_pos = np.where(unique_days == np.datetime64(current_day))[0]
    if len(day_pos) == 0:
        return 1.0
    end = int(day_pos[0])
    start = max(0, end - ADR_LOOKBACK_DAYS)
    ranges: list[float] = []
    for day in unique_days[start:end]:
        mask = day_norm == day
        if int(mask.sum()) >= 4:
            ranges.append(float(work.loc[mask, "high"].max() - work.loc[mask, "low"].min()))
    if not ranges:
        return 1.0
    adr_avg = float(np.mean(ranges))
    if adr_avg <= 0:
        return 1.0
    return max(0.0, (adr_avg - adr_used) / adr_avg)


def adr_used_fraction(work: pd.DataFrame, bar_index: int, atr: float) -> float:
    if bar_index < 1 or atr <= 0:
        return 0.0
    remaining = compute_adr_remaining(work, bar_index, atr)
    return max(0.0, min(1.0, 1.0 - remaining))


def minutes_from_session_open(ts: pd.Timestamp, session: SessionType) -> int:
    ts = pd.Timestamp(ts)
    if session == "LONDON":
        open_ts = ts.normalize() + pd.Timedelta(hours=8)
    elif session == "NY":
        open_ts = ts.normalize() + pd.Timedelta(hours=13)
    elif session == "ASIA":
        open_ts = ts.normalize() + pd.Timedelta(hours=0)
    else:
        return 0
    if ts < open_ts:
        open_ts -= pd.Timedelta(days=1)
    return max(0, int((ts - open_ts).total_seconds() // 60))


def liquidity_distances(
    work: pd.DataFrame,
    bar_index: int,
    price: float,
    atr: float,
) -> dict[str, float]:
    if bar_index < 0 or atr <= 0:
        return {
            "distance_daily_high": 0.0,
            "distance_daily_low": 0.0,
            "distance_session_high": 0.0,
            "distance_session_low": 0.0,
        }
    day_norm = pd.to_datetime(work["datetime"]).dt.normalize()
    current_day = day_norm.iloc[bar_index]
    day_mask = day_norm == current_day
    day_slice = work.loc[day_mask]
    day_high = float(day_slice["high"].max())
    day_low = float(day_slice["low"].min())
    session = resolve_cspa_session_type(pd.Timestamp(work.iloc[bar_index]["datetime"]))
    if session == "LONDON":
        hour_mask = day_slice["datetime"].apply(lambda t: 8 <= pd.Timestamp(t).hour < 17)
    elif session == "NY":
        hour_mask = day_slice["datetime"].apply(lambda t: 13 <= pd.Timestamp(t).hour < 22)
    elif session == "ASIA":
        hour_mask = day_slice["datetime"].apply(lambda t: pd.Timestamp(t).hour < 8)
    else:
        hour_mask = day_mask
    sess_slice = day_slice.loc[hour_mask] if hour_mask.any() else day_slice
    sess_high = float(sess_slice["high"].max())
    sess_low = float(sess_slice["low"].min())
    return {
        "distance_daily_high": (day_high - price) / atr,
        "distance_daily_low": (price - day_low) / atr,
        "distance_session_high": (sess_high - price) / atr,
        "distance_session_low": (price - sess_low) / atr,
    }


def compute_recovery_close_ratio(
    work: pd.DataFrame,
    trigger_idx: int,
    recovery_idx: int,
    *,
    direction: TradeDirection = "BUY",
) -> float:
    if recovery_idx < trigger_idx:
        return 0.0
    favorable = 0.0
    counter = 0.0
    for _, row in work.iloc[trigger_idx : recovery_idx + 1].iterrows():
        open_p = float(row["open"])
        close_p = float(row["close"])
        body = abs(close_p - open_p)
        if body <= 0:
            continue
        is_bull = close_p > open_p
        is_bear = close_p < open_p
        if direction == "BUY":
            if is_bull:
                favorable += body
            elif is_bear:
                counter += body
        else:
            if is_bear:
                favorable += body
            elif is_bull:
                counter += body
    if counter <= 0:
        return favorable if favorable > 0 else 0.0
    return favorable / counter


def atr_at(work: pd.DataFrame, bar_index: int, atr_series: pd.Series | None = None) -> float:
    if bar_index < 0 or bar_index >= len(work):
        return 0.0
    if atr_series is not None:
        val = float(atr_series.iloc[bar_index])
        return val if np.isfinite(val) and val > 0 else 0.0
    clipped = work.iloc[: bar_index + 1]
    atr = compute_atr(clipped, ATR_PERIOD)
    if bar_index >= len(atr):
        return 0.0
    val = float(atr.iloc[bar_index])
    return val if np.isfinite(val) else 0.0
