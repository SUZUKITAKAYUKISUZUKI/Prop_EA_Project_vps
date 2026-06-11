"""
NumPy helpers for LGR / reversal feature build (BT + WFT hot path).

Pandas equivalents remain in ``reversal_feature_helpers.py`` for legacy callers.
"""

from __future__ import annotations

import numpy as np

from strategies.bt_ohlcv import OhlcvArrays, ts_ns_to_pd
from strategies.archive.cspa_arrays import atr_at_index
from strategies.archive.cspa_scan_hot import (
    minutes_from_session_open_ns,
    resolve_cspa_session_type_ns,
    volatility_percentile_np,
)

ADR_LOOKBACK_DAYS = 14
ATR_PERIOD = 14


def _bar_hour_ns(ts_ns: int) -> int:
    return int(ts_ns_to_pd(ts_ns).hour)


def spread_percentile_np(spread_pips: float, *, ref_pips: float = 3.0) -> float:
    return round(min(max(spread_pips / ref_pips, 0.0), 1.0), 4)


def compute_adr_remaining_np(
    arr: OhlcvArrays,
    bar_index: int,
    atr: float,
    day_norm: np.ndarray,
) -> float:
    if bar_index < 1 or atr <= 0:
        return 1.0
    current_day = day_norm[bar_index]
    day_mask = day_norm == current_day
    day_high = float(np.max(arr.high[day_mask]))
    day_low = float(np.min(arr.low[day_mask]))
    adr_used = day_high - day_low

    unique_days = np.unique(day_norm[: bar_index + 1])
    day_pos = np.where(unique_days == current_day)[0]
    if len(day_pos) == 0:
        return 1.0
    end = int(day_pos[0])
    start = max(0, end - ADR_LOOKBACK_DAYS)
    ranges: list[float] = []
    for day in unique_days[start:end]:
        mask = day_norm == day
        if int(mask.sum()) >= 4:
            ranges.append(float(np.max(arr.high[mask]) - np.min(arr.low[mask])))
    if not ranges:
        return 1.0
    adr_avg = float(np.mean(ranges))
    if adr_avg <= 0:
        return 1.0
    return max(0.0, (adr_avg - adr_used) / adr_avg)


def adr_used_fraction_np(
    arr: OhlcvArrays,
    bar_index: int,
    atr: float,
    day_norm: np.ndarray,
) -> float:
    if bar_index < 1 or atr <= 0:
        return 0.0
    remaining = compute_adr_remaining_np(arr, bar_index, atr, day_norm)
    return max(0.0, min(1.0, 1.0 - remaining))


def liquidity_distances_np(
    arr: OhlcvArrays,
    bar_index: int,
    price: float,
    atr: float,
    day_norm: np.ndarray,
    bar_hours: np.ndarray,
) -> dict[str, float]:
    if bar_index < 0 or atr <= 0:
        return {
            "distance_daily_high": 0.0,
            "distance_daily_low": 0.0,
            "distance_session_high": 0.0,
            "distance_session_low": 0.0,
        }
    current_day = day_norm[bar_index]
    day_mask = day_norm == current_day
    day_high = float(np.max(arr.high[day_mask]))
    day_low = float(np.min(arr.low[day_mask]))
    session = resolve_cspa_session_type_ns(int(arr.datetime_ns[bar_index]))
    day_indices = np.where(day_mask)[0]
    if session == "LONDON":
        hour_mask = bar_hours[day_indices] >= 8
        hour_mask &= bar_hours[day_indices] < 17
    elif session == "NY":
        hour_mask = bar_hours[day_indices] >= 13
        hour_mask &= bar_hours[day_indices] < 22
    elif session == "ASIA":
        hour_mask = bar_hours[day_indices] < 8
    else:
        hour_mask = np.ones(len(day_indices), dtype=bool)
    sess_indices = day_indices[hour_mask] if np.any(hour_mask) else day_indices
    sess_high = float(np.max(arr.high[sess_indices]))
    sess_low = float(np.min(arr.low[sess_indices]))
    return {
        "distance_daily_high": (day_high - price) / atr,
        "distance_daily_low": (price - day_low) / atr,
        "distance_session_high": (sess_high - price) / atr,
        "distance_session_low": (price - sess_low) / atr,
    }


def compute_recovery_close_ratio_np(
    arr: OhlcvArrays,
    trigger_idx: int,
    recovery_idx: int,
    *,
    direction: str = "BUY",
) -> float:
    if recovery_idx < trigger_idx:
        return 0.0
    opens = arr.open[trigger_idx : recovery_idx + 1]
    closes = arr.close[trigger_idx : recovery_idx + 1]
    body = np.abs(closes - opens)
    valid = body > 0
    if not np.any(valid):
        return 0.0
    bull = closes > opens
    bear = closes < opens
    if direction == "BUY":
        favorable = float(body[bull].sum())
        counter = float(body[bear].sum())
    else:
        favorable = float(body[bear].sum())
        counter = float(body[bull].sum())
    if counter <= 0:
        return favorable if favorable > 0 else 0.0
    return favorable / counter


def volatility_percentile_for_bar(
    atr_series: np.ndarray,
    bar_index: int,
    *,
    lookback: int = 120,
    atr_period: int = ATR_PERIOD,
) -> float:
    pct = volatility_percentile_np(
        atr_series, bar_index, lookback=lookback, atr_period=atr_period
    )
    return round(pct * 100.0, 4)


def atr_at_np(atr_series: np.ndarray, bar_index: int) -> float:
    return atr_at_index(atr_series, bar_index)


def build_bar_hours(arr: OhlcvArrays) -> np.ndarray:
    return np.array([_bar_hour_ns(int(t)) for t in arr.datetime_ns], dtype=np.int32)


__all__ = [
    "adr_used_fraction_np",
    "atr_at_np",
    "build_bar_hours",
    "compute_adr_remaining_np",
    "compute_recovery_close_ratio_np",
    "liquidity_distances_np",
    "minutes_from_session_open_ns",
    "resolve_cspa_session_type_ns",
    "spread_percentile_np",
    "volatility_percentile_for_bar",
]
