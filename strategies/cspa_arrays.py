"""
strategies/cspa_arrays.py — Phase 1: OHLCV numpy views for CSPA scan hot path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from strategies.cspa import ImpulseLeg, StagnationCluster, TradeDirection

_FIB_RETRACE_MAX = float(os.getenv("CSPA_FIB_RETRACE_MAX", "0.618"))
_STAGNATION_MAX_BODY_ATR = float(os.getenv("CSPA_STAGNATION_MAX_BODY_ATR", "0.35"))
_STAGNATION_MIN_BARS = int(os.getenv("CSPA_STAGNATION_MIN_BARS", "1"))


def timestamp_from_ns(ts_ns: int) -> datetime:
    """Naive server-time ``datetime`` from nanosecond epoch (matches OHLCV CSV)."""
    return datetime.fromtimestamp(int(ts_ns) / 1_000_000_000)


def datetime_ns_from_column(values: Any) -> np.ndarray:
    """Convert prepared OHLCV datetime column to int64 nanoseconds without pandas."""
    arr = np.asarray(values)
    if arr.dtype.kind == "M":
        return arr.astype("datetime64[ns]").astype(np.int64)
    if arr.dtype.kind in ("O", "U", "S"):
        parsed = np.array([np.datetime64(v) for v in arr], dtype="datetime64[ns]")
        return parsed.astype(np.int64)
    raise TypeError(f"unsupported datetime column dtype: {arr.dtype}")


def years_from_datetime_ns(datetime_ns: np.ndarray) -> np.ndarray:
    """Calendar year per bar (for Phase 4 year-chunk parallel)."""
    if len(datetime_ns) == 0:
        return np.array([], dtype=np.int32)
    year_floor = datetime_ns.astype("datetime64[ns]").astype("datetime64[Y]")
    return np.fromiter(
        (int(str(value)[:4]) for value in year_floor),
        dtype=np.int32,
        count=len(year_floor),
    )


@dataclass(frozen=True, slots=True)
class OhlcvArrays:
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    datetime_ns: np.ndarray

    @property
    def length(self) -> int:
        return int(len(self.close))

    @classmethod
    def from_prepared_df(cls, df: Any) -> OhlcvArrays:
        from strategies.bt_ohlcv import lookup_ohlcv

        found = lookup_ohlcv(df)
        if found is not None:
            return found
        vol = (
            np.asarray(df["volume"], dtype=np.float64)
            if "volume" in df.columns
            else np.zeros(len(df), dtype=np.float64)
        )
        return cls(
            open=np.asarray(df["open"], dtype=np.float64),
            high=np.asarray(df["high"], dtype=np.float64),
            low=np.asarray(df["low"], dtype=np.float64),
            close=np.asarray(df["close"], dtype=np.float64),
            volume=vol,
            datetime_ns=datetime_ns_from_column(df["datetime"]),
        )


@dataclass(slots=True)
class CspaScanArrays:
    trigger: OhlcvArrays
    structure: OhlcvArrays
    bias: OhlcvArrays
    trigger_atr: np.ndarray
    structure_atr: np.ndarray
    bias_atr: np.ndarray
    bias_ema50: np.ndarray
    struct_idx_by_bar: np.ndarray
    bias_idx_by_bar: np.ndarray


def series_to_float64(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64)


def build_scan_arrays_from_ohlcv(
    trigger: OhlcvArrays,
    structure: OhlcvArrays,
    bias: OhlcvArrays,
    *,
    trigger_atr: np.ndarray,
    structure_atr: np.ndarray,
    bias_atr: np.ndarray,
    bias_ema50: np.ndarray,
    struct_idx_by_bar: np.ndarray,
    bias_idx_by_bar: np.ndarray,
) -> CspaScanArrays:
    return CspaScanArrays(
        trigger=trigger,
        structure=structure,
        bias=bias,
        trigger_atr=series_to_float64(trigger_atr),
        structure_atr=series_to_float64(structure_atr),
        bias_atr=series_to_float64(bias_atr),
        bias_ema50=series_to_float64(bias_ema50),
        struct_idx_by_bar=np.asarray(struct_idx_by_bar, dtype=np.int64),
        bias_idx_by_bar=np.asarray(bias_idx_by_bar, dtype=np.int64),
    )


def build_scan_arrays(
    trigger: Any,
    structure: Any,
    bias: Any,
    *,
    trigger_atr: np.ndarray,
    structure_atr: np.ndarray,
    bias_atr: np.ndarray,
    bias_ema50: np.ndarray,
    struct_idx_by_bar: np.ndarray,
    bias_idx_by_bar: np.ndarray,
) -> CspaScanArrays:
    return build_scan_arrays_from_ohlcv(
        OhlcvArrays.from_prepared_df(trigger),
        OhlcvArrays.from_prepared_df(structure),
        OhlcvArrays.from_prepared_df(bias),
        trigger_atr=trigger_atr,
        structure_atr=structure_atr,
        bias_atr=bias_atr,
        bias_ema50=bias_ema50,
        struct_idx_by_bar=struct_idx_by_bar,
        bias_idx_by_bar=bias_idx_by_bar,
    )


def atr_at_index(atr: np.ndarray, bar_index: int) -> float:
    if bar_index < 0 or bar_index >= len(atr):
        return 0.0
    val = float(atr[bar_index])
    return val if val > 0.0 else 0.0


def body_size(open_: float, close: float) -> float:
    return abs(close - open_)


def measure_retrace_ratio_np(
    structure: OhlcvArrays,
    impulse: ImpulseLeg,
    bar_index: int,
) -> float:
    if impulse.impulse_size <= 0:
        return 0.0
    start = impulse.end_index
    end = bar_index + 1
    if start >= end or start >= structure.length:
        return 0.0
    if impulse.direction == "UP":
        correction_low = float(np.min(structure.low[start:end]))
        return (impulse.end_price - correction_low) / impulse.impulse_size
    correction_high = float(np.max(structure.high[start:end]))
    return (correction_high - impulse.end_price) / impulse.impulse_size


def m1_over_retraces_structure_np(
    trigger: OhlcvArrays,
    structure: OhlcvArrays,
    trigger_index: int,
    impulse: ImpulseLeg,
) -> bool:
    if impulse.impulse_size <= 0 or trigger_index < 0 or trigger_index >= trigger.length:
        return False
    if impulse.end_index < 0 or impulse.end_index >= structure.length:
        return False
    start_ts = structure.datetime_ns[impulse.end_index]
    start_i = int(np.searchsorted(trigger.datetime_ns, start_ts, side="left"))
    end_i = trigger_index + 1
    if start_i >= end_i:
        return False
    if impulse.direction == "UP":
        deepest = float(np.min(trigger.low[start_i:end_i]))
        ratio = (impulse.end_price - deepest) / impulse.impulse_size
    else:
        highest = float(np.max(trigger.high[start_i:end_i]))
        ratio = (highest - impulse.end_price) / impulse.impulse_size
    return ratio > _FIB_RETRACE_MAX


def detect_stagnation_cluster_np(
    trigger: OhlcvArrays,
    trigger_atr: np.ndarray,
    end_index: int,
    direction: TradeDirection,
    *,
    max_bars: int,
) -> StagnationCluster | None:
    from strategies.cspa import StagnationCluster

    if end_index < 1:
        return None

    stagnation_indices: list[int] = []
    for i in range(end_index - 1, max(end_index - max_bars - 1, -1), -1):
        if i < 0:
            break
        atr = atr_at_index(trigger_atr, i)
        if atr <= 0:
            break
        open_ = float(trigger.open[i])
        close = float(trigger.close[i])
        high = float(trigger.high[i])
        low = float(trigger.low[i])
        body = body_size(open_, close)
        body_atr = body / atr

        is_small_body = body_atr <= _STAGNATION_MAX_BODY_ATR
        rejection = False
        if direction == "BUY":
            lower_wick = min(open_, close) - low
            rejection = lower_wick >= body * 1.5 and body_atr <= _STAGNATION_MAX_BODY_ATR * 1.2
        else:
            upper_wick = high - max(open_, close)
            rejection = upper_wick >= body * 1.5 and body_atr <= _STAGNATION_MAX_BODY_ATR * 1.2

        if is_small_body or rejection:
            stagnation_indices.insert(0, i)
        else:
            break

    if len(stagnation_indices) < _STAGNATION_MIN_BARS:
        return None

    bodies: list[float] = []
    zone_high = float("-inf")
    zone_low = float("inf")
    for i in stagnation_indices:
        atr = atr_at_index(trigger_atr, i)
        open_ = float(trigger.open[i])
        close = float(trigger.close[i])
        bodies.append(body_size(open_, close) / atr if atr > 0 else 0.0)
        zone_high = max(zone_high, float(trigger.high[i]))
        zone_low = min(zone_low, float(trigger.low[i]))

    solid = len(stagnation_indices) >= 2 or (
        len(stagnation_indices) == 1 and bodies[0] <= _STAGNATION_MAX_BODY_ATR * 0.8
    )
    return StagnationCluster(
        start_index=stagnation_indices[0],
        end_index=stagnation_indices[-1],
        bar_count=len(stagnation_indices),
        avg_body_atr=round(sum(bodies) / len(bodies), 4),
        zone_high=zone_high,
        zone_low=zone_low,
        solid_ground=solid,
    )


def compute_atr_np(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
) -> np.ndarray:
    """ATR rolling mean of true range (matches ``market_utils.compute_atr``)."""
    n = len(close)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    if n > 1:
        prev = close[:-1]
        tr[1:] = np.maximum(
            high[1:] - low[1:],
            np.maximum(np.abs(high[1:] - prev), np.abs(low[1:] - prev)),
        )
    if n >= period:
        kernel = np.ones(period, dtype=np.float64) / period
        out[period - 1 :] = np.convolve(tr, kernel, mode="valid")
    return out


def compute_ema_np(close: np.ndarray, span: int) -> np.ndarray:
    out = np.empty(len(close), dtype=np.float64)
    if len(close) == 0:
        return out
    alpha = 2.0 / (span + 1.0)
    out[0] = float(close[0])
    for i in range(1, len(close)):
        out[i] = alpha * float(close[i]) + (1.0 - alpha) * out[i - 1]
    return out


def build_adr_cache_np(structure: OhlcvArrays):
    """Build ``_StructureAdrCache`` from numpy OHLCV (no pandas)."""
    from strategies.cspa import CSPA_ADR_LOOKBACK_DAYS, _StructureAdrCache

    dt = structure.datetime_ns.astype("datetime64[ns]")
    day_norm = dt.astype("datetime64[D]")
    highs = structure.high
    lows = structure.low
    n = structure.length
    day_start = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        day_start[i] = day_start[i - 1] if day_norm[i] == day_norm[i - 1] else i
    daily_range: dict[np.datetime64, float] = {}
    for day in np.unique(day_norm):
        mask = day_norm == day
        if int(mask.sum()) >= 4:
            daily_range[day] = float(highs[mask].max() - lows[mask].min())
    return _StructureAdrCache(
        day_norm=day_norm,
        day_start=day_start,
        highs=highs,
        lows=lows,
        daily_range=daily_range,
    )


def build_mtf_index_maps(
    trigger: OhlcvArrays,
    structure: OhlcvArrays,
    bias: OhlcvArrays,
) -> tuple[np.ndarray, np.ndarray]:
    struct_idx = np.searchsorted(structure.datetime_ns, trigger.datetime_ns, side="right") - 1
    bias_idx = np.searchsorted(bias.datetime_ns, trigger.datetime_ns, side="right") - 1
    return struct_idx.astype(np.int64), bias_idx.astype(np.int64)
