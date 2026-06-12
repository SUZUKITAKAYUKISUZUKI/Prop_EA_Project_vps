"""
strategies/lbo_detector.py — London Break Out (LBO) session / breakout / retest detection.

Tokyo range (JST 06:00–15:00) on H1, London breakout + retest on M15.
Hot paths use Numba on numpy arrays; DataFrame inputs are converted once at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from strategies.dbbs_common import atr_at, ohlcv_to_arrays, precompute_atr_series
from strategies.htf_trend_analyzer import clip_as_of
from strategies.scan_numba_util import njit

NS_PER_SEC = 1_000_000_000
NS_PER_DAY = 24 * 3600 * NS_PER_SEC
JST_OFFSET_NS = 9 * 3600 * NS_PER_SEC

TOKYO_START_JST = 6
TOKYO_END_JST = 15
LONDON_START_JST = 15
BREAKOUT_WINDOW_END_JST = 21
MAX_RETEST_BARS = 12
MIN_TOKYO_RANGE_PIPS = 5.0
MIN_BREAKOUT_PIPS = 3.0
DEFAULT_PIP_SIZE = 0.0001
ATR_PERIOD = 14
PA_PIN_BAR = "PIN_BAR"
PA_ENGULFING = "ENGULFING"
PA_INSIDE_BAR = "INSIDE_BAR_BREAK"
PA_CLOSE_ONLY = "CLOSE_ONLY"


@dataclass(frozen=True)
class TokyoRange:
    """東京セッションレンジの定義。"""

    date: date
    high: float
    low: float
    range_pips: float
    range_atr_ratio: float
    formation_bars: int
    tokyo_high_time_ratio: float
    tokyo_low_time_ratio: float
    is_valid: bool


def jst_fields_from_ns(ts_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (jst_day_ordinal, jst_hour, jst_minute) for each bar timestamp (ns)."""
    jst_ns = ts_ns.astype(np.int64) + JST_OFFSET_NS
    days = jst_ns // NS_PER_DAY
    sec_in_day = (jst_ns % NS_PER_DAY) // NS_PER_SEC
    hours = sec_in_day // 3600
    minutes = (sec_in_day % 3600) // 60
    return days.astype(np.int64), hours.astype(np.int64), minutes.astype(np.int64)


def jst_day_ordinal(ts: pd.Timestamp) -> int:
    return int((int(pd.Timestamp(ts).value) + JST_OFFSET_NS) // NS_PER_DAY)


def jst_hour_minute(ts: pd.Timestamp) -> tuple[int, int]:
    jst_ns = int(pd.Timestamp(ts).value) + JST_OFFSET_NS
    sec_in_day = (jst_ns % NS_PER_DAY) // NS_PER_SEC
    return int(sec_in_day // 3600), int((sec_in_day % 3600) // 60)


def ordinal_to_date(day_ord: int) -> date:
    base = pd.Timestamp("1970-01-01") + pd.Timedelta(days=int(day_ord))
    return base.date()


@njit(cache=True)
def _in_tokyo_session(jst_hour: int) -> bool:
    return TOKYO_START_JST <= jst_hour < TOKYO_END_JST


@njit(cache=True)
def _in_london_breakout_window(jst_hour: int) -> bool:
    return LONDON_START_JST <= jst_hour < BREAKOUT_WINDOW_END_JST


@njit(cache=True)
def _percentile_rank(values: np.ndarray, value: float) -> float:
    if len(values) == 0:
        return 0.5
    count = 0
    for i in range(len(values)):
        if values[i] <= value:
            count += 1
    return count / len(values)


@njit(cache=True)
def _volatility_percentile_at(atr_series: np.ndarray, idx: int, lookback: int) -> float:
    start = max(0, idx - lookback + 1)
    samples = atr_series[start : idx + 1]
    valid = samples[samples > 0.0]
    if len(valid) == 0:
        return 0.5
    return _percentile_rank(valid, float(atr_series[idx]))


@njit(cache=True)
def _classify_pa(
    direction: int,
    o: float,
    h: float,
    l: float,
    c: float,
    prev_o: float,
    prev_h: float,
    prev_l: float,
    prev_c: float,
    boundary: float,
    pip: float,
) -> str:
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    if direction == 1:
        if lower_wick >= body * 1.5 and lower_wick >= pip * 2.0 and c > o:
            return PA_PIN_BAR
    else:
        if upper_wick >= body * 1.5 and upper_wick >= pip * 2.0 and c < o:
            return PA_PIN_BAR
    if body > 0.0:
        prev_body = abs(prev_c - prev_o)
        if prev_body > 0.0:
            if direction == 1 and c > o and prev_c < prev_o and c >= prev_o and o <= prev_c:
                return PA_ENGULFING
            if direction == -1 and c < o and prev_c > prev_o and c <= prev_o and o >= prev_c:
                return PA_ENGULFING
    if prev_h >= prev_l and h <= prev_h and l >= prev_l and c != prev_c:
        if direction == 1 and c > boundary:
            return PA_INSIDE_BAR
        if direction == -1 and c < boundary:
            return PA_INSIDE_BAR
    return PA_CLOSE_ONLY


@njit(cache=True)
def scan_tokyo_range_core(
    high: np.ndarray,
    low: np.ndarray,
    jst_days: np.ndarray,
    jst_hours: np.ndarray,
    target_day: int,
    atr_value: float,
    pip: float,
    min_range_pips: float,
) -> tuple[float, float, float, float, int, float, float, int]:
    hi = -1.0e18
    lo = 1.0e18
    count = 0
    hi_hour = TOKYO_START_JST
    lo_hour = TOKYO_START_JST
    span = max(TOKYO_END_JST - TOKYO_START_JST, 1)
    for i in range(len(jst_days)):
        if jst_days[i] != target_day:
            continue
        hour = int(jst_hours[i])
        if not _in_tokyo_session(hour):
            continue
        count += 1
        if high[i] > hi:
            hi = high[i]
            hi_hour = hour
        if low[i] < lo:
            lo = low[i]
            lo_hour = hour
    if count == 0 or hi <= lo:
        return 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0
    range_pips = (hi - lo) / pip
    atr_ratio = range_pips * pip / atr_value if atr_value > 0.0 else 0.0
    hi_ratio = float(hi_hour - TOKYO_START_JST) / span
    lo_ratio = float(lo_hour - TOKYO_START_JST) / span
    valid = 1 if range_pips >= min_range_pips else 0
    return hi, lo, range_pips, atr_ratio, count, hi_ratio, lo_ratio, valid


@njit(cache=True)
def detect_breakout_bar_core(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    jst_days: np.ndarray,
    jst_hours: np.ndarray,
    target_day: int,
    tokyo_high: float,
    tokyo_low: float,
    pip: float,
    min_breakout_pips: float,
) -> tuple[int, int, float, float, float, float]:
    """Returns (bar_index, direction, close_dist_pips, candle_pips, momentum, breakout_close). direction: 1 BUY / -1 SELL / 0 none."""
    min_dist = min_breakout_pips * pip
    for i in range(len(close)):
        if jst_days[i] != target_day:
            continue
        if not _in_london_breakout_window(int(jst_hours[i])):
            continue
        c = close[i]
        if c > tokyo_high + min_dist and c > tokyo_high:
            if high[i] <= tokyo_high:
                continue
            dist = (c - tokyo_high) / pip
            candle = (high[i] - low[i]) / pip
            return i, 1, dist, candle, dist, c
        if c < tokyo_low - min_dist and c < tokyo_low:
            if low[i] >= tokyo_low:
                continue
            dist = (tokyo_low - c) / pip
            candle = (high[i] - low[i]) / pip
            return i, -1, dist, candle, dist, c
    return -1, 0, 0.0, 0.0, 0.0, 0.0


@njit(cache=True)
def detect_retest_core(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    breakout_idx: int,
    direction: int,
    tokyo_high: float,
    tokyo_low: float,
    range_pips: float,
    pip: float,
    max_retest_bars: int,
    retest_buffer_pips: float,
) -> tuple[int, float, int, float, str, int, int, float, float, int]:
    """
    Returns:
      retest_idx, depth_pips, duration, depth_ratio, pa_type, absorbed(0/1),
      rejection_strength, tick_volume_ratio, trade_density, retest_count
    """
    buffer = retest_buffer_pips * pip
    boundary = tokyo_high if direction == 1 else tokyo_low
    end = min(len(close), breakout_idx + max_retest_bars + 1)
    retest_count = 0
    breakout_vol = volume[breakout_idx] if breakout_idx < len(volume) else 0.0
    avg_vol = 0.0
    vol_n = 0
    for j in range(max(0, breakout_idx - 20), breakout_idx):
        if volume[j] > 0.0:
            avg_vol += volume[j]
            vol_n += 1
    avg_vol = avg_vol / vol_n if vol_n > 0 else breakout_vol

    for i in range(breakout_idx + 1, end):
        duration = i - breakout_idx
        c = close[i]
        h = high[i]
        l = low[i]
        if direction == 1:
            if c < tokyo_high:
                return -1, 0.0, duration, 0.0, PA_CLOSE_ONLY, 1, 0.0, 0.0, 0.0, retest_count
            touched = l <= tokyo_high + buffer
            if touched:
                retest_count += 1
                depth = max(tokyo_high - l, 0.0) / pip
                depth_ratio = depth / range_pips if range_pips > 0.0 else 0.0
                rejection = max(c - tokyo_high, 0.0) / pip
                pa = _classify_pa(
                    direction,
                    open_[i],
                    h,
                    l,
                    c,
                    open_[i - 1],
                    high[i - 1],
                    low[i - 1],
                    close[i - 1],
                    tokyo_high,
                    pip,
                )
                if c > boundary and rejection > 0.0:
                    tick_ratio = volume[i] / avg_vol if avg_vol > 0.0 else 1.0
                    density = volume[i] / max(h - l, pip)
                    return i, depth, duration, depth_ratio, pa, 0, rejection, tick_ratio, density, retest_count
        else:
            if c > tokyo_low:
                return -1, 0.0, duration, 0.0, PA_CLOSE_ONLY, 1, 0.0, 0.0, 0.0, retest_count
            touched = h >= tokyo_low - buffer
            if touched:
                retest_count += 1
                depth = max(h - tokyo_low, 0.0) / pip
                depth_ratio = depth / range_pips if range_pips > 0.0 else 0.0
                rejection = max(tokyo_low - c, 0.0) / pip
                pa = _classify_pa(
                    direction,
                    open_[i],
                    h,
                    l,
                    c,
                    open_[i - 1],
                    high[i - 1],
                    low[i - 1],
                    close[i - 1],
                    tokyo_low,
                    pip,
                )
                if c < boundary and rejection > 0.0:
                    tick_ratio = volume[i] / avg_vol if avg_vol > 0.0 else 1.0
                    density = volume[i] / max(h - l, pip)
                    return i, depth, duration, depth_ratio, pa, 0, rejection, tick_ratio, density, retest_count
    return -1, 0.0, max_retest_bars, 0.0, PA_CLOSE_ONLY, 0, 0.0, 0.0, 0.0, retest_count


def detect_tokyo_range(
    h1_df: pd.DataFrame,
    current_ts: pd.Timestamp,
    *,
    tokyo_start_jst: int = TOKYO_START_JST,
    tokyo_end_jst: int = TOKYO_END_JST,
    min_range_pips: float = MIN_TOKYO_RANGE_PIPS,
    pip_size: float = DEFAULT_PIP_SIZE,
) -> TokyoRange | None:
    """Detect completed Tokyo session range on H1 up to ``current_ts`` (no lookahead)."""
    del tokyo_start_jst, tokyo_end_jst
    if h1_df is None or h1_df.empty:
        return None
    clipped = clip_as_of(h1_df, current_ts)
    if clipped.empty:
        return None

    hour, _minute = jst_hour_minute(current_ts)
    if hour < TOKYO_END_JST:
        return None

    high, low, _close, _open, _vol = _arrays_from_df(clipped)
    ts_ns = _datetime_ns(clipped)
    jst_days, jst_hours, _mins = jst_fields_from_ns(ts_ns)
    target_day = jst_day_ordinal(current_ts)
    atr_series = precompute_atr_series(high, low, _close, ATR_PERIOD)
    atr_val = float(atr_series[-1]) if len(atr_series) else 0.0

    hi, lo, range_pips, atr_ratio, count, hi_ratio, lo_ratio, valid_flag = scan_tokyo_range_core(
        high,
        low,
        jst_days,
        jst_hours,
        target_day,
        atr_val,
        pip_size,
        min_range_pips,
    )
    if count == 0:
        return None
    symmetry = 1.0 - abs(hi_ratio - lo_ratio)
    return TokyoRange(
        date=ordinal_to_date(target_day),
        high=float(hi),
        low=float(lo),
        range_pips=float(range_pips),
        range_atr_ratio=float(atr_ratio),
        formation_bars=int(count),
        tokyo_high_time_ratio=float(hi_ratio),
        tokyo_low_time_ratio=float(lo_ratio),
        is_valid=bool(valid_flag),
    )


def detect_london_breakout(
    m15_df: pd.DataFrame,
    tokyo_range: TokyoRange,
    current_ts: pd.Timestamp,
    *,
    min_breakout_pips: float = MIN_BREAKOUT_PIPS,
    pip_size: float = DEFAULT_PIP_SIZE,
) -> tuple[str, int] | None:
    """First valid London breakout on M15 up to ``current_ts``."""
    if m15_df is None or m15_df.empty or not tokyo_range.is_valid:
        return None
    clipped = clip_as_of(m15_df, current_ts)
    if clipped.empty:
        return None

    open_, high, low, close, _vol = ohlcv_to_arrays(clipped)
    ts_ns = _datetime_ns(clipped)
    jst_days, jst_hours, _mins = jst_fields_from_ns(ts_ns)
    target_day = jst_day_ordinal(pd.Timestamp(tokyo_range.date))

    idx, direction, _dist, _candle, _mom, _bc = detect_breakout_bar_core(
        close,
        high,
        low,
        jst_days,
        jst_hours,
        target_day,
        tokyo_range.high,
        tokyo_range.low,
        pip_size,
        min_breakout_pips,
    )
    if idx < 0 or direction == 0:
        return None
    return ("BUY" if direction == 1 else "SELL"), int(idx)


def detect_retest(
    m15_df: pd.DataFrame,
    tokyo_range: TokyoRange,
    breakout_bar_index: int,
    direction: str,
    *,
    max_retest_bars: int = MAX_RETEST_BARS,
    retest_buffer_pips: float = 2.0,
    pip_size: float = DEFAULT_PIP_SIZE,
) -> dict[str, Any] | None:
    """Detect retest after breakout on M15."""
    if m15_df is None or m15_df.empty or breakout_bar_index < 0:
        return None
    open_, high, low, close, volume = ohlcv_to_arrays(m15_df)
    dir_i = 1 if direction == "BUY" else -1
    (
        retest_idx,
        depth,
        duration,
        depth_ratio,
        pa_type,
        absorbed,
        rejection,
        tick_ratio,
        density,
        retest_count,
    ) = detect_retest_core(
        open_,
        high,
        low,
        close,
        volume,
        breakout_bar_index,
        dir_i,
        tokyo_range.high,
        tokyo_range.low,
        tokyo_range.range_pips,
        pip_size,
        max_retest_bars,
        retest_buffer_pips,
    )
    if retest_idx < 0:
        return None
    return {
        "retest_bar_index": int(retest_idx),
        "retest_depth_pips": float(depth),
        "retest_duration_bars": int(duration),
        "retest_pa_type": str(pa_type),
        "retest_rejection_strength": float(rejection),
        "is_absorbed": bool(absorbed),
        "retest_depth_ratio": float(depth_ratio),
        "tick_volume_ratio": float(tick_ratio),
        "breakout_trade_density": float(density),
        "retest_count": int(retest_count),
    }


def _datetime_ns(df: pd.DataFrame) -> np.ndarray:
    from strategies.bt_ohlcv import lookup_ohlcv

    found = lookup_ohlcv(df)
    if found is not None:
        return np.asarray(found.datetime_ns, dtype=np.int64)
    return np.asarray(pd.to_datetime(df["datetime"]).astype(np.int64))


def _arrays_from_df(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    o, h, l, c, v = ohlcv_to_arrays(df)
    return h, l, c, o, v


def past_tokyo_range_pips(
    h1_df: pd.DataFrame,
    target_day_ord: int,
    *,
    lookback_days: int = 20,
    pip_size: float = DEFAULT_PIP_SIZE,
) -> np.ndarray:
    """Past Tokyo range widths (pips) for percentile / compression."""
    high, low, _c, _o, _v = _arrays_from_df(h1_df)
    ts_ns = _datetime_ns(h1_df)
    jst_days, jst_hours, _m = jst_fields_from_ns(ts_ns)
    widths: list[float] = []
    for day in range(target_day_ord - lookback_days, target_day_ord):
        hi, lo, range_pips, _atr_r, count, _hr, _lr, _vflag = scan_tokyo_range_core(
            high,
            low,
            jst_days,
            jst_hours,
            day,
            1.0,
            pip_size,
            0.0,
        )
        del hi, lo
        if count > 0 and range_pips > 0.0:
            widths.append(range_pips)
    return np.asarray(widths, dtype=np.float64)


def london_bars_before_breakout(
    jst_hours: np.ndarray,
    jst_days: np.ndarray,
    target_day: int,
    breakout_idx: int,
) -> int:
    count = 0
    for i in range(breakout_idx + 1):
        if jst_days[i] != target_day:
            continue
        if _in_london_breakout_window(int(jst_hours[i])):
            count += 1
    return max(count - 1, 0)


def london_range_before_break_pips(
    high: np.ndarray,
    low: np.ndarray,
    jst_days: np.ndarray,
    jst_hours: np.ndarray,
    target_day: int,
    breakout_idx: int,
    pip: float,
) -> float:
    hi = -1.0e18
    lo = 1.0e18
    for i in range(breakout_idx + 1):
        if jst_days[i] != target_day:
            continue
        if not _in_london_breakout_window(int(jst_hours[i])):
            continue
        if high[i] > hi:
            hi = high[i]
        if low[i] < lo:
            lo = low[i]
    if hi <= lo:
        return 0.0
    return (hi - lo) / pip


__all__ = [
    "TokyoRange",
    "detect_tokyo_range",
    "detect_london_breakout",
    "detect_retest",
    "jst_fields_from_ns",
    "jst_day_ordinal",
    "jst_hour_minute",
    "past_tokyo_range_pips",
    "london_bars_before_breakout",
    "london_range_before_break_pips",
    "TOKYO_START_JST",
    "TOKYO_END_JST",
    "LONDON_START_JST",
    "BREAKOUT_WINDOW_END_JST",
    "MAX_RETEST_BARS",
    "MIN_TOKYO_RANGE_PIPS",
    "MIN_BREAKOUT_PIPS",
]
