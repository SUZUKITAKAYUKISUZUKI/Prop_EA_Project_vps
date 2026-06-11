"""
strategies/cspa_scan_numba.py — Phase 2: @njit kernels for CSPA scan hot path.

Requires ``numba`` (optional). Falls back to pure numpy when unavailable or disabled.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from strategies.archive.cspa import ImpulseLeg, MomentumSignal, StagnationCluster

_NUMBA_OK = False
try:
    from numba import njit

    _NUMBA_OK = True
except ImportError:

    def njit(*args, **kwargs):  # type: ignore[misc]
        def decorator(fn):
            return fn

        if args and callable(args[0]):
            return args[0]
        return decorator


_STAGNATION_MAX_BODY_ATR = float(os.getenv("CSPA_STAGNATION_MAX_BODY_ATR", "0.35"))
_STAGNATION_MIN_BARS = int(os.getenv("CSPA_STAGNATION_MIN_BARS", "1"))

MOM_NONE = 0
MOM_BODY = 1
MOM_ENGULF = 2
MOM_PIN = 3
MOM_SWEEP = 4

_MOMENTUM_TYPE_NAMES = ("NONE", "BODY_BREAK", "ENGULFING", "PIN_BAR", "SWEEP_ENGULFING")


def numba_available() -> bool:
    return _NUMBA_OK


def swing_prices_array(swings) -> np.ndarray:
    if not swings:
        return np.empty(0, dtype=np.float64)
    return np.array([s.price for s in swings], dtype=np.float64)


@njit(cache=True)
def _body_size_numba(open_: float, close: float) -> float:
    return abs(close - open_)


@njit(cache=True)
def _bisect_right_int(arr: np.ndarray, x: int) -> int:
    lo = 0
    hi = len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


PHASE_NONE = 0
PHASE_UPTREND = 1
PHASE_DOWNTREND = 2
PHASE_RANGE = 3

_PHASE_NAMES = ("NONE", "UPTREND", "DOWNTREND", "RANGE")


@njit(cache=True)
def classify_bias_dow_phase_numba(
    bar_index: int,
    high_indices: np.ndarray,
    high_prices: np.ndarray,
    low_indices: np.ndarray,
    low_prices: np.ndarray,
) -> int:
    if bar_index < 0:
        return PHASE_NONE
    hi = _bisect_right_int(high_indices, bar_index)
    lo = _bisect_right_int(low_indices, bar_index)
    if hi < 2 or lo < 2:
        return PHASE_NONE
    h1 = high_prices[hi - 2]
    h2 = high_prices[hi - 1]
    l1 = low_prices[lo - 2]
    l2 = low_prices[lo - 1]
    if h2 > h1 and l2 > l1:
        return PHASE_UPTREND
    if h2 < h1 and l2 < l1:
        return PHASE_DOWNTREND
    return PHASE_RANGE


@njit(cache=True)
def find_latest_impulse_numba(
    bar_index: int,
    phase_code: int,
    high_indices: np.ndarray,
    high_prices: np.ndarray,
    low_indices: np.ndarray,
    low_prices: np.ndarray,
    min_warmup: int,
):
    """Returns (found, is_up, start_idx, end_idx, start_price, end_price, size)."""
    empty = (False, True, -1, -1, 0.0, 0.0, 0.0)
    if phase_code not in (PHASE_UPTREND, PHASE_DOWNTREND) or bar_index < min_warmup:
        return empty
    up_to = bar_index - 1
    hi = _bisect_right_int(high_indices, up_to)
    lo = _bisect_right_int(low_indices, up_to)
    if phase_code == PHASE_UPTREND:
        if hi < 1 or lo < 1:
            return empty
        peak_idx = high_indices[hi - 1]
        peak_price = high_prices[hi - 1]
        trough_idx = -1
        trough_price = 0.0
        for j in range(lo):
            if low_indices[j] < peak_idx:
                trough_idx = low_indices[j]
                trough_price = low_prices[j]
        if trough_idx < 0:
            return empty
        size = peak_price - trough_price
        if size <= 0.0:
            return empty
        return (True, True, trough_idx, peak_idx, trough_price, peak_price, size)
    if lo < 1 or hi < 1:
        return empty
    trough_idx = low_indices[lo - 1]
    trough_price = low_prices[lo - 1]
    peak_idx = -1
    peak_price = 0.0
    for j in range(hi):
        if high_indices[j] < trough_idx:
            peak_idx = high_indices[j]
            peak_price = high_prices[j]
    if peak_idx < 0:
        return empty
    size = peak_price - trough_price
    if size <= 0.0:
        return empty
    return (True, False, peak_idx, trough_idx, peak_price, trough_price, size)


def phase_from_numba(code: int) -> str:
    return _PHASE_NAMES[int(code)]


def impulse_from_numba_tuple(result) -> ImpulseLeg | None:
    from strategies.archive.cspa import ImpulseLeg

    found, is_up, start_idx, end_idx, start_price, end_price, size = result
    if not found:
        return None
    return ImpulseLeg(
        direction="UP" if is_up else "DOWN",
        start_index=int(start_idx),
        end_index=int(end_idx),
        start_price=float(start_price),
        end_price=float(end_price),
        impulse_size=float(size),
    )


@njit(cache=True)
def measure_retrace_ratio_numba(
    high: np.ndarray,
    low: np.ndarray,
    is_up: bool,
    impulse_end_index: int,
    impulse_end_price: float,
    impulse_size: float,
    bar_index: int,
) -> float:
    if impulse_size <= 0.0:
        return 0.0
    start = impulse_end_index
    end = bar_index + 1
    n = len(high)
    if start >= end or start >= n:
        return 0.0
    if end > n:
        end = n
    if is_up:
        correction_low = low[start]
        for i in range(start + 1, end):
            if low[i] < correction_low:
                correction_low = low[i]
        return (impulse_end_price - correction_low) / impulse_size
    correction_high = high[start]
    for i in range(start + 1, end):
        if high[i] > correction_high:
            correction_high = high[i]
    return (correction_high - impulse_end_price) / impulse_size


@njit(cache=True)
def m1_over_retraces_structure_numba(
    trigger_dt_ns: np.ndarray,
    trigger_high: np.ndarray,
    trigger_low: np.ndarray,
    structure_dt_ns: np.ndarray,
    impulse_end_index: int,
    trigger_index: int,
    is_up: bool,
    impulse_end_price: float,
    impulse_size: float,
    fib_retrace_max: float,
) -> bool:
    if impulse_size <= 0.0 or trigger_index < 0 or trigger_index >= len(trigger_dt_ns):
        return False
    if impulse_end_index < 0 or impulse_end_index >= len(structure_dt_ns):
        return False
    start_ts = structure_dt_ns[impulse_end_index]
    start_i = int(np.searchsorted(trigger_dt_ns, start_ts, side="left"))
    end_i = trigger_index + 1
    n_trig = len(trigger_dt_ns)
    if start_i >= end_i:
        return False
    if end_i > n_trig:
        end_i = n_trig
    if is_up:
        deepest = trigger_low[start_i]
        for i in range(start_i + 1, end_i):
            if trigger_low[i] < deepest:
                deepest = trigger_low[i]
        ratio = (impulse_end_price - deepest) / impulse_size
    else:
        highest = trigger_high[start_i]
        for i in range(start_i + 1, end_i):
            if trigger_high[i] > highest:
                highest = trigger_high[i]
        ratio = (highest - impulse_end_price) / impulse_size
    return ratio > fib_retrace_max


@njit(cache=True)
def prior_correction_ratio_numba(
    high: np.ndarray,
    low: np.ndarray,
    is_uptrend: bool,
    impulse_start_index: int,
    high_indices: np.ndarray,
    high_prices: np.ndarray,
    low_indices: np.ndarray,
    low_prices: np.ndarray,
) -> float:
    """Returns ``np.nan`` when prior correction ratio is undefined."""
    hi = _bisect_right_int(high_indices, impulse_start_index)
    lo = _bisect_right_int(low_indices, impulse_start_index)
    if is_uptrend:
        if hi < 2 or lo < 2:
            return np.nan
        prev_peak = high_prices[hi - 2]
        prev_trough = low_prices[lo - 2]
        prev_peak_idx = high_indices[hi - 2]
        prev_trough_idx = low_indices[lo - 2]
        size = prev_peak - prev_trough
        if size <= 0.0:
            return np.nan
        start = prev_trough_idx
        end = prev_peak_idx + 1
        if start >= end:
            return np.nan
        if end > len(low):
            end = len(low)
        correction_low = low[start]
        for i in range(start + 1, end):
            if low[i] < correction_low:
                correction_low = low[i]
        return (prev_peak - correction_low) / size
    if lo < 2 or hi < 2:
        return np.nan
    prev_trough = low_prices[lo - 2]
    prev_peak = high_prices[hi - 2]
    prev_peak_idx = high_indices[hi - 2]
    prev_trough_idx = low_indices[lo - 2]
    size = prev_peak - prev_trough
    if size <= 0.0:
        return np.nan
    start = prev_peak_idx
    end = prev_trough_idx + 1
    if start >= end:
        return np.nan
    if end > len(high):
        end = len(high)
    correction_high = high[start]
    for i in range(start + 1, end):
        if high[i] > correction_high:
            correction_high = high[i]
    return (correction_high - prev_trough) / size


@njit(cache=True)
def correction_rhythm_ok_numba(
    high: np.ndarray,
    low: np.ndarray,
    is_uptrend: bool,
    impulse_start_index: int,
    high_indices: np.ndarray,
    high_prices: np.ndarray,
    low_indices: np.ndarray,
    low_prices: np.ndarray,
    current_ratio: float,
    max_ratio: float,
) -> bool:
    prev = prior_correction_ratio_numba(
        high,
        low,
        is_uptrend,
        impulse_start_index,
        high_indices,
        high_prices,
        low_indices,
        low_prices,
    )
    if np.isnan(prev) or prev <= 0.0:
        return True
    return current_ratio <= prev * max_ratio


@njit(cache=True)
def volatility_percentile_numba(
    atr: np.ndarray,
    bar_index: int,
    lookback: int,
    atr_period: int,
) -> float:
    if bar_index < atr_period or bar_index >= len(atr):
        return 0.5
    start = atr_period
    if bar_index - lookback + 1 > start:
        start = bar_index - lookback + 1
    current = atr[bar_index]
    if current <= 0.0:
        return 0.5
    below = 0
    count = 0
    for i in range(start, bar_index + 1):
        count += 1
        if atr[i] <= current:
            below += 1
    if count == 0:
        return 0.5
    return round(below / count, 4)


@njit(cache=True)
def observe_overlap_ratio_numba(
    high: np.ndarray,
    low: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> float:
    if end_idx <= start_idx:
        return 0.0
    start = start_idx + 1
    end = end_idx + 1
    if start >= end:
        return 0.0
    total = 0.0
    count = end - start
    for i in range(start, end):
        prev_i = i - 1
        overlap = min(high[prev_i], high[i]) - max(low[prev_i], low[i])
        if overlap > 0.0:
            total += overlap
    return round(total / count, 6)


@njit(cache=True)
def observe_correction_smoothness_numba(
    high: np.ndarray,
    low: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> float:
    if end_idx <= start_idx:
        return 0.5
    n = end_idx - start_idx + 1
    if n < 2:
        return 0.5
    mean_r = 0.0
    for i in range(start_idx, end_idx + 1):
        mean_r += high[i] - low[i]
    mean_r /= n
    if mean_r <= 0.0:
        return 0.5
    var = 0.0
    for i in range(start_idx, end_idx + 1):
        r = high[i] - low[i]
        d = r - mean_r
        var += d * d
    var /= n
    cv = (var ** 0.5) / mean_r
    capped = cv
    if capped > 1.0:
        capped = 1.0
    score = 1.0 - capped
    if score < 0.0:
        score = 0.0
    return round(score, 4)


@njit(cache=True)
def observe_wick_balance_numba(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> float:
    total_ratio = 0.0
    count = 0
    for i in range(start_idx, end_idx + 1):
        o = open_[i]
        c = close[i]
        h = high[i]
        l = low[i]
        bar_range = h - l
        if bar_range <= 0.0:
            continue
        upper = h - max(o, c)
        lower = min(o, c) - l
        total_ratio += (upper + lower) / bar_range
        count += 1
    if count == 0:
        return 0.0
    return round(total_ratio / count, 4)


@njit(cache=True)
def detect_stagnation_cluster_numba(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    end_index: int,
    is_buy: bool,
    max_bars: int,
):
    """Returns tuple (start, end, count, avg_body_atr, zone_high, zone_low, solid) or (-1,...)."""
    if end_index < 1:
        return (-1, -1, 0, 0.0, 0.0, 0.0, False)

    idx_buf = np.empty(max_bars + 2, dtype=np.int64)
    count = 0
    lo = end_index - 1
    hi = end_index - max_bars - 1
    if hi < -1:
        hi = -1

    for i in range(lo, hi, -1):
        if i < 0:
            break
        if i >= len(atr):
            break
        atr_v = atr[i]
        if atr_v <= 0.0:
            break
        o = open_[i]
        c = close[i]
        h = high[i]
        l = low[i]
        body = _body_size_numba(o, c)
        body_atr = body / atr_v
        is_small = body_atr <= _STAGNATION_MAX_BODY_ATR
        rejection = False
        if is_buy:
            lower_wick = min(o, c) - l
            if lower_wick >= body * 1.5 and body_atr <= _STAGNATION_MAX_BODY_ATR * 1.2:
                rejection = True
        else:
            upper_wick = h - max(o, c)
            if upper_wick >= body * 1.5 and body_atr <= _STAGNATION_MAX_BODY_ATR * 1.2:
                rejection = True
        if is_small or rejection:
            idx_buf[count] = i
            count += 1
        else:
            break

    if count < _STAGNATION_MIN_BARS:
        return (-1, -1, 0, 0.0, 0.0, 0.0, False)

    for j in range(count // 2):
        a = idx_buf[j]
        b = idx_buf[count - 1 - j]
        idx_buf[j] = b
        idx_buf[count - 1 - j] = a

    zone_high = -1.0e100
    zone_low = 1.0e100
    body_sum = 0.0
    for j in range(count):
        i = idx_buf[j]
        atr_v = atr[i]
        o = open_[i]
        c = close[i]
        body_sum += _body_size_numba(o, c) / atr_v if atr_v > 0.0 else 0.0
        if high[i] > zone_high:
            zone_high = high[i]
        if low[i] < zone_low:
            zone_low = low[i]

    avg_body = body_sum / count
    first_body_atr = _body_size_numba(open_[idx_buf[0]], close[idx_buf[0]]) / atr[idx_buf[0]]
    solid = count >= 2 or (count == 1 and first_body_atr <= _STAGNATION_MAX_BODY_ATR * 0.8)
    return (idx_buf[0], idx_buf[count - 1], count, avg_body, zone_high, zone_low, solid)


@njit(cache=True)
def detect_momentum_breakout_numba(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_arr: np.ndarray,
    bar_index: int,
    is_buy: bool,
    zone_high: float,
    zone_low: float,
    min_body_atr: float,
):
    """Returns (detected, type_code, entry, high, low, body_atr, atr)."""
    if bar_index < 1 or bar_index >= len(close):
        return (False, MOM_NONE, 0.0, 0.0, 0.0, 0.0, 0.0)
    o = open_[bar_index]
    c = close[bar_index]
    h = high[bar_index]
    l = low[bar_index]
    prev_o = open_[bar_index - 1]
    prev_c = close[bar_index - 1]
    atr_v = atr_arr[bar_index]
    if atr_v <= 0.0:
        return (False, MOM_NONE, 0.0, 0.0, 0.0, 0.0, 0.0)
    body = _body_size_numba(o, c)
    if body < min_body_atr * atr_v:
        return (False, MOM_NONE, 0.0, 0.0, 0.0, 0.0, 0.0)
    body_atr = body / atr_v
    trigger_type = MOM_BODY
    if is_buy:
        if c <= o or c <= zone_high:
            return (False, MOM_NONE, 0.0, 0.0, 0.0, 0.0, 0.0)
        if prev_c < prev_o and c > o and c >= prev_o:
            trigger_type = MOM_ENGULF
        lower_wick = min(o, c) - l
        if lower_wick >= 2.0 * body:
            trigger_type = MOM_PIN
    else:
        if c >= o or c >= zone_low:
            return (False, MOM_NONE, 0.0, 0.0, 0.0, 0.0, 0.0)
        if prev_c > prev_o and c < o and c <= prev_o:
            trigger_type = MOM_ENGULF
        upper_wick = h - max(o, c)
        if upper_wick >= 2.0 * body:
            trigger_type = MOM_PIN
    return (True, trigger_type, c, h, l, round(body_atr, 4), atr_v)


@njit(cache=True)
def detect_sweep_engulfing_numba(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_arr: np.ndarray,
    bar_index: int,
    is_buy: bool,
    zone_high: float,
    zone_low: float,
    min_body_atr: float,
    min_range_atr: float,
    min_outside_ratio: float,
):
    """Returns (detected, type_code, entry, high, low, body_atr, atr)."""
    if bar_index < 1 or bar_index >= len(close):
        return (False, MOM_NONE, 0.0, 0.0, 0.0, 0.0, 0.0)
    o = open_[bar_index]
    c = close[bar_index]
    h = high[bar_index]
    l = low[bar_index]
    prev_h = high[bar_index - 1]
    prev_l = low[bar_index - 1]
    atr_v = atr_arr[bar_index]
    body = _body_size_numba(o, c)
    if atr_v <= 0.0 or body < min_body_atr * atr_v:
        return (False, MOM_NONE, 0.0, 0.0, 0.0, 0.0, 0.0)
    bar_range = h - l
    prev_range = prev_h - prev_l
    if bar_range < min_range_atr * atr_v:
        return (False, MOM_NONE, 0.0, 0.0, 0.0, 0.0, 0.0)
    if prev_range > 0.0 and bar_range < prev_range * min_outside_ratio:
        return (False, MOM_NONE, 0.0, 0.0, 0.0, 0.0, 0.0)
    if is_buy:
        if l >= prev_l or c <= prev_h or c <= o or h <= prev_h:
            return (False, MOM_NONE, 0.0, 0.0, 0.0, 0.0, 0.0)
        swept_zone_low = l < zone_low
        if not swept_zone_low and c <= zone_high:
            return (False, MOM_NONE, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        if h <= prev_h or c >= prev_l or c >= o or l >= prev_l:
            return (False, MOM_NONE, 0.0, 0.0, 0.0, 0.0, 0.0)
        swept_zone_high = h > zone_high
        if not swept_zone_high and c >= zone_low:
            return (False, MOM_NONE, 0.0, 0.0, 0.0, 0.0, 0.0)
    return (True, MOM_SWEEP, c, h, l, round(body / atr_v, 4), atr_v)


def stagnation_from_numba_tuple(result) -> StagnationCluster | None:
    from strategies.archive.cspa import StagnationCluster

    start, end, count, avg_body, zone_high, zone_low, solid = result
    if int(start) < 0:
        return None
    return StagnationCluster(
        start_index=int(start),
        end_index=int(end),
        bar_count=int(count),
        avg_body_atr=round(float(avg_body), 4),
        zone_high=float(zone_high),
        zone_low=float(zone_low),
        solid_ground=bool(solid),
    )


def prior_ratio_from_numba(value: float) -> float | None:
    if np.isnan(value):
        return None
    return float(value)


def momentum_from_numba_tuple(
    trigger,
    bar_index: int,
    result,
) -> MomentumSignal:
    from strategies.archive.cspa import MomentumSignal
    from strategies.archive.cspa_scan_hot import _empty_momentum, _ts_at

    detected, type_code, entry, high, low, body_atr, atr_v = result
    if not detected:
        return _empty_momentum(bar_index)
    return MomentumSignal(
        detected=True,
        trigger_type=_MOMENTUM_TYPE_NAMES[int(type_code)],  # type: ignore[arg-type]
        bar_index=bar_index,
        timestamp=_ts_at(trigger, bar_index),
        entry_price=float(entry),
        trigger_high=float(high),
        trigger_low=float(low),
        body_atr=float(body_atr),
        atr=float(atr_v),
    )
