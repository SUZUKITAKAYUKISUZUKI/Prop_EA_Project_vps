"""
strategies/archive/wyckoff_scan_hot.py — NumPy-only Wyckoff Reversal (WR) precompute scan.

ARCHIVED: WR 本体と同様 — LGR 構築に向けた発展的廃止に伴い移動。
"""

from __future__ import annotations

from datetime import date
from typing import Any, Callable

import numpy as np

from strategies.bt_l5 import compute_trade_excursions_np
from strategies.bt_ohlcv import OhlcvArrays, asof_end_index, slice_arrays, ts_ns_to_pd
from strategies.cspa_arrays import atr_at_index, compute_atr_np
from strategies.cspa_scan_hot import (
    HtfDirectionCache,
    minutes_from_session_open_ns,
    resolve_cspa_session_type_ns,
    volatility_percentile_np,
)
from strategies.market_utils import pip_size_for_pair
from strategies.archive.wyckoff_reversal import (
    ADR_LOOKBACK_DAYS,
    ALLOWED_PAIRS,
    AR_MIN_RALLY_ATR,
    ATR_PERIOD,
    AccumulationRange,
    LOOKBACK_BARS,
    MAX_HOLDING_BARS,
    MAX_SETUPS_PER_DAY,
    MIN_BARS_AFTER_SC,
    MIN_RR,
    MONTHLY_BARS,
    RECOVERY_QUALITY_BARS,
    ReversalSetup,
    SC_BODY_ATR_MULT,
    SC_DOWNTREND_LOOKBACK,
    SC_VOLUME_ZSCORE_LOOKBACK,
    SC_VOLUME_ZSCORE_MIN,
    SESSION_VOLUME_LOOKBACK_DAYS,
    SL_BUFFER_ATR,
    SPRING_MAX_DEPTH_ATR,
    SPRING_MIN_DEPTH_ATR,
    SPRING_VOLUME_RATIO_MIN,
    SUPPORT_BAND_ATR,
    UPTHRUST_MAX_HEIGHT_ATR,
    UPTHRUST_MIN_HEIGHT_ATR,
    UPTHRUST_VOLUME_RATIO_MIN,
    UPTHRUST_WICK_RATIO_MIN,
    VOLATILITY_LOOKBACK,
    VOLUME_LOOKBACK,
    WEEKLY_BARS,
    WYCKOFF_BT_SPREAD_PIPS,
    WYCKOFF_EXEC_BAR_MINUTES,
    WyckoffFeatures,
    _neutral_kalman_features,
    _position_in_range,
    classify_wyckoff_macro_phase,
    wr_mode_from_direction,
)

__all__ = ["detect_wyckoff_reversal_setups_np"]


def _ts_ns(ts: Any) -> int:
    if hasattr(ts, "value"):
        return int(ts.value)
    return int(ts)


def _bar_hour_ns(ts_ns: int) -> int:
    return int(ts_ns_to_pd(ts_ns).hour)


def _day_norm_ns(ts_ns: int) -> np.datetime64:
    return np.datetime64(ts_ns, "ns").astype("datetime64[D]")


def _session_volume_samples_np(
    arr: OhlcvArrays,
    bar_index: int,
    lookback_days: int = SESSION_VOLUME_LOOKBACK_DAYS,
) -> list[float]:
    if bar_index < 0 or bar_index >= arr.length:
        return []
    ts_ns = int(arr.datetime_ns[bar_index])
    target_hour = _bar_hour_ns(ts_ns)
    current_day = _day_norm_ns(ts_ns)
    day_norm = arr.datetime_ns.astype("datetime64[D]")
    samples: list[float] = []
    for offset in range(1, lookback_days + 1):
        prior_day = current_day - np.timedelta64(offset, "D")
        mask = (day_norm == prior_day) & (np.arange(arr.length) < bar_index)
        if not np.any(mask):
            continue
        hours = np.array([_bar_hour_ns(int(arr.datetime_ns[i])) for i in np.where(mask)[0]])
        hour_mask = hours == target_hour
        if not np.any(hour_mask):
            continue
        prior_idx = np.where(mask)[0][hour_mask]
        if len(prior_idx) == 0:
            continue
        samples.append(float(arr.volume[int(prior_idx[-1])]))
    return samples


def _rolling_volume_zscore_np(
    arr: OhlcvArrays,
    bar_index: int,
    lookback: int = SC_VOLUME_ZSCORE_LOOKBACK,
) -> float:
    start = max(0, bar_index - lookback)
    window = arr.volume[start:bar_index]
    if len(window) < 3:
        return 0.0
    val = float(arr.volume[bar_index])
    mean = float(np.mean(window))
    std = float(np.std(window, ddof=0))
    if std <= 0:
        return 0.0
    return (val - mean) / std


def _session_volume_zscore_np(
    arr: OhlcvArrays,
    bar_index: int,
    lookback_days: int = SESSION_VOLUME_LOOKBACK_DAYS,
) -> float:
    samples = _session_volume_samples_np(arr, bar_index, lookback_days)
    val = float(arr.volume[bar_index])
    if len(samples) >= 2:
        mean = float(np.mean(samples))
        std = float(np.std(samples, ddof=0))
        if std > 0:
            return (val - mean) / std
        return 2.0 if val > mean else 0.0
    return _rolling_volume_zscore_np(arr, bar_index)


def _in_downtrend_context_np(
    arr: OhlcvArrays,
    sc_idx: int,
    h1_atr: np.ndarray,
    lookback: int = SC_DOWNTREND_LOOKBACK,
) -> bool:
    start = max(0, sc_idx - lookback)
    if sc_idx - start < 10:
        return False
    closes = arr.close[start:sc_idx]
    sc_low = float(arr.low[sc_idx])
    window_low = float(np.min(arr.low[start:sc_idx]))
    atr = atr_at_index(h1_atr, sc_idx)
    near_low = sc_low <= window_low + max(0.15 * atr, 1e-9)
    x = np.arange(len(closes), dtype=np.float64)
    slope = float(np.polyfit(x, closes, 1)[0])
    return slope < 0 and near_low


def _support_band(sc_price: float, atr: float) -> tuple[float, float]:
    half = SUPPORT_BAND_ATR * atr
    return sc_price - half, sc_price + half


def _touches_support_band(low: float, band_low: float, band_high: float) -> bool:
    return low <= band_high and low >= band_low - (band_high - band_low)


def _find_ps_price_np(arr: OhlcvArrays, sc_idx: int) -> float:
    start = max(0, sc_idx - 40)
    sc_low = float(arr.low[sc_idx])
    if start >= sc_idx:
        return sc_low * 1.001
    ps_ref = float(np.min(arr.low[start:sc_idx]))
    return max(ps_ref, sc_low * 1.001)


def _detect_phase_b_ut_np(
    arr: OhlcvArrays,
    ar_idx: int,
    end_idx: int,
    resistance_level: float,
) -> bool:
    for i in range(ar_idx + 1, end_idx + 1):
        if float(arr.high[i]) > resistance_level and float(arr.close[i]) < resistance_level:
            return True
    return False


def _count_support_tests_np(
    arr: OhlcvArrays,
    ar_idx: int,
    end_idx: int,
    band_low: float,
    band_high: float,
) -> tuple[int, float]:
    count = 0
    st_price = float(arr.high[ar_idx])
    for i in range(ar_idx + 1, end_idx + 1):
        low = float(arr.low[i])
        if _touches_support_band(low, band_low, band_high):
            count += 1
            st_price = min(st_price, low)
    return count, st_price


def _spread_percentile(spread_pips: float) -> float:
    ref = 3.0
    return round(min(max(spread_pips / ref, 0.0), 1.0), 4)


def _range_age_bars_np(accumulation: AccumulationRange, recovery_ts_ns: int) -> int:
    delta_min = (recovery_ts_ns - _ts_ns(accumulation.ar_timestamp)) / 60_000_000_000.0
    return max(0, int(delta_min // max(WYCKOFF_EXEC_BAR_MINUTES, 1)))


def _spring_wick_ratio_np(open_: float, high: float, low: float, close: float) -> float:
    span = high - low
    if span <= 0:
        return 0.0
    lower_wick = min(open_, close) - low
    return max(0.0, min(1.0, lower_wick / span))


def _upthrust_wick_ratio_np(open_: float, high: float, low: float, close: float) -> float:
    span = high - low
    if span <= 0:
        return 0.0
    upper_wick = high - max(open_, close)
    return max(0.0, min(1.0, upper_wick / span))


def _try_build_accumulation_np(
    arr: OhlcvArrays,
    sc_idx: int,
    h1_atr: np.ndarray,
    as_of_idx: int,
) -> AccumulationRange | None:
    if sc_idx >= as_of_idx - MIN_BARS_AFTER_SC or sc_idx < ATR_PERIOD:
        return None

    sc_atr = atr_at_index(h1_atr, sc_idx)
    if sc_atr <= 0:
        return None

    body = abs(float(arr.close[sc_idx]) - float(arr.open[sc_idx]))
    if float(arr.close[sc_idx]) >= float(arr.open[sc_idx]) or body < SC_BODY_ATR_MULT * sc_atr:
        return None

    vol_z = _session_volume_zscore_np(arr, sc_idx)
    if vol_z < SC_VOLUME_ZSCORE_MIN:
        return None
    if not _in_downtrend_context_np(arr, sc_idx, h1_atr):
        return None

    sc_price = float(arr.low[sc_idx])
    sc_ts = ts_ns_to_pd(int(arr.datetime_ns[sc_idx]))

    ar_price = sc_price
    ar_idx = sc_idx
    for j in range(sc_idx + 1, as_of_idx + 1):
        high = float(arr.high[j])
        ar_price = max(ar_price, high)
        rally = ar_price - sc_price
        if rally >= AR_MIN_RALLY_ATR * sc_atr:
            ar_idx = j
            break
    else:
        return None

    ar_ts = ts_ns_to_pd(int(arr.datetime_ns[ar_idx]))
    band_low, band_high = _support_band(sc_price, sc_atr)
    support_level = sc_price
    resistance_level = ar_price
    test_count, st_price = _count_support_tests_np(arr, ar_idx, as_of_idx, band_low, band_high)
    if test_count < 2:
        return None

    ps_price = _find_ps_price_np(arr, sc_idx)
    range_width = max(resistance_level - support_level, 0.0)
    range_width_atr = range_width / sc_atr if sc_atr > 0 else 0.0
    phase_b_ut = _detect_phase_b_ut_np(arr, ar_idx, as_of_idx, resistance_level)
    as_of_ts = ts_ns_to_pd(int(arr.datetime_ns[as_of_idx]))
    days_in_acc = max(0, (as_of_ts.normalize() - sc_ts.normalize()).days)
    phase_b_duration = max(0, as_of_idx - ar_idx)

    return AccumulationRange(
        ps_price=round(ps_price, 6),
        sc_price=round(sc_price, 6),
        ar_price=round(ar_price, 6),
        st_price=round(st_price, 6),
        support_level=round(support_level, 6),
        resistance_level=round(resistance_level, 6),
        range_width_atr=round(range_width_atr, 4),
        test_count=test_count,
        phase_b_ut_occurred=phase_b_ut,
        days_in_accumulation=days_in_acc,
        phase_b_duration=phase_b_duration,
        is_valid=True,
        sc_bar_index=sc_idx,
        ar_bar_index=ar_idx,
        sc_timestamp=sc_ts,
        ar_timestamp=ar_ts,
    )


def detect_accumulation_range_np(
    h1: OhlcvArrays,
    as_of_idx: int,
    h1_atr: np.ndarray,
    lookback: int = LOOKBACK_BARS,
) -> AccumulationRange | None:
    if as_of_idx < 0 or h1.length < ATR_PERIOD + MIN_BARS_AFTER_SC + 5:
        return None
    start_idx = max(ATR_PERIOD, as_of_idx - lookback)

    from strategies.archive.wyckoff_scan_numba import wr_scan_numba_active

    if wr_scan_numba_active():
        from strategies.archive.wyckoff_scan_numba import (
            accumulation_from_packed,
            find_accumulation_for_asof_numba,
        )

        packed = find_accumulation_for_asof_numba(
            h1.open,
            h1.high,
            h1.low,
            h1.close,
            h1.volume,
            h1_atr,
            as_of_idx,
            start_idx,
            ATR_PERIOD,
            MIN_BARS_AFTER_SC,
            SC_BODY_ATR_MULT,
            SC_VOLUME_ZSCORE_MIN,
            SC_DOWNTREND_LOOKBACK,
            AR_MIN_RALLY_ATR,
            SUPPORT_BAND_ATR,
            SC_VOLUME_ZSCORE_LOOKBACK,
        )
        return accumulation_from_packed(packed, h1, as_of_idx)

    for sc_idx in range(as_of_idx - MIN_BARS_AFTER_SC, start_idx - 1, -1):
        acc = _try_build_accumulation_np(h1, sc_idx, h1_atr, as_of_idx)
        if acc is not None and acc.is_valid:
            return acc
    return None


def _volatility_percentile_wyckoff_np(
    atr: np.ndarray,
    bar_index: int,
    *,
    lookback: int = VOLATILITY_LOOKBACK,
) -> float:
    pct = volatility_percentile_np(atr, bar_index, lookback=lookback, atr_period=ATR_PERIOD)
    if pct <= 1.0:
        return round(pct * 100.0, 4)
    return pct


def _atr_regime(vol_pct: float) -> str:
    if vol_pct < 33.0:
        return "LOW"
    if vol_pct < 66.0:
        return "NORMAL"
    return "HIGH"


def _compute_adr_remaining_np(arr: OhlcvArrays, bar_index: int, atr: float) -> float:
    if bar_index < 1 or atr <= 0:
        return 1.0
    day_norm = arr.datetime_ns.astype("datetime64[D]")
    current_day = day_norm[bar_index]
    day_mask = day_norm == current_day
    day_high = float(np.max(arr.high[day_mask]))
    day_low = float(np.min(arr.low[day_mask]))
    adr_used = day_high - day_low

    unique_days = np.unique(day_norm)
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


def _distance_to_rolling_low_np(
    arr: OhlcvArrays,
    bar_index: int,
    lookback: int,
    price: float,
    atr: float,
) -> float:
    if atr <= 0 or bar_index < 0:
        return 0.0
    start = max(0, bar_index - lookback + 1)
    rolling_low = float(np.min(arr.low[start : bar_index + 1]))
    return (price - rolling_low) / atr


def _session_normalized_volume_ratio_np(
    arr: OhlcvArrays,
    bar_index: int,
    lookback_days: int = SESSION_VOLUME_LOOKBACK_DAYS,
) -> float:
    val = float(arr.volume[bar_index])
    samples = _session_volume_samples_np(arr, bar_index, lookback_days)
    if samples:
        baseline = float(np.mean(samples))
    else:
        start = max(0, bar_index - VOLUME_LOOKBACK + 1)
        window = arr.volume[start:bar_index]
        baseline = float(np.mean(window)) if len(window) > 0 else val
    if baseline <= 0:
        return 1.0
    return val / baseline


def _volume_percentile_np(
    arr: OhlcvArrays,
    bar_index: int,
    lookback: int = VOLUME_LOOKBACK,
) -> float:
    start = max(0, bar_index - lookback + 1)
    window = arr.volume[start : bar_index + 1]
    if len(window) == 0:
        return 50.0
    current = float(arr.volume[bar_index])
    rank = int(np.sum(window <= current))
    return round(100.0 * rank / len(window), 4)


def _range_compression_np(
    h1: OhlcvArrays,
    accumulation: AccumulationRange,
    spring_ts_ns: int,
    atr: float,
) -> float:
    if atr <= 0 or accumulation.range_width_atr <= 0:
        return 1.0
    ar_ns = _ts_ns(accumulation.ar_timestamp)
    mask = (h1.datetime_ns >= ar_ns) & (h1.datetime_ns <= spring_ts_ns)
    if not np.any(mask):
        return 1.0
    phase_b_width_atr = float(np.max(h1.high[mask]) - np.min(h1.low[mask])) / atr
    return phase_b_width_atr / accumulation.range_width_atr


def _recovery_close_ratio_np(
    arr: OhlcvArrays,
    trigger_idx: int,
    recovery_idx: int,
    *,
    direction: str = "BUY",
) -> float:
    if recovery_idx < trigger_idx:
        return 0.0
    o = arr.open[trigger_idx : recovery_idx + 1].astype(np.float64)
    c = arr.close[trigger_idx : recovery_idx + 1].astype(np.float64)
    body = np.abs(c - o)
    bull = c > o
    bear = c < o
    if direction == "BUY":
        favorable = float(body[bull].sum())
        counter = float(body[bear].sum())
    else:
        favorable = float(body[bear].sum())
        counter = float(body[bull].sum())
    if counter <= 0:
        return favorable if favorable > 0 else 0.0
    return favorable / counter


def _recovery_quality_metrics_np(
    arr: OhlcvArrays,
    trigger_idx: int,
    recovery_idx: int,
    extreme_price: float,
    atr: float,
    *,
    direction: str = "BUY",
) -> dict[str, float | int]:
    if recovery_idx < trigger_idx or atr <= 0:
        return {
            "spring_recovery_atr": 0.0,
            "recovery_duration_bars": 0,
            "recovery_velocity": 0.0,
            "recovery_velocity_atr": 0.0,
            "recovery_close_ratio": 0.0,
            "recovery_acceleration": 0.0,
            "consecutive_higher_closes": 0,
            "positive_close_ratio": 0.0,
            "directional_efficiency": 0.0,
            "noise_ratio": 1.0,
        }

    recovery_close = float(arr.close[recovery_idx])
    if direction == "BUY":
        spring_recovery_atr = (recovery_close - extreme_price) / atr
    else:
        spring_recovery_atr = (extreme_price - recovery_close) / atr
    recovery_duration_bars = max(0, recovery_idx - trigger_idx)
    duration = max(1, recovery_duration_bars)
    recovery_velocity_atr = spring_recovery_atr / duration
    recovery_close_ratio = _recovery_close_ratio_np(
        arr,
        trigger_idx,
        recovery_idx,
        direction=direction,
    )

    quality_end = min(recovery_idx + RECOVERY_QUALITY_BARS, arr.length - 1)
    quality_closes = arr.close[recovery_idx : quality_end + 1].astype(np.float64)
    deltas = np.diff(quality_closes) if len(quality_closes) >= 2 else np.array([], dtype=np.float64)
    if direction == "SELL":
        positive_mask = deltas < 0
    else:
        positive_mask = deltas > 0
    positive_ratio = float(positive_mask.sum() / max(len(deltas), 1))
    consecutive = 0
    for d in reversed(deltas):
        if direction == "SELL" and d < 0:
            consecutive += 1
        elif direction == "BUY" and d > 0:
            consecutive += 1
        else:
            break

    if len(deltas) >= 2:
        first_vel = abs(float(deltas[0])) / atr if atr > 0 else 0.0
        last_vel = abs(float(deltas[-1])) / atr if atr > 0 else 0.0
        recovery_acceleration = last_vel - first_vel
        if direction == "SELL":
            recovery_acceleration = -recovery_acceleration
    else:
        recovery_acceleration = 0.0

    if len(deltas) >= 1 and len(quality_closes) >= 2:
        net = abs(float(quality_closes[-1] - quality_closes[0]))
        path = float(np.abs(deltas).sum())
        directional_efficiency = net / path if path > 0 else 0.0
    else:
        directional_efficiency = 0.0
    noise_ratio = 1.0 - directional_efficiency

    return {
        "spring_recovery_atr": spring_recovery_atr,
        "recovery_duration_bars": recovery_duration_bars,
        "recovery_velocity": recovery_velocity_atr,
        "recovery_velocity_atr": recovery_velocity_atr,
        "recovery_close_ratio": recovery_close_ratio,
        "recovery_acceleration": recovery_acceleration,
        "consecutive_higher_closes": consecutive,
        "positive_close_ratio": positive_ratio,
        "directional_efficiency": directional_efficiency,
        "noise_ratio": noise_ratio,
    }


def _learning_log_features_np(
    arr: OhlcvArrays,
    trigger_idx: int,
    recovery_idx: int,
    direction: str,
    atr: float,
    range_width: float,
) -> dict[str, float | int]:
    defaults = {
        "trend_age_bars": 0,
        "impulse_quality": 0.0,
        "impulse_atr_ratio": 0.0,
        "stagnation_duration": 0,
        "stagnation_width": 0.0,
        "breakout_velocity": 0.0,
        "breakout_momentum_ratio": 0.0,
        "tail_move_after_entry": 0.0,
    }
    if trigger_idx < 0 or recovery_idx >= arr.length or atr <= 0:
        return defaults

    lookback = min(20, trigger_idx)
    closes = arr.close[trigger_idx - lookback : trigger_idx + 1].astype(np.float64)
    impulse_quality = 0.0
    impulse_atr = 0.0
    trend_age = 0
    if len(closes) >= 3:
        net = abs(float(closes[-1] - closes[0]))
        path = float(np.abs(np.diff(closes)).sum())
        impulse_quality = net / path if path > 0 else 0.0
        impulse_atr = net / atr
        for i in range(len(closes) - 1, 0, -1):
            delta = closes[i] - closes[i - 1]
            if direction == "BUY" and delta > 0:
                trend_age += 1
            elif direction == "SELL" and delta < 0:
                trend_age += 1
            else:
                break

    span = max(1, recovery_idx - trigger_idx)
    breakout_move = abs(float(arr.close[recovery_idx]) - float(arr.close[trigger_idx]))
    breakout_velocity = breakout_move / (span * atr)
    stagnation_duration = max(0, span - 1)
    stagnation_width = range_width / atr if atr > 0 else 0.0
    vol = max(1.0, float(arr.volume[trigger_idx]))
    return {
        "trend_age_bars": int(trend_age),
        "impulse_quality": round(float(impulse_quality), 4),
        "impulse_atr_ratio": round(float(impulse_atr), 4),
        "stagnation_duration": int(stagnation_duration),
        "stagnation_width": round(float(stagnation_width), 4),
        "breakout_velocity": round(float(breakout_velocity), 4),
        "breakout_momentum_ratio": round(float(breakout_velocity * vol), 4),
        "tail_move_after_entry": round(float(breakout_move / atr), 4),
    }


def _liquidity_distances_np(
    arr: OhlcvArrays,
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
    day_norm = arr.datetime_ns.astype("datetime64[D]")
    current_day = day_norm[bar_index]
    day_mask = day_norm == current_day
    day_high = float(np.max(arr.high[day_mask]))
    day_low = float(np.min(arr.low[day_mask]))
    session = resolve_cspa_session_type_ns(int(arr.datetime_ns[bar_index]))
    day_indices = np.where(day_mask)[0]
    if session == "LONDON":
        hour_mask = np.array([8 <= _bar_hour_ns(int(arr.datetime_ns[i])) < 17 for i in day_indices])
    elif session == "NY":
        hour_mask = np.array([13 <= _bar_hour_ns(int(arr.datetime_ns[i])) < 22 for i in day_indices])
    elif session == "ASIA":
        hour_mask = np.array([_bar_hour_ns(int(arr.datetime_ns[i])) < 8 for i in day_indices])
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


def _count_spring_attempts_np(
    h1: OhlcvArrays,
    accumulation: AccumulationRange,
    spring_ts_ns: int,
    support_level: float,
    h1_atr: np.ndarray,
) -> int:
    if not accumulation.is_valid:
        return 1

    ar_ns = _ts_ns(accumulation.ar_timestamp)
    attempts = 0
    in_cluster = False
    for i in range(h1.length):
        ts_ns = int(h1.datetime_ns[i])
        if ts_ns < ar_ns:
            continue
        if ts_ns >= spring_ts_ns:
            break

        atr = atr_at_index(h1_atr, i)
        if atr <= 0:
            continue

        min_depth = SPRING_MIN_DEPTH_ATR * atr
        max_depth = SPRING_MAX_DEPTH_ATR * atr
        low = float(h1.low[i])
        if low >= support_level:
            in_cluster = False
            continue

        penetration = support_level - low
        if penetration < min_depth or penetration > max_depth:
            in_cluster = False
            continue

        recovered = float(h1.close[i]) >= support_level
        if not recovered and i + 1 < h1.length:
            recovered = float(h1.close[i + 1]) >= support_level

        if recovered:
            if not in_cluster:
                attempts += 1
                in_cluster = True
        else:
            in_cluster = False

    return attempts + 1


def _count_upthrust_attempts_np(
    h1: OhlcvArrays,
    accumulation: AccumulationRange,
    upthrust_ts_ns: int,
    resistance_level: float,
    h1_atr: np.ndarray,
) -> int:
    if not accumulation.is_valid:
        return 1

    ar_ns = _ts_ns(accumulation.ar_timestamp)
    attempts = 0
    in_cluster = False
    for i in range(h1.length):
        ts_ns = int(h1.datetime_ns[i])
        if ts_ns < ar_ns:
            continue
        if ts_ns >= upthrust_ts_ns:
            break

        atr = atr_at_index(h1_atr, i)
        if atr <= 0:
            continue

        high = float(h1.high[i])
        close = float(h1.close[i])
        if high <= resistance_level or close >= resistance_level:
            in_cluster = False
            continue

        height = high - resistance_level
        if height < UPTHRUST_MIN_HEIGHT_ATR * atr or height > UPTHRUST_MAX_HEIGHT_ATR * atr:
            in_cluster = False
            continue
        if _upthrust_wick_ratio_np(
            float(h1.open[i]), high, float(h1.low[i]), close
        ) < UPTHRUST_WICK_RATIO_MIN:
            in_cluster = False
            continue

        if not in_cluster:
            attempts += 1
            in_cluster = True

    return max(1, attempts)


def _build_wyckoff_features_np(
    *,
    exec_arr: OhlcvArrays,
    h1: OhlcvArrays,
    h1_end: int,
    h1_atr: np.ndarray,
    accumulation: AccumulationRange,
    trigger_idx: int,
    recovery_idx: int,
    extreme_price: float,
    penetration_or_height: float,
    pattern_duration_bars: int,
    attempt_number: int,
    volume_ratio: float,
    entry_price: float,
    atr: float,
    pair: str,
    htf_cache: HtfDirectionCache,
    reversal_pattern: str = "SPRING",
    direction: str = "BUY",
    spread_pips: float = WYCKOFF_BT_SPREAD_PIPS,
) -> WyckoffFeatures:
    recovery_ts_ns = int(exec_arr.datetime_ns[recovery_idx])
    band_width = 2.0 * SUPPORT_BAND_ATR * atr
    if reversal_pattern == "SPRING":
        support_penetration_percent = (
            (penetration_or_height / band_width * 100.0) if band_width > 0 else 0.0
        )
    else:
        support_penetration_percent = 0.0

    pip = pip_size_for_pair(pair)
    spring_velocity = (
        (penetration_or_height / pip / max(pattern_duration_bars, 1)) if pip > 0 else 0.0
    )
    recovery = _recovery_quality_metrics_np(
        exec_arr,
        trigger_idx,
        recovery_idx,
        extreme_price,
        atr,
        direction=direction,
    )
    kalman = _neutral_kalman_features()
    vol_pct = _volatility_percentile_wyckoff_np(h1_atr, h1_end)
    session = resolve_cspa_session_type_ns(recovery_ts_ns)  # type: ignore[assignment]
    htf = htf_cache.direction_at_ns(recovery_ts_ns)
    range_low = accumulation.support_level
    range_high = accumulation.resistance_level
    range_width = max(range_high - range_low, 0.0)
    pos_in_range = _position_in_range(entry_price, range_low, range_high)
    liquidity = _liquidity_distances_np(exec_arr, recovery_idx, entry_price, atr)
    learning = _learning_log_features_np(
        exec_arr,
        trigger_idx,
        recovery_idx,
        direction,
        atr,
        range_width,
    )
    macro_phase = classify_wyckoff_macro_phase(
        accumulation=accumulation,
        entry_price=entry_price,
        htf_trend_d1=htf,
        reversal_pattern=reversal_pattern,  # type: ignore[arg-type]
    )
    spring_depth = (
        penetration_or_height / atr if atr > 0 and reversal_pattern == "SPRING" else 0.0
    )
    upthrust_height = (
        penetration_or_height / atr if atr > 0 and reversal_pattern == "UPTHRUST" else 0.0
    )
    if reversal_pattern == "SPRING":
        wick_ratio = _spring_wick_ratio_np(
            float(exec_arr.open[trigger_idx]),
            float(exec_arr.high[trigger_idx]),
            float(exec_arr.low[trigger_idx]),
            float(exec_arr.close[trigger_idx]),
        )
    else:
        wick_ratio = _upthrust_wick_ratio_np(
            float(exec_arr.open[trigger_idx]),
            float(exec_arr.high[trigger_idx]),
            float(exec_arr.low[trigger_idx]),
            float(exec_arr.close[trigger_idx]),
        )
    phase_label = "SPRING" if reversal_pattern == "SPRING" else "ACCUMULATION"
    h1_atr_val = atr_at_index(h1_atr, h1_end) or atr
    h1_clipped = slice_arrays(h1, 0, h1_end)

    return WyckoffFeatures(
        phase_identified=phase_label,  # type: ignore[arg-type]
        support_test_count=accumulation.test_count,
        days_in_accumulation=accumulation.days_in_accumulation,
        phase_b_duration=accumulation.phase_b_duration,
        range_width_atr=accumulation.range_width_atr,
        spring_attempt_number=attempt_number,
        spring_depth_atr=spring_depth,
        spring_velocity=spring_velocity,
        spring_duration_bars=pattern_duration_bars,
        support_penetration_percent=support_penetration_percent,
        spring_volume_ratio=volume_ratio,
        spring_recovery_time=int(recovery["recovery_duration_bars"]),
        resistance_distance=(
            max(0.0, (accumulation.resistance_level - entry_price) / atr) if atr > 0 else 0.0
        ),
        spring_wick_ratio=wick_ratio,
        range_compression=_range_compression_np(h1_clipped, accumulation, recovery_ts_ns, h1_atr_val),
        spring_recovery_atr=float(recovery["spring_recovery_atr"]),
        recovery_duration_bars=int(recovery["recovery_duration_bars"]),
        recovery_velocity=float(recovery["recovery_velocity"]),
        recovery_velocity_atr=float(recovery["recovery_velocity_atr"]),
        recovery_close_ratio=float(recovery["recovery_close_ratio"]),
        recovery_acceleration=float(recovery["recovery_acceleration"]),
        kalman_velocity_at_entry=float(kalman["kalman_velocity_at_entry"]),
        velocity_positive_bars=int(kalman["velocity_positive_bars"]),
        kalman_noise_ratio=float(kalman["kalman_noise_ratio"]),
        consecutive_higher_closes=int(recovery["consecutive_higher_closes"]),
        positive_close_ratio=float(recovery["positive_close_ratio"]),
        directional_efficiency=float(recovery["directional_efficiency"]),
        noise_ratio=float(recovery["noise_ratio"]),
        session_type=session,  # type: ignore[arg-type]
        volatility_percentile=vol_pct,
        atr_regime=_atr_regime(vol_pct),  # type: ignore[arg-type]
        adr_remaining=_compute_adr_remaining_np(exec_arr, recovery_idx, atr),
        distance_weekly_low=_distance_to_rolling_low_np(
            exec_arr, recovery_idx, WEEKLY_BARS * 4, entry_price, atr
        ),
        distance_monthly_low=_distance_to_rolling_low_np(
            exec_arr, recovery_idx, MONTHLY_BARS * 4, entry_price, atr
        ),
        htf_trend_d1=htf,
        phase_b_ut_occurred=accumulation.phase_b_ut_occurred,
        reversal_pattern=reversal_pattern,  # type: ignore[arg-type]
        wr_mode=wr_mode_from_direction(direction),  # type: ignore[arg-type]
        wyckoff_phase=macro_phase,
        range_width=range_width,
        range_age_bars=_range_age_bars_np(accumulation, recovery_ts_ns),
        upthrust_height=upthrust_height,
        recovery_speed=float(recovery["recovery_velocity"]),
        volume_ratio=volume_ratio,
        volume_percentile=_volume_percentile_np(exec_arr, trigger_idx),
        volume_expansion=max(0.0, volume_ratio - 1.0),
        distance_from_range_low=(entry_price - range_low) / atr if atr > 0 else 0.0,
        distance_from_range_high=(range_high - entry_price) / atr if atr > 0 else 0.0,
        position_in_range=pos_in_range,
        atr_ratio=accumulation.range_width_atr,
        minutes_from_session_open=minutes_from_session_open_ns(recovery_ts_ns, session),
        adr_used=max(0.0, min(1.0, 1.0 - _compute_adr_remaining_np(exec_arr, recovery_idx, atr))),
        distance_daily_high=liquidity["distance_daily_high"],
        distance_daily_low=liquidity["distance_daily_low"],
        distance_session_high=liquidity["distance_session_high"],
        distance_session_low=liquidity["distance_session_low"],
        spread=spread_pips,
        spread_percentile=_spread_percentile(spread_pips),
        tail_move_after_entry=float(learning["tail_move_after_entry"]),
        trend_age_bars=int(learning["trend_age_bars"]),
        impulse_quality=float(learning["impulse_quality"]),
        impulse_atr_ratio=float(learning["impulse_atr_ratio"]),
        stagnation_duration=int(learning["stagnation_duration"]),
        stagnation_width=float(learning["stagnation_width"]),
        breakout_velocity=float(learning["breakout_velocity"]),
        breakout_momentum_ratio=float(learning["breakout_momentum_ratio"]),
    )


def detect_spring_np(
    exec_arr: OhlcvArrays,
    accumulation: AccumulationRange,
    pip: float,
    *,
    pair: str,
    spring_bar_index: int,
    h1: OhlcvArrays,
    h1_atr: np.ndarray,
    exec_atr: np.ndarray,
    htf_cache: HtfDirectionCache,
) -> ReversalSetup | None:
    _ = pip
    if not accumulation.is_valid:
        return None
    idx = spring_bar_index
    if idx < 0 or idx >= exec_arr.length:
        return None
    if int(exec_arr.datetime_ns[idx]) < _ts_ns(accumulation.ar_timestamp):
        return None

    atr = atr_at_index(exec_atr, idx)
    if atr <= 0:
        return None

    support_level = accumulation.support_level
    spring_low = float(exec_arr.low[idx])
    if spring_low >= support_level:
        return None

    penetration = support_level - spring_low
    if penetration < SPRING_MIN_DEPTH_ATR * atr or penetration > SPRING_MAX_DEPTH_ATR * atr:
        return None

    recovery_idx = idx
    if float(exec_arr.close[idx]) < support_level:
        if idx + 1 >= exec_arr.length:
            return None
        if float(exec_arr.close[idx + 1]) < support_level:
            return None
        recovery_idx = idx + 1

    spring_duration_bars = max(1, recovery_idx - idx + 1)
    entry_price = float(exec_arr.close[recovery_idx])
    stop_loss = accumulation.sc_price - SL_BUFFER_ATR * atr
    take_profit = accumulation.ar_price
    risk = entry_price - stop_loss
    reward = take_profit - entry_price
    if risk <= 0:
        return None
    rr = reward / risk
    if rr < MIN_RR:
        return None

    spring_attempt_number = _count_spring_attempts_np(
        h1,
        accumulation,
        int(exec_arr.datetime_ns[idx]),
        support_level,
        h1_atr,
    )
    volume_on_spring = _session_normalized_volume_ratio_np(exec_arr, idx)
    if volume_on_spring < SPRING_VOLUME_RATIO_MIN:
        return None

    h1_end = asof_end_index(h1, int(exec_arr.datetime_ns[recovery_idx]))
    features = _build_wyckoff_features_np(
        exec_arr=exec_arr,
        h1=h1,
        h1_end=h1_end,
        h1_atr=h1_atr,
        accumulation=accumulation,
        trigger_idx=idx,
        recovery_idx=recovery_idx,
        extreme_price=spring_low,
        penetration_or_height=penetration,
        pattern_duration_bars=spring_duration_bars,
        attempt_number=spring_attempt_number,
        volume_ratio=volume_on_spring,
        entry_price=entry_price,
        atr=atr,
        pair=pair.upper(),
        htf_cache=htf_cache,
        reversal_pattern="SPRING",
        direction="BUY",
    )

    return ReversalSetup(
        timestamp=ts_ns_to_pd(int(exec_arr.datetime_ns[recovery_idx])),
        pair=pair.upper(),
        accumulation=accumulation,
        spring_depth_atr=features.spring_depth_atr,
        spring_velocity=features.spring_velocity,
        spring_duration_bars=spring_duration_bars,
        support_penetration_percent=features.support_penetration_percent,
        spring_attempt_number=spring_attempt_number,
        volume_on_spring=volume_on_spring,
        entry_price=round(entry_price, 6),
        stop_loss=round(stop_loss, 6),
        take_profit=round(take_profit, 6),
        risk_reward=round(rr, 4),
        spring_bar_index=idx,
        recovery_bar_index=recovery_idx,
        wyckoff_features=features,
        reason_codes=(),
        direction="BUY",
        reversal_pattern="SPRING",
    )


def detect_upthrust_np(
    exec_arr: OhlcvArrays,
    accumulation: AccumulationRange,
    pip: float,
    *,
    pair: str,
    upthrust_bar_index: int,
    h1: OhlcvArrays,
    h1_atr: np.ndarray,
    exec_atr: np.ndarray,
    htf_cache: HtfDirectionCache,
) -> ReversalSetup | None:
    _ = pip
    if not accumulation.is_valid:
        return None
    idx = upthrust_bar_index
    if idx < 0 or idx >= exec_arr.length:
        return None
    if int(exec_arr.datetime_ns[idx]) < _ts_ns(accumulation.ar_timestamp):
        return None

    atr = atr_at_index(exec_atr, idx)
    if atr <= 0:
        return None

    resistance_level = accumulation.resistance_level
    upthrust_high = float(exec_arr.high[idx])
    close = float(exec_arr.close[idx])

    if upthrust_high <= resistance_level:
        return None
    if close >= resistance_level:
        return None
    if _upthrust_wick_ratio_np(
        float(exec_arr.open[idx]),
        upthrust_high,
        float(exec_arr.low[idx]),
        close,
    ) < UPTHRUST_WICK_RATIO_MIN:
        return None

    height = upthrust_high - resistance_level
    if height < UPTHRUST_MIN_HEIGHT_ATR * atr or height > UPTHRUST_MAX_HEIGHT_ATR * atr:
        return None

    pos_at_high = _position_in_range(
        upthrust_high,
        accumulation.support_level,
        resistance_level,
    )
    if pos_at_high < 0.55:
        return None

    recovery_idx = idx
    if close >= resistance_level:
        if idx + 1 >= exec_arr.length:
            return None
        if float(exec_arr.close[idx + 1]) >= resistance_level:
            return None
        recovery_idx = idx + 1

    volume_on_upthrust = _session_normalized_volume_ratio_np(exec_arr, idx)
    if volume_on_upthrust < UPTHRUST_VOLUME_RATIO_MIN:
        return None

    pattern_duration_bars = max(1, recovery_idx - idx + 1)
    entry_price = float(exec_arr.close[recovery_idx])
    stop_loss = accumulation.ar_price + SL_BUFFER_ATR * atr
    take_profit = accumulation.support_level
    risk = stop_loss - entry_price
    reward = entry_price - take_profit
    if risk <= 0 or reward <= 0:
        return None
    rr = reward / risk
    if rr < MIN_RR:
        return None

    attempt_number = _count_upthrust_attempts_np(
        h1,
        accumulation,
        int(exec_arr.datetime_ns[idx]),
        resistance_level,
        h1_atr,
    )
    h1_end = asof_end_index(h1, int(exec_arr.datetime_ns[recovery_idx]))
    features = _build_wyckoff_features_np(
        exec_arr=exec_arr,
        h1=h1,
        h1_end=h1_end,
        h1_atr=h1_atr,
        accumulation=accumulation,
        trigger_idx=idx,
        recovery_idx=recovery_idx,
        extreme_price=upthrust_high,
        penetration_or_height=height,
        pattern_duration_bars=pattern_duration_bars,
        attempt_number=attempt_number,
        volume_ratio=volume_on_upthrust,
        entry_price=entry_price,
        atr=atr,
        pair=pair.upper(),
        htf_cache=htf_cache,
        reversal_pattern="UPTHRUST",
        direction="SELL",
    )

    return ReversalSetup(
        timestamp=ts_ns_to_pd(int(exec_arr.datetime_ns[recovery_idx])),
        pair=pair.upper(),
        accumulation=accumulation,
        spring_depth_atr=0.0,
        spring_velocity=features.spring_velocity,
        spring_duration_bars=pattern_duration_bars,
        support_penetration_percent=0.0,
        spring_attempt_number=attempt_number,
        volume_on_spring=volume_on_upthrust,
        entry_price=round(entry_price, 6),
        stop_loss=round(stop_loss, 6),
        take_profit=round(take_profit, 6),
        risk_reward=round(rr, 4),
        spring_bar_index=idx,
        recovery_bar_index=recovery_idx,
        wyckoff_features=features,
        reason_codes=(),
        direction="SELL",
        reversal_pattern="UPTHRUST",
        upthrust_height_atr=features.upthrust_height,
    )


def _detect_reversal_at_bar_np(
    exec_arr: OhlcvArrays,
    accumulation: AccumulationRange,
    pip: float,
    *,
    pair: str,
    bar_index: int,
    h1: OhlcvArrays,
    h1_atr: np.ndarray,
    exec_atr: np.ndarray,
    htf_cache: HtfDirectionCache,
) -> ReversalSetup | None:
    spring = detect_spring_np(
        exec_arr,
        accumulation,
        pip,
        pair=pair,
        spring_bar_index=bar_index,
        h1=h1,
        h1_atr=h1_atr,
        exec_atr=exec_atr,
        htf_cache=htf_cache,
    )
    if spring is not None:
        return spring
    return detect_upthrust_np(
        exec_arr,
        accumulation,
        pip,
        pair=pair,
        upthrust_bar_index=bar_index,
        h1=h1,
        h1_atr=h1_atr,
        exec_atr=exec_atr,
        htf_cache=htf_cache,
    )


def _enrich_setup_with_outcome_np(
    setup: ReversalSetup,
    exec_arr: OhlcvArrays,
) -> ReversalSetup:
    excursions = compute_trade_excursions_np(
        exec_arr,
        setup.recovery_bar_index,
        setup.entry_price,
        setup.stop_loss,
        setup.take_profit,
        max_holding_bars=MAX_HOLDING_BARS,
        direction=setup.direction,
    )
    base = setup.wyckoff_features.as_dict()
    base.update(excursions)
    fields = WyckoffFeatures.__dataclass_fields__
    features = WyckoffFeatures(**{k: base[k] for k in fields})
    return ReversalSetup(
        timestamp=setup.timestamp,
        pair=setup.pair,
        accumulation=setup.accumulation,
        spring_depth_atr=setup.spring_depth_atr,
        spring_velocity=setup.spring_velocity,
        spring_duration_bars=setup.spring_duration_bars,
        support_penetration_percent=setup.support_penetration_percent,
        spring_attempt_number=setup.spring_attempt_number,
        volume_on_spring=setup.volume_on_spring,
        entry_price=setup.entry_price,
        stop_loss=setup.stop_loss,
        take_profit=setup.take_profit,
        risk_reward=setup.risk_reward,
        spring_bar_index=setup.spring_bar_index,
        recovery_bar_index=setup.recovery_bar_index,
        wyckoff_features=features,
        reason_codes=setup.reason_codes,
        direction=setup.direction,
        reversal_pattern=setup.reversal_pattern,
        upthrust_height_atr=setup.upthrust_height_atr,
        candidate_score=setup.candidate_score,
        ws_sim=None,
    )


def detect_wyckoff_reversal_setups_np(
    h1: OhlcvArrays,
    exec_arr: OhlcvArrays,
    pair: str,
    *,
    lookback_bars: int = LOOKBACK_BARS,
    max_setups_per_day: int = MAX_SETUPS_PER_DAY,
    include_outcomes: bool = False,
    progress_hook: Callable[[int], None] | None = None,
    resume_from_bar: int | None = None,
    initial_setups: list[ReversalSetup] | None = None,
    on_checkpoint: Callable[[int, list[ReversalSetup], dict[str, Any] | None], None] | None = None,
    checkpoint_every: int = 0,
) -> list[ReversalSetup]:
    """Walk-forward scan — H1 Trading Range + exec-TF Spring / Upthrust (numpy-only)."""
    if pair.upper() not in ALLOWED_PAIRS:
        return []

    min_required_h1 = ATR_PERIOD + MIN_BARS_AFTER_SC + 5
    min_required_exec = ATR_PERIOD + 5
    if h1.length < min_required_h1 or exec_arr.length < min_required_exec:
        return []

    h1_atr = compute_atr_np(h1.high, h1.low, h1.close, ATR_PERIOD)
    exec_atr = compute_atr_np(exec_arr.high, exec_arr.low, exec_arr.close, ATR_PERIOD)
    h1_end_by_exec = np.searchsorted(h1.datetime_ns, exec_arr.datetime_ns, side="right") - 1
    htf_cache = HtfDirectionCache(h1)

    setups: list[ReversalSetup] = list(initial_setups) if initial_setups else []
    day_counts: dict[date, int] = {}
    for setup in setups:
        if max_setups_per_day > 0:
            day_counts[setup.timestamp.date()] = day_counts.get(setup.timestamp.date(), 0) + 1

    pip = pip_size_for_pair(pair)
    min_start = max(
        ATR_PERIOD + 2,
        min(lookback_bars // 3, max(exec_arr.length // 3, ATR_PERIOD + 2)),
    )
    if min_start >= exec_arr.length:
        return setups

    loop_start = max(min_start, resume_from_bar) if resume_from_bar is not None else min_start
    for i in range(loop_start, exec_arr.length):
        if progress_hook is not None:
            progress_hook(1)

        if i > loop_start and on_checkpoint and checkpoint_every > 0:
            bars_done = i - loop_start
            if bars_done % checkpoint_every == 0:
                on_checkpoint(i, setups, None)

        h1_end = int(h1_end_by_exec[i])
        if h1_end < 0:
            continue

        accumulation = detect_accumulation_range_np(h1, h1_end, h1_atr, lookback_bars)
        if accumulation is None or not accumulation.is_valid:
            continue

        reversal = _detect_reversal_at_bar_np(
            exec_arr,
            accumulation,
            pip,
            pair=pair,
            bar_index=i,
            h1=h1,
            h1_atr=h1_atr,
            exec_atr=exec_atr,
            htf_cache=htf_cache,
        )
        if reversal is None:
            continue

        reversal = ReversalSetup(
            timestamp=reversal.timestamp,
            pair=pair.upper(),
            accumulation=reversal.accumulation,
            spring_depth_atr=reversal.spring_depth_atr,
            spring_velocity=reversal.spring_velocity,
            spring_duration_bars=reversal.spring_duration_bars,
            support_penetration_percent=reversal.support_penetration_percent,
            spring_attempt_number=reversal.spring_attempt_number,
            volume_on_spring=reversal.volume_on_spring,
            entry_price=reversal.entry_price,
            stop_loss=reversal.stop_loss,
            take_profit=reversal.take_profit,
            risk_reward=reversal.risk_reward,
            spring_bar_index=reversal.spring_bar_index,
            recovery_bar_index=reversal.recovery_bar_index,
            wyckoff_features=reversal.wyckoff_features,
            reason_codes=reversal.reason_codes,
            direction=reversal.direction,
            reversal_pattern=reversal.reversal_pattern,
            upthrust_height_atr=reversal.upthrust_height_atr,
        )

        day = reversal.timestamp.date()
        if max_setups_per_day > 0:
            if day_counts.get(day, 0) >= max_setups_per_day:
                continue
            day_counts[day] = day_counts.get(day, 0) + 1

        if include_outcomes:
            reversal = _enrich_setup_with_outcome_np(reversal, exec_arr)
        setups.append(reversal)

    if on_checkpoint:
        on_checkpoint(exec_arr.length, setups, None)
    if progress_hook is not None:
        progress_hook(0)
    return setups
