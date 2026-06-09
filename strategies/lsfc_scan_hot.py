"""
strategies/lsfc_scan_hot.py — NumPy-only LSFC setup scan (precompute hot path).
"""

from __future__ import annotations

from datetime import date
from typing import Callable

import numpy as np

from strategies.bt_ohlcv import resolve_bar_position_np, ts_ns_to_pd
from strategies.cspa_arrays import OhlcvArrays, atr_at_index, compute_atr_np
from strategies.london_sweep_failure import (
    MIN_RR,
    PIP_SIZE,
    LsfcConfig,
    LsfcSetup,
    _resolve_lsfc_session_hours,
    load_lsfc_config,
)

__all__ = ["detect_london_sweep_failure_setups_np"]

_ATR_PERIOD = 14


def _date_from_day_norm(day: np.datetime64) -> date:
    y, m, d = (int(x) for x in str(day.astype("datetime64[D]"))[:10].split("-"))
    return date(y, m, d)


def _day_norm_from_datetime_ns(datetime_ns: np.ndarray) -> np.ndarray:
    return datetime_ns.astype("datetime64[ns]").astype("datetime64[D]")


def _hours_from_datetime_ns(datetime_ns: np.ndarray) -> np.ndarray:
    dt = datetime_ns.astype("datetime64[ns]")
    day = dt.astype("datetime64[D]")
    return ((dt - day) / np.timedelta64(1, "h")).astype(np.int32)


def _build_day_groups(day_norm: np.ndarray) -> list[tuple[np.datetime64, int, int]]:
    """Return ``(day, start_idx, end_idx_exclusive)`` per calendar day."""
    n = len(day_norm)
    if n == 0:
        return []
    changes = np.concatenate([[True], day_norm[1:] != day_norm[:-1]])
    starts = np.where(changes)[0]
    groups: list[tuple[np.datetime64, int, int]] = []
    for i, start in enumerate(starts):
        end = int(starts[i + 1]) if i + 1 < len(starts) else n
        groups.append((day_norm[start], int(start), end))
    return groups


def _atr_at_bar_np(
    exec_index: int,
    exec: OhlcvArrays,
    structure: OhlcvArrays,
    structure_atr: np.ndarray,
) -> float:
    if exec_index < len(structure_atr):
        val = atr_at_index(structure_atr, exec_index)
        if val > 0.0:
            return val
    ts_ns = int(exec.datetime_ns[exec_index])
    struct_pos = resolve_bar_position_np(structure, ts_ns)
    if struct_pos is not None:
        val = atr_at_index(structure_atr, struct_pos)
        if val > 0.0:
            return val
    return max(float(exec.high[exec_index] - exec.low[exec_index]), PIP_SIZE * 10)


def _pool_levels_np(
    high: np.ndarray,
    low: np.ndarray,
    bar_index: int,
    lookback: int,
) -> tuple[float, float] | None:
    """Swing high/low over prior ``lookback`` bars (current bar excluded)."""
    if bar_index < lookback:
        return None
    start = bar_index - lookback
    return float(np.max(high[start:bar_index])), float(np.min(low[start:bar_index]))


def _bar_in_session_np(
    bar_index: int,
    session_day_norm: np.datetime64,
    london_hours: range,
    day_norm: np.ndarray,
    hour: np.ndarray,
) -> bool:
    if day_norm[bar_index] != session_day_norm:
        return False
    return int(hour[bar_index]) in london_hours


def _detect_failure_pattern_np(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    sweep_index: int,
    direction: str,
    sweep_level: float,
    failure_window: int,
    max_depth_price: float,
    session_day_norm: np.datetime64,
    london_hours: range,
    day_norm: np.ndarray,
    hour: np.ndarray,
    n_bars: int,
) -> tuple[bool, float, float, int]:
    from strategies.lsfc_scan_numba import lsfc_scan_numba_active

    if lsfc_scan_numba_active():
        from strategies.lsfc_scan_numba import (
            detect_failure_pattern_numba,
            direction_code,
            london_hours_mask,
        )

        session_day = np.int64(session_day_norm.astype("datetime64[D]").astype(np.int64))
        mask = london_hours_mask(london_hours)
        ok, depth, extreme, turn_idx = detect_failure_pattern_numba(
            high,
            low,
            close,
            sweep_index,
            direction_code(direction),
            sweep_level,
            failure_window,
            max_depth_price,
            session_day,
            day_norm.astype(np.int64),
            hour,
            mask,
            n_bars,
        )
        if ok <= 0:
            return False, 0.0, 0.0, -1
        return True, depth, extreme, turn_idx

    window_end = min(sweep_index + failure_window, n_bars - 1)
    retracement_seen = False
    failure_depth = 0.0
    failure_extreme: float | None = None
    deepest_bar = sweep_index
    turn_back_index = -1

    for j in range(sweep_index + 1, window_end + 1):
        if not _bar_in_session_np(j, session_day_norm, london_hours, day_norm, hour):
            break

        if direction == "BUY":
            if float(low[j]) < sweep_level:
                retracement_seen = True
                pull = float(low[j])
                depth = sweep_level - pull
                if depth > failure_depth:
                    failure_depth = depth
                    deepest_bar = j
                failure_extreme = pull if failure_extreme is None else min(failure_extreme, pull)
        else:
            if float(high[j]) > sweep_level:
                retracement_seen = True
                pull = float(high[j])
                depth = pull - sweep_level
                if depth > failure_depth:
                    failure_depth = depth
                    deepest_bar = j
                failure_extreme = pull if failure_extreme is None else max(failure_extreme, pull)

    if not retracement_seen or failure_depth <= 0 or failure_depth > max_depth_price:
        return False, 0.0, 0.0, -1
    if failure_extreme is None:
        return False, 0.0, 0.0, -1

    for j in range(deepest_bar, window_end + 1):
        if not _bar_in_session_np(j, session_day_norm, london_hours, day_norm, hour):
            break
        if direction == "BUY" and float(close[j]) > sweep_level:
            turn_back_index = j
            break
        if direction == "SELL" and float(close[j]) < sweep_level:
            turn_back_index = j
            break

    if turn_back_index < 0:
        return False, 0.0, 0.0, -1

    return True, failure_depth, failure_extreme, turn_back_index


def _scan_trigger_bar_np(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    start_index: int,
    direction: str,
    trigger_level: float,
    session_day_norm: np.datetime64,
    london_hours: range,
    day_norm: np.ndarray,
    hour: np.ndarray,
    n_bars: int,
) -> int | None:
    from strategies.lsfc_scan_numba import lsfc_scan_numba_active

    if lsfc_scan_numba_active():
        from strategies.lsfc_scan_numba import (
            direction_code,
            london_hours_mask,
            scan_trigger_bar_numba,
        )

        session_day = np.int64(session_day_norm.astype("datetime64[D]").astype(np.int64))
        mask = london_hours_mask(london_hours)
        idx = scan_trigger_bar_numba(
            high,
            low,
            close,
            start_index,
            direction_code(direction),
            trigger_level,
            session_day,
            day_norm.astype(np.int64),
            hour,
            mask,
            n_bars,
        )
        return None if idx < 0 else int(idx)

    for j in range(start_index, n_bars):
        if day_norm[j] != session_day_norm:
            break
        if int(hour[j]) not in london_hours:
            continue

        if direction == "BUY":
            if float(close[j]) >= trigger_level or float(high[j]) >= trigger_level:
                return j
        elif float(close[j]) <= trigger_level or float(low[j]) <= trigger_level:
            return j
    return None


def _build_lsfc_setup_np(
    open_: np.ndarray,
    close: np.ndarray,
    datetime_ns: np.ndarray,
    trigger_index: int,
    pair_name: str,
    direction: str,
    pool_high: float,
    pool_low: float,
    sweep_level: float,
    sweep_extreme: float,
    sweep_bar_index: int,
    failure_extreme: float,
    failure_depth: float,
    sweep_hl_distance_atr: float,
    atr_val: float,
    structure: OhlcvArrays,
    trigger_offset_pips: float,
) -> LsfcSetup | None:
    entry = float(close[trigger_index])
    offset = trigger_offset_pips * PIP_SIZE
    bar_open = float(open_[trigger_index])

    if direction == "BUY":
        stop_loss = min(failure_extreme, bar_open)
        risk = entry - stop_loss
        if risk <= 0:
            return None
        take_profit = entry + MIN_RR * risk
        sweep_distance = sweep_extreme - sweep_level
    else:
        stop_loss = max(failure_extreme, bar_open)
        risk = stop_loss - entry
        if risk <= 0:
            return None
        take_profit = entry - MIN_RR * risk
        sweep_distance = sweep_level - sweep_extreme

    ts_ns = int(datetime_ns[trigger_index])
    struct_pos = resolve_bar_position_np(structure, ts_ns)
    if struct_pos is not None:
        bar_index = struct_pos
    else:
        # Match pandas fallback: use exec trigger index when no structure datetime match.
        bar_index = trigger_index
    bar_index = min(max(bar_index, 0), structure.length - 1)

    return LsfcSetup(
        timestamp=ts_ns_to_pd(ts_ns),
        pair=pair_name,
        direction=direction,
        pool_high=pool_high,
        pool_low=pool_low,
        sweep_level=sweep_level,
        sweep_extreme=sweep_extreme,
        sweep_bar_index=sweep_bar_index,
        failure_extreme=failure_extreme,
        failure_retracement_depth=failure_depth,
        sweep_high_low_distance_atr=sweep_hl_distance_atr,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        sweep_distance=sweep_distance,
        atr=atr_val,
        bar_index=bar_index,
    )


def detect_london_sweep_failure_setups_np(
    exec: OhlcvArrays,
    structure: OhlcvArrays,
    pair_name: str,
    config: LsfcConfig | None = None,
    progress_hook: Callable[[int, int], None] | None = None,
) -> list[LsfcSetup]:
    """
    Detect LSFC patterns (Sweep → Failure → Continuation) per calendar day.
    At most one setup per day (first match wins).
    """
    cfg = config or load_lsfc_config()
    n = exec.length
    if n == 0:
        return []

    structure_atr = compute_atr_np(
        structure.high,
        structure.low,
        structure.close,
        _ATR_PERIOD,
    )

    day_norm = _day_norm_from_datetime_ns(exec.datetime_ns)
    hour = _hours_from_datetime_ns(exec.datetime_ns)

    high = exec.high
    low = exec.low
    close = exec.close
    open_ = exec.open

    setups: list[LsfcSetup] = []
    day_groups = _build_day_groups(day_norm)
    day_total = len(day_groups)

    for day_idx, (session_day_norm, day_start, day_end) in enumerate(day_groups, start=1):
        if progress_hook is not None:
            progress_hook(day_idx, day_total)

        session_date = _date_from_day_norm(session_day_norm)
        london_hours = _resolve_lsfc_session_hours(session_date)
        emitted = False

        for sweep_index in range(day_start, day_end):
            if emitted:
                break

            if int(hour[sweep_index]) not in london_hours:
                continue

            pools = _pool_levels_np(high, low, sweep_index, cfg.lookback_candles)
            if pools is None:
                continue
            pool_high, pool_low = pools

            high_i = float(high[sweep_index])
            low_i = float(low[sweep_index])
            direction: str | None = None
            sweep_level = 0.0
            sweep_extreme = 0.0

            if high_i > pool_high and low_i < pool_low:
                up_dist = high_i - pool_high
                down_dist = pool_low - low_i
                if up_dist >= down_dist:
                    direction = "BUY"
                    sweep_level = pool_high
                    sweep_extreme = high_i
                else:
                    direction = "SELL"
                    sweep_level = pool_low
                    sweep_extreme = low_i
            elif high_i > pool_high:
                direction = "BUY"
                sweep_level = pool_high
                sweep_extreme = high_i
            elif low_i < pool_low:
                direction = "SELL"
                sweep_level = pool_low
                sweep_extreme = low_i
            else:
                continue

            atr_val = _atr_at_bar_np(sweep_index, exec, structure, structure_atr)
            if atr_val <= 0:
                continue

            max_depth = cfg.retracement_atr_ratio * atr_val
            confirmed, failure_depth, failure_extreme, turn_back_index = _detect_failure_pattern_np(
                high,
                low,
                close,
                sweep_index,
                direction,
                sweep_level,
                cfg.failure_window,
                max_depth,
                session_day_norm,
                london_hours,
                day_norm,
                hour,
                n,
            )
            if not confirmed:
                continue

            trigger_offset = cfg.trigger_offset_pips * PIP_SIZE
            if direction == "BUY":
                trigger_level = sweep_level + trigger_offset
            else:
                trigger_level = sweep_level - trigger_offset

            trigger_index = _scan_trigger_bar_np(
                high,
                low,
                close,
                turn_back_index,
                direction,
                trigger_level,
                session_day_norm,
                london_hours,
                day_norm,
                hour,
                n,
            )
            if trigger_index is None:
                continue

            pool_width_atr = (pool_high - pool_low) / atr_val if atr_val > 0 else 0.0
            setup = _build_lsfc_setup_np(
                open_,
                close,
                exec.datetime_ns,
                trigger_index,
                pair_name,
                direction,
                pool_high,
                pool_low,
                sweep_level,
                sweep_extreme,
                sweep_index,
                failure_extreme,
                failure_depth,
                pool_width_atr,
                atr_val,
                structure,
                cfg.trigger_offset_pips,
            )
            if setup is not None:
                setups.append(setup)
                emitted = True

    return setups
