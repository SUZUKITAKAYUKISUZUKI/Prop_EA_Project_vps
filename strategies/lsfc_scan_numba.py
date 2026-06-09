"""Numba kernels for LSFC precompute scan (``LSFC_SCAN_NUMBA=1``)."""

from __future__ import annotations

import numpy as np

from strategies.scan_numba_util import njit, scan_numba_active

_DIR_BUY = 0
_DIR_SELL = 1


def lsfc_scan_numba_enabled() -> bool:
    return scan_numba_active("LSFC_SCAN_NUMBA", default=False)


def lsfc_scan_numba_active() -> bool:
    return lsfc_scan_numba_enabled()


def london_hours_mask(london_hours: range) -> np.ndarray:
    mask = np.zeros(24, dtype=np.int8)
    for h in london_hours:
        if 0 <= int(h) < 24:
            mask[int(h)] = 1
    return mask


@njit(cache=True)
def _bar_in_session_numba(
    day_norm: np.ndarray,
    hour: np.ndarray,
    bar_index: int,
    session_day: np.int64,
    london_mask: np.ndarray,
) -> bool:
    if day_norm[bar_index] != session_day:
        return False
    h = hour[bar_index]
    if h < 0 or h >= 24:
        return False
    return london_mask[h] == 1


@njit(cache=True)
def detect_failure_pattern_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    sweep_index: int,
    direction: int,
    sweep_level: float,
    failure_window: int,
    max_depth_price: float,
    session_day: np.int64,
    day_norm: np.ndarray,
    hour: np.ndarray,
    london_mask: np.ndarray,
    n_bars: int,
) -> tuple:
    window_end = min(sweep_index + failure_window, n_bars - 1)
    retracement_seen = False
    failure_depth = 0.0
    failure_extreme = 0.0
    has_extreme = False
    deepest_bar = sweep_index
    turn_back_index = -1

    for j in range(sweep_index + 1, window_end + 1):
        if not _bar_in_session_numba(day_norm, hour, j, session_day, london_mask):
            break
        if direction == _DIR_BUY:
            if low[j] < sweep_level:
                retracement_seen = True
                pull = low[j]
                depth = sweep_level - pull
                if depth > failure_depth:
                    failure_depth = depth
                    deepest_bar = j
                if not has_extreme or pull < failure_extreme:
                    failure_extreme = pull
                    has_extreme = True
        else:
            if high[j] > sweep_level:
                retracement_seen = True
                pull = high[j]
                depth = pull - sweep_level
                if depth > failure_depth:
                    failure_depth = depth
                    deepest_bar = j
                if not has_extreme or pull > failure_extreme:
                    failure_extreme = pull
                    has_extreme = True

    if not retracement_seen or failure_depth <= 0.0 or failure_depth > max_depth_price or not has_extreme:
        return 0, 0.0, 0.0, -1

    for j in range(deepest_bar, window_end + 1):
        if not _bar_in_session_numba(day_norm, hour, j, session_day, london_mask):
            break
        if direction == _DIR_BUY and close[j] > sweep_level:
            turn_back_index = j
            break
        if direction == _DIR_SELL and close[j] < sweep_level:
            turn_back_index = j
            break

    if turn_back_index < 0:
        return 0, 0.0, 0.0, -1
    return 1, failure_depth, failure_extreme, turn_back_index


@njit(cache=True)
def scan_trigger_bar_numba(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    start_index: int,
    direction: int,
    trigger_level: float,
    session_day: np.int64,
    day_norm: np.ndarray,
    hour: np.ndarray,
    london_mask: np.ndarray,
    n_bars: int,
) -> int:
    for j in range(start_index, n_bars):
        if day_norm[j] != session_day:
            break
        if not _bar_in_session_numba(day_norm, hour, j, session_day, london_mask):
            continue
        if direction == _DIR_BUY:
            if close[j] >= trigger_level or high[j] >= trigger_level:
                return j
        elif close[j] <= trigger_level or low[j] <= trigger_level:
            return j
    return -1


def direction_code(direction: str) -> int:
    return _DIR_BUY if direction == "BUY" else _DIR_SELL
