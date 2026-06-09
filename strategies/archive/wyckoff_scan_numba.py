"""Numba kernels for Wyckoff Reversal (WR) precompute (``WR_SCAN_NUMBA=1``).

ARCHIVED: WR 本体と同様 — LGR 構築に向けた発展的廃止に伴い移動。
"""

from __future__ import annotations

import numpy as np

from strategies.scan_numba_util import njit, scan_numba_active

# packed try_build result: valid flag + 11 floats
_ACC_FIELDS = 11


def wr_scan_numba_enabled() -> bool:
    return scan_numba_active("WR_SCAN_NUMBA", default=False)


def wr_scan_numba_active() -> bool:
    return wr_scan_numba_enabled()


@njit(cache=True)
def _rolling_volume_zscore_numba(volume: np.ndarray, bar_index: int, lookback: int) -> float:
    start = max(0, bar_index - lookback)
    window = volume[start:bar_index]
    if len(window) < 3:
        return 0.0
    val = volume[bar_index]
    mean = 0.0
    for i in range(len(window)):
        mean += window[i]
    mean /= len(window)
    var = 0.0
    for i in range(len(window)):
        d = window[i] - mean
        var += d * d
    std = (var / len(window)) ** 0.5
    if std <= 0.0:
        return 0.0
    return (val - mean) / std


@njit(cache=True)
def _in_downtrend_context_numba(
    low: np.ndarray,
    close: np.ndarray,
    h1_atr: np.ndarray,
    sc_idx: int,
    lookback: int,
) -> bool:
    start = max(0, sc_idx - lookback)
    if sc_idx - start < 10:
        return False
    sc_low = low[sc_idx]
    window_low = low[start]
    for i in range(start + 1, sc_idx):
        if low[i] < window_low:
            window_low = low[i]
    atr = h1_atr[sc_idx]
    if atr <= 0.0:
        return False
    near_low = sc_low <= window_low + 0.15 * atr
    n = sc_idx - start
    x_mean = (n - 1) * 0.5
    y_mean = 0.0
    for i in range(start, sc_idx):
        y_mean += close[i]
    y_mean /= n
    num = 0.0
    den = 0.0
    for k in range(n):
        x = k - x_mean
        y = close[start + k] - y_mean
        num += x * y
        den += x * x
    slope = num / den if den > 0.0 else 0.0
    return slope < 0.0 and near_low


@njit(cache=True)
def _count_support_tests_numba(
    low: np.ndarray,
    high: np.ndarray,
    ar_idx: int,
    end_idx: int,
    band_low: float,
    band_high: float,
) -> tuple:
    count = 0
    st_price = high[ar_idx]
    band_width = band_high - band_low
    for i in range(ar_idx + 1, end_idx + 1):
        lv = low[i]
        if lv <= band_high and lv >= band_low - band_width:
            count += 1
            if lv < st_price:
                st_price = lv
    return count, st_price


@njit(cache=True)
def _detect_phase_b_ut_numba(
    high: np.ndarray,
    close: np.ndarray,
    ar_idx: int,
    end_idx: int,
    resistance_level: float,
) -> bool:
    for i in range(ar_idx + 1, end_idx + 1):
        if high[i] > resistance_level and close[i] < resistance_level:
            return True
    return False


@njit(cache=True)
def _find_ps_price_numba(low: np.ndarray, sc_idx: int) -> float:
    start = max(0, sc_idx - 40)
    sc_low = low[sc_idx]
    if start >= sc_idx:
        return sc_low * 1.001
    ps_ref = low[start]
    for i in range(start + 1, sc_idx):
        if low[i] < ps_ref:
            ps_ref = low[i]
    return max(ps_ref, sc_low * 1.001)


@njit(cache=True)
def try_build_accumulation_numba(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    h1_atr: np.ndarray,
    sc_idx: int,
    as_of_idx: int,
    atr_period: int,
    min_bars_after_sc: int,
    sc_body_atr_mult: float,
    sc_volume_zscore_min: float,
    sc_downtrend_lookback: int,
    ar_min_rally_atr: float,
    support_band_atr: float,
    vol_zscore_lookback: int,
) -> np.ndarray:
    out = np.zeros(_ACC_FIELDS + 1, dtype=np.float64)
    if sc_idx >= as_of_idx - min_bars_after_sc or sc_idx < atr_period:
        return out

    sc_atr = h1_atr[sc_idx]
    if sc_atr <= 0.0:
        return out

    body = abs(close[sc_idx] - open_[sc_idx])
    if close[sc_idx] >= open_[sc_idx] or body < sc_body_atr_mult * sc_atr:
        return out

    vol_z = _rolling_volume_zscore_numba(volume, sc_idx, vol_zscore_lookback)
    if vol_z < sc_volume_zscore_min:
        return out
    if not _in_downtrend_context_numba(low, close, h1_atr, sc_idx, sc_downtrend_lookback):
        return out

    sc_price = low[sc_idx]
    ar_price = sc_price
    ar_idx = sc_idx
    found_ar = False
    for j in range(sc_idx + 1, as_of_idx + 1):
        if high[j] > ar_price:
            ar_price = high[j]
        rally = ar_price - sc_price
        if rally >= ar_min_rally_atr * sc_atr:
            ar_idx = j
            found_ar = True
            break
    if not found_ar:
        return out

    half = support_band_atr * sc_atr
    band_low = sc_price - half
    band_high = sc_price + half
    support_level = sc_price
    resistance_level = ar_price
    test_count, st_price = _count_support_tests_numba(
        low, high, ar_idx, as_of_idx, band_low, band_high
    )
    if test_count < 2:
        return out

    ps_price = _find_ps_price_numba(low, sc_idx)
    range_width = resistance_level - support_level
    if range_width < 0.0:
        range_width = 0.0
    range_width_atr = range_width / sc_atr if sc_atr > 0.0 else 0.0
    phase_b_ut = 1.0 if _detect_phase_b_ut_numba(high, close, ar_idx, as_of_idx, resistance_level) else 0.0

    out[0] = 1.0
    out[1] = sc_price
    out[2] = ar_price
    out[3] = st_price
    out[4] = support_level
    out[5] = resistance_level
    out[6] = range_width_atr
    out[7] = float(test_count)
    out[8] = phase_b_ut
    out[9] = float(ar_idx)
    out[10] = ps_price
    out[11] = float(sc_idx)
    return out


@njit(cache=True)
def find_accumulation_for_asof_numba(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    h1_atr: np.ndarray,
    as_of_idx: int,
    start_idx: int,
    atr_period: int,
    min_bars_after_sc: int,
    sc_body_atr_mult: float,
    sc_volume_zscore_min: float,
    sc_downtrend_lookback: int,
    ar_min_rally_atr: float,
    support_band_atr: float,
    vol_zscore_lookback: int,
) -> np.ndarray:
    empty = np.zeros(_ACC_FIELDS + 1, dtype=np.float64)
    for sc_idx in range(as_of_idx - min_bars_after_sc, start_idx - 1, -1):
        acc = try_build_accumulation_numba(
            open_,
            high,
            low,
            close,
            volume,
            h1_atr,
            sc_idx,
            as_of_idx,
            atr_period,
            min_bars_after_sc,
            sc_body_atr_mult,
            sc_volume_zscore_min,
            sc_downtrend_lookback,
            ar_min_rally_atr,
            support_band_atr,
            vol_zscore_lookback,
        )
        if acc[0] > 0.0:
            return acc
    return empty


def accumulation_from_packed(
    packed: np.ndarray,
    arr,
    as_of_idx: int,
):
    from strategies.bt_ohlcv import ts_ns_to_pd
    from strategies.archive.wyckoff_reversal import AccumulationRange

    if packed[0] <= 0.0:
        return None

    sc_idx = int(packed[11])
    ar_idx = int(packed[9])
    sc_price = float(packed[1])
    ar_price = float(packed[2])
    st_price = float(packed[3])
    sc_ts = ts_ns_to_pd(int(arr.datetime_ns[sc_idx]))
    ar_ts = ts_ns_to_pd(int(arr.datetime_ns[ar_idx]))
    as_of_ts = ts_ns_to_pd(int(arr.datetime_ns[as_of_idx]))
    days_in_acc = max(0, (as_of_ts.normalize() - sc_ts.normalize()).days)
    phase_b_duration = max(0, as_of_idx - ar_idx)

    return AccumulationRange(
        ps_price=round(float(packed[10]), 6),
        sc_price=round(sc_price, 6),
        ar_price=round(ar_price, 6),
        st_price=round(st_price, 6),
        support_level=round(float(packed[4]), 6),
        resistance_level=round(float(packed[5]), 6),
        range_width_atr=round(float(packed[6]), 4),
        test_count=int(packed[7]),
        phase_b_ut_occurred=bool(packed[8]),
        days_in_accumulation=days_in_acc,
        phase_b_duration=phase_b_duration,
        is_valid=True,
        sc_bar_index=sc_idx,
        ar_bar_index=ar_idx,
        sc_timestamp=sc_ts,
        ar_timestamp=ar_ts,
    )
