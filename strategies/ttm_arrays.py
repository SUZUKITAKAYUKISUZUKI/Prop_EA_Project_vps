"""TTM scan arrays — numpy-only views (no pandas in hot path)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from strategies.bt_ohlcv import BtOhlcvFrame, resample_to_h1, take_arrays
from strategies.archive.cspa_arrays import OhlcvArrays, compute_atr_np

JST_OFFSET_SEC = 9 * 3600
EVENT_START_MIN = 8 * 60 + 30
EVENT_END_MIN = 10 * 60 + 30
TTM_MINUTE_OF_DAY = 9 * 60 + 55


def jst_minute_of_day_ns(dt_ns: np.ndarray) -> np.ndarray:
    sec = dt_ns // 1_000_000_000
    jst_sec = sec + JST_OFFSET_SEC
    return ((jst_sec % 86400) // 60).astype(np.int32)


def jst_day_id_ns(dt_ns: np.ndarray) -> np.ndarray:
    sec = dt_ns // 1_000_000_000
    jst_sec = sec + JST_OFFSET_SEC
    return (jst_sec // 86400).astype(np.int64)


def event_window_mask(minute_of_day: np.ndarray) -> np.ndarray:
    return (minute_of_day >= EVENT_START_MIN) & (minute_of_day <= EVENT_END_MIN)


def filter_ttm_event_window(frame: BtOhlcvFrame) -> BtOhlcvFrame:
    """Keep M1 bars in 08:30–10:30 JST (BT sim / exec track)."""
    arr = frame.arrays
    if arr.length == 0:
        return BtOhlcvFrame.make_empty()
    minute = jst_minute_of_day_ns(arr.datetime_ns)
    mask = event_window_mask(minute)
    if not mask.any():
        return BtOhlcvFrame.make_empty()
    return BtOhlcvFrame(take_arrays(arr, mask))


def align_htf_atr_to_ltf(
    ltf: OhlcvArrays,
    htf: OhlcvArrays,
    htf_atr: np.ndarray,
    *,
    period: int,
) -> np.ndarray:
    idx = np.searchsorted(htf.datetime_ns, ltf.datetime_ns, side="right") - 1
    out = np.zeros(ltf.length, dtype=np.float64)
    valid = idx >= period - 1
    out[valid] = htf_atr[idx[valid]]
    return out


@dataclass(slots=True)
class TtmScanArrays:
    m1: OhlcvArrays
    m5: OhlcvArrays
    m15: OhlcvArrays
    m1_minute_jst: np.ndarray
    m1_day_jst: np.ndarray
    m5_atr_on_m1: np.ndarray
    m15_atr_on_m1: np.ndarray
    h1_atr_on_m1: np.ndarray
    asia_high_by_day: np.ndarray
    asia_low_by_day: np.ndarray
    event_indices: np.ndarray
    event_sim_index: np.ndarray


def build_ttm_scan_arrays(m1: BtOhlcvFrame, m5: BtOhlcvFrame, m15: BtOhlcvFrame) -> TtmScanArrays:
    m1_a = m1.arrays
    m5_a = m5.arrays
    m15_a = m15.arrays
    minute = jst_minute_of_day_ns(m1_a.datetime_ns)
    day_id = jst_day_id_ns(m1_a.datetime_ns)

    from strategies.ttm_scan_numba import build_asia_extremes_numba

    asia_high_by_day, asia_low_by_day = build_asia_extremes_numba(
        day_id,
        minute,
        m1_a.high,
        m1_a.low,
        m1_a.length,
    )

    m5_atr = compute_atr_np(m5_a.high, m5_a.low, m5_a.close, 14)
    m15_atr = compute_atr_np(m15_a.high, m15_a.low, m15_a.close, 14)
    h1_frame = resample_to_h1(m1)
    h1_a = h1_frame.arrays
    h1_atr = compute_atr_np(h1_a.high, h1_a.low, h1_a.close, 14)
    m5_on_m1 = align_htf_atr_to_ltf(m1_a, m5_a, m5_atr, period=14)
    m15_on_m1 = align_htf_atr_to_ltf(m1_a, m15_a, m15_atr, period=14)
    h1_on_m1 = align_htf_atr_to_ltf(m1_a, h1_a, h1_atr, period=14)

    event_mask = event_window_mask(minute)
    event_indices = np.nonzero(event_mask)[0].astype(np.int64)
    event_sim_index = np.arange(event_indices.shape[0], dtype=np.int64)

    return TtmScanArrays(
        m1=m1_a,
        m5=m5_a,
        m15=m15_a,
        m1_minute_jst=minute,
        m1_day_jst=day_id,
        m5_atr_on_m1=m5_on_m1,
        m15_atr_on_m1=m15_on_m1,
        h1_atr_on_m1=h1_on_m1,
        asia_high_by_day=asia_high_by_day,
        asia_low_by_day=asia_low_by_day,
        event_indices=event_indices,
        event_sim_index=event_sim_index,
    )
