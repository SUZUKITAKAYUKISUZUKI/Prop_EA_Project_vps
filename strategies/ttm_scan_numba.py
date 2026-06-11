"""TTM scan Numba kernels — LOW_UPDATE x SHORT only (flow features, no Kalman)."""

from __future__ import annotations

import numpy as np

from strategies.scan_numba_util import njit, scan_numba_active

EVENT_START_MIN = 8 * 60 + 30
EVENT_END_MIN = 10 * 60 + 30
TTM_MINUTE_OF_DAY = 9 * 60 + 55
ASIA_END_MIN = EVENT_START_MIN


def ttm_scan_numba_enabled() -> bool:
    return scan_numba_active("TTM_SCAN_NUMBA", default=True)


def ttm_scan_numba_active() -> bool:
    return ttm_scan_numba_enabled()


@njit(cache=True)
def build_asia_extremes_numba(
    day_ids: np.ndarray,
    minute_of_day: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    max_day = 0
    for i in range(n):
        if day_ids[i] > max_day:
            max_day = int(day_ids[i])
    size = max_day + 1
    asia_h = np.full(size, -1.0e18, dtype=np.float64)
    asia_l = np.full(size, 1.0e18, dtype=np.float64)
    for i in range(n):
        if minute_of_day[i] < ASIA_END_MIN:
            d = int(day_ids[i])
            if high[i] > asia_h[d]:
                asia_h[d] = high[i]
            if low[i] < asia_l[d]:
                asia_l[d] = low[i]
    return asia_h, asia_l


@njit(cache=True)
def scan_ttm_events_numba(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    datetime_ns: np.ndarray,
    minute_of_day: np.ndarray,
    day_ids: np.ndarray,
    m5_atr_on_m1: np.ndarray,
    m15_atr_on_m1: np.ndarray,
    h1_atr_on_m1: np.ndarray,
    asia_high_by_day: np.ndarray,
    asia_low_by_day: np.ndarray,
    event_indices: np.ndarray,
    pip: float,
    max_events_per_day: int,
    bar_minutes: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    n_events = len(event_indices)
    max_out = n_events
    out_sim_idx = np.empty(max_out, dtype=np.int64)
    out_full_idx = np.empty(max_out, dtype=np.int64)
    out_dt_ns = np.empty(max_out, dtype=np.int64)
    out_dir = np.empty(max_out, dtype=np.int8)
    out_entry = np.empty(max_out, dtype=np.float64)
    out_sl = np.empty(max_out, dtype=np.float64)
    out_tp = np.empty(max_out, dtype=np.float64)
    out_trigger = np.empty(max_out, dtype=np.int8)
    out_pattern = np.empty(max_out, dtype=np.int8)
    out_mins_to_ttm = np.empty(max_out, dtype=np.float64)
    out_weekday = np.empty(max_out, dtype=np.int8)
    out_atr_m5 = np.empty(max_out, dtype=np.float64)
    out_atr_m15 = np.empty(max_out, dtype=np.float64)
    out_atr_h1 = np.empty(max_out, dtype=np.float64)
    out_tokyo_range = np.empty(max_out, dtype=np.float64)
    out_dist_al = np.empty(max_out, dtype=np.float64)
    out_asia_pct = np.empty(max_out, dtype=np.float64)
    out_pre_ret = np.empty(max_out, dtype=np.float64)
    out_pre_vel = np.empty(max_out, dtype=np.float64)
    out_pre_atr = np.empty(max_out, dtype=np.float64)
    out_low_break_dist = np.empty(max_out, dtype=np.float64)
    out_low_break_vel = np.empty(max_out, dtype=np.float64)

    max_day = 0
    for ei in range(n_events):
        i = int(event_indices[ei])
        if day_ids[i] > max_day:
            max_day = int(day_ids[i])
    day_cap = max_day + 1
    events_today = np.zeros(day_cap, dtype=np.int32)

    last_day = -1
    asia_high_broken = False
    asia_low_broken = False
    window_first_open = 0.0
    count = 0
    sim_idx = 0

    for ei in range(n_events):
        i = int(event_indices[ei])
        d = int(day_ids[i])
        minute = int(minute_of_day[i])

        if d != last_day:
            last_day = d
            asia_high_broken = False
            asia_low_broken = False
            window_first_open = open_[i]

        if events_today[d] >= max_events_per_day:
            sim_idx += 1
            continue

        asia_h = asia_high_by_day[d]
        asia_l = asia_low_by_day[d]
        if asia_h < -1.0e17:
            asia_h = high[i]
        if asia_l > 1.0e17:
            asia_l = low[i]

        pre_return = (close[i] - window_first_open) / pip
        pre_velocity = (close[i] - close[i - 1]) / pip if i > 0 else 0.0

        trigger = 0
        if high[i] > asia_h and not asia_high_broken:
            asia_high_broken = True
        elif low[i] < asia_l and not asia_low_broken:
            trigger = 2
            asia_low_broken = True

        if trigger != 2:
            sim_idx += 1
            continue

        events_today[d] += 1

        atr_m5 = m5_atr_on_m1[i]
        atr_m15 = m15_atr_on_m1[i]
        atr_h1 = h1_atr_on_m1[i]
        tokyo_range = (asia_h - asia_l) / pip if asia_h > asia_l else 0.0
        dist_al = (close[i] - asia_l) / pip
        asia_pct = (
            (close[i] - asia_l) / (asia_h - asia_l) * 100.0 if asia_h > asia_l else 50.0
        )
        pre_atr_ratio = abs(pre_return) * pip / atr_m15 if atr_m15 > 1.0e-12 else 0.0
        low_break_dist = (asia_l - low[i]) / pip
        low_break_vel = pre_velocity

        sl_dist = max(atr_m15 * 1.2, pip * 8.0)
        sl = close[i] + sl_dist
        tp = close[i] - sl_dist * 1.5

        jst_sec = datetime_ns[i] // 1_000_000_000 + 9 * 3600
        weekday = int((jst_sec // 86400 + 3) % 7)

        out_sim_idx[count] = sim_idx
        out_full_idx[count] = i
        out_dt_ns[count] = datetime_ns[i]
        out_dir[count] = -1
        out_entry[count] = close[i]
        out_sl[count] = sl
        out_tp[count] = tp
        out_trigger[count] = trigger
        out_pattern[count] = 5
        out_mins_to_ttm[count] = float(TTM_MINUTE_OF_DAY - minute)
        out_weekday[count] = weekday
        out_atr_m5[count] = atr_m5 / pip if atr_m5 > 0.0 else 0.0
        out_atr_m15[count] = atr_m15 / pip if atr_m15 > 0.0 else 0.0
        out_atr_h1[count] = atr_h1 / pip if atr_h1 > 0.0 else 0.0
        out_tokyo_range[count] = tokyo_range
        out_dist_al[count] = dist_al
        out_asia_pct[count] = asia_pct
        out_pre_ret[count] = pre_return
        out_pre_vel[count] = pre_velocity
        out_pre_atr[count] = pre_atr_ratio
        out_low_break_dist[count] = low_break_dist
        out_low_break_vel[count] = low_break_vel
        count += 1
        sim_idx += 1

    return (
        out_sim_idx,
        out_full_idx,
        out_dt_ns,
        out_dir,
        out_entry,
        out_sl,
        out_tp,
        out_trigger,
        out_pattern,
        out_mins_to_ttm,
        out_weekday,
        out_atr_m5,
        out_atr_m15,
        out_atr_h1,
        out_tokyo_range,
        out_dist_al,
        out_asia_pct,
        out_pre_ret,
        out_pre_vel,
        out_pre_atr,
        out_low_break_dist,
        out_low_break_vel,
        count,
    )
