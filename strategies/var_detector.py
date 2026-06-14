"""
strategies/var_detector.py — Volume Area Reversal (VAR) Numba kernels.

Hot paths are numpy/numba only. Pandas must not be used inside scan/sim loops.
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np

from strategies.scan_numba_util import njit

ALLOWED_PAIRS = frozenset({"AUDNZD", "EURGBP", "USDCAD"})
EXEC_BAR_MINUTES = 60
VP_BAR_MINUTES = 5
STRUCTURE_BAR_MINUTES = 240
ATR_PERIOD = 14
ENTROPY_BINS = 10
BB_PERIOD_SHORT = 20
BB_PERIOD_LONG = 50
BB_STD_MULT = 2.0
HV_PERIOD = 20
SESSION_START_HOUR_UTC = 7
VP_TOUCH_ATR_MULT = 0.15
VP_BREAKOUT_ATR_MULT = 0.35
SL_ATR_MULT = 0.5
MIN_RR_RATIO = 1.0
MIN_RISK_PIPS = 3.0
MAX_SIM_RESULT_R = 50.0
WARMUP_BARS = 120
VALUE_AREA_FRACTION = 0.70
TIME_EXIT_MINUTES = (30, 60, 120, 180)
VAR_PURE_DATA_MODE = True

PA_PIN_BAR = 1
PA_ENGULFING = 2
PA_INSIDE_BAR = 4
PA_CLOSE_ONLY = 8

Direction = Literal["BUY", "SELL"]

EXIT_SL = 1
EXIT_TP_POC = 2
EXIT_TP_VA = 3
EXIT_TIME_30 = 4
EXIT_TIME_60 = 5
EXIT_TIME_120 = 6
EXIT_TIME_180 = 7
EXIT_OPEN = 8

EXIT_REASON_LABELS = {
    EXIT_SL: "SL_HIT",
    EXIT_TP_POC: "TP_POC",
    EXIT_TP_VA: "TP_VA",
    EXIT_TIME_30: "TIME_EXIT_30M",
    EXIT_TIME_60: "TIME_EXIT_60M",
    EXIT_TIME_120: "TIME_EXIT_120M",
    EXIT_TIME_180: "TIME_EXIT_180M",
    EXIT_OPEN: "OPEN",
}

PA_LABELS = {
    PA_PIN_BAR: "PIN_BAR",
    PA_ENGULFING: "ENGULFING",
    PA_INSIDE_BAR: "INSIDE_BAR",
    PA_CLOSE_ONLY: "CLOSE_ONLY",
}


def is_var_pure_data_mode() -> bool:
    raw = os.getenv("VAR_PURE_DATA_MODE", "1" if VAR_PURE_DATA_MODE else "0")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def is_var_enabled() -> bool:
    raw = os.getenv("VAR_ENABLED", "1")
    return raw.strip().lower() in ("1", "true", "yes", "on")


@njit(cache=True)
def percentile_rank(window: np.ndarray, current: float) -> float:
    if window.size == 0 or not np.isfinite(current):
        return 50.0
    count = 0
    valid = 0
    for i in range(window.size):
        v = window[i]
        if not np.isfinite(v):
            continue
        valid += 1
        if v <= current:
            count += 1
    if valid == 0:
        return 50.0
    return float(count) / float(valid) * 100.0


@njit(cache=True)
def compute_atr_series(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = close.shape[0]
    atr = np.full(n, np.nan)
    if n == 0:
        return atr
    prev_close = np.empty(n)
    prev_close[0] = close[0]
    for i in range(1, n):
        prev_close[i] = close[i - 1]
    tr = np.empty(n)
    for i in range(n):
        hl = high[i] - low[i]
        hc = abs(high[i] - prev_close[i])
        lc = abs(low[i] - prev_close[i])
        tr[i] = max(hl, max(hc, lc))
    if n < period:
        return atr
    run = 0.0
    for i in range(period):
        run += tr[i]
    atr[period - 1] = run / period
    for i in range(period, n):
        run += tr[i] - tr[i - period]
        atr[i] = run / period
    return atr


@njit(cache=True)
def compute_bb_width_series(
    close: np.ndarray,
    period: int,
    std_mult: float,
    pip_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = close.shape[0]
    width_pips = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    if n < period or pip_size <= 0.0:
        return width_pips, upper, lower
    for i in range(period - 1, n):
        start = i - period + 1
        mean = 0.0
        for j in range(start, i + 1):
            mean += close[j]
        mean /= period
        var = 0.0
        for j in range(start, i + 1):
            diff = close[j] - mean
            var += diff * diff
        var /= period
        std = var ** 0.5
        up = mean + std_mult * std
        lo = mean - std_mult * std
        upper[i] = up
        lower[i] = lo
        width_pips[i] = (up - lo) / pip_size
    return width_pips, upper, lower


@njit(cache=True)
def compute_hv_series(close: np.ndarray, period: int) -> np.ndarray:
    n = close.shape[0]
    hv = np.full(n, np.nan)
    if n <= period:
        return hv
    for i in range(period, n):
        rets = np.empty(period)
        for j in range(period):
            prev = close[i - period + j]
            cur = close[i - period + j + 1]
            if prev <= 0.0:
                rets[j] = 0.0
            else:
                rets[j] = (cur - prev) / prev
        mean = 0.0
        for j in range(period):
            mean += rets[j]
        mean /= period
        var = 0.0
        for j in range(period):
            diff = rets[j] - mean
            var += diff * diff
        var /= max(period - 1, 1)
        hv[i] = var ** 0.5
    return hv


@njit(cache=True)
def shannon_entropy_returns(close: np.ndarray, end_idx: int, window: int, n_bins: int) -> float:
    if end_idx < 1 or window < 2:
        return np.nan
    start = max(1, end_idx - window + 1)
    count = end_idx - start + 1
    if count < 2:
        return np.nan
    rets = np.empty(count - 1)
    min_r = 1e18
    max_r = -1e18
    for i in range(count - 1):
        idx = start + i
        prev = close[idx - 1]
        cur = close[idx]
        if prev <= 0.0:
            r = 0.0
        else:
            r = (cur - prev) / prev
        rets[i] = r
        if r < min_r:
            min_r = r
        if r > max_r:
            max_r = r
    span = max_r - min_r
    if span <= 1e-12:
        return 0.0
    hist = np.zeros(n_bins)
    for i in range(rets.shape[0]):
        pos = int((rets[i] - min_r) / span * (n_bins - 1))
        if pos < 0:
            pos = 0
        if pos >= n_bins:
            pos = n_bins - 1
        hist[pos] += 1.0
    total = rets.shape[0]
    ent = 0.0
    for i in range(n_bins):
        if hist[i] <= 0.0:
            continue
        p = hist[i] / total
        ent -= p * np.log2(p)
    return ent


@njit(cache=True)
def hurst_exponent(close: np.ndarray, end_idx: int, window: int) -> float:
    if end_idx < window or window < 20:
        return 0.5
    start = end_idx - window + 1
    mean = 0.0
    for i in range(start + 1, end_idx + 1):
        prev = close[i - 1]
        cur = close[i]
        if prev <= 0.0:
            continue
        mean += (cur - prev) / prev
    n = window - 1
    if n <= 1:
        return 0.5
    mean /= n
    cum = 0.0
    min_cum = 0.0
    max_cum = 0.0
    var = 0.0
    for i in range(start + 1, end_idx + 1):
        prev = close[i - 1]
        cur = close[i]
        if prev <= 0.0:
            r = 0.0
        else:
            r = (cur - prev) / prev - mean
        cum += r
        if cum < min_cum:
            min_cum = cum
        if cum > max_cum:
            max_cum = cum
        var += r * r
    if var <= 1e-18:
        return 0.5
    rs = (max_cum - min_cum) / (var ** 0.5)
    if rs <= 0.0:
        return 0.5
    return float(np.log(rs) / np.log(float(window)))


@njit(cache=True)
def session_start_ns(ts_ns: int) -> int:
    ns_per_day = np.int64(86_400_000_000_000)
    ns_per_hour = np.int64(3_600_000_000_000)
    day_start = (np.int64(ts_ns) // ns_per_day) * ns_per_day
    start = day_start + np.int64(SESSION_START_HOUR_UTC) * ns_per_hour
    if np.int64(ts_ns) < start:
        start -= ns_per_day
    return int(start)


@njit(cache=True)
def calculate_vp_profile_np(
    close: np.ndarray,
    volume: np.ndarray,
    bin_step: float,
) -> tuple[float, float, float]:
    if close.shape[0] == 0 or bin_step <= 0.0:
        return np.nan, np.nan, np.nan
    n = close.shape[0]
    vol = volume if volume.shape[0] == n else np.ones(n)
    bins = np.round(close / bin_step) * bin_step
    uniq = np.unique(bins)
    vol_sum = np.empty(uniq.shape[0])
    for i in range(uniq.shape[0]):
        total = 0.0
        for j in range(n):
            if abs(bins[j] - uniq[i]) < bin_step * 0.01:
                v = vol[j]
                if v < 0.0:
                    v = 0.0
                total += v
        vol_sum[i] = total
    if vol_sum.shape[0] == 0:
        return np.nan, np.nan, np.nan
    total = 0.0
    for i in range(vol_sum.shape[0]):
        total += vol_sum[i]
    if total <= 0.0:
        return np.nan, np.nan, np.nan
    poc_idx = 0
    poc_vol = vol_sum[0]
    for i in range(1, vol_sum.shape[0]):
        if vol_sum[i] > poc_vol:
            poc_vol = vol_sum[i]
            poc_idx = i
    target = total * VALUE_AREA_FRACTION
    lower_idx = poc_idx
    upper_idx = poc_idx
    current = vol_sum[poc_idx]
    while current < target:
        has_lower = lower_idx > 0
        has_upper = upper_idx < vol_sum.shape[0] - 1
        if not has_lower and not has_upper:
            break
        v_lower = vol_sum[lower_idx - 1] if has_lower else -1.0
        v_upper = vol_sum[upper_idx + 1] if has_upper else -1.0
        if v_lower >= v_upper:
            lower_idx -= 1
            current += v_lower
        else:
            upper_idx += 1
            current += v_upper
    return float(uniq[upper_idx]), float(uniq[lower_idx]), float(uniq[poc_idx])


@njit(cache=True)
def detect_pa_flags(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    idx: int,
    direction_code: int,
) -> int:
    if idx <= 0:
        return PA_CLOSE_ONLY
    o = open_[idx]
    h = high[idx]
    l = low[idx]
    c = close[idx]
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    flags = PA_CLOSE_ONLY
    if direction_code < 0 and body > 0.0 and upper_wick >= body * 2.0:
        flags |= PA_PIN_BAR
    if direction_code > 0 and body > 0.0 and lower_wick >= body * 2.0:
        flags |= PA_PIN_BAR
    po = open_[idx - 1]
    ph = high[idx - 1]
    pl = low[idx - 1]
    pc = close[idx - 1]
    prev_body_high = max(po, pc)
    prev_body_low = min(po, pc)
    cur_body_high = max(o, c)
    cur_body_low = min(o, c)
    if cur_body_high >= prev_body_high and cur_body_low <= prev_body_low:
        flags |= PA_ENGULFING
    if h <= ph and l >= pl:
        flags |= PA_INSIDE_BAR
    return flags


@njit(cache=True)
def vp_touch_direction(
    high: float,
    low: float,
    close: float,
    vah: float,
    val: float,
    atr: float,
    pip_size: float,
) -> int:
    if not np.isfinite(vah) or not np.isfinite(val) or atr <= 0.0:
        return 0
    touch_buf = max(VP_TOUCH_ATR_MULT * atr, pip_size * 2.0)
    break_buf = max(VP_BREAKOUT_ATR_MULT * atr, pip_size * 3.0)
    if high >= vah - touch_buf and close <= vah + break_buf:
        return -1
    if low <= val + touch_buf and close >= val - break_buf:
        return 1
    return 0


@njit(cache=True)
def resolve_take_profit(
    direction_code: int,
    entry: float,
    stop: float,
    poc: float,
    vah: float,
    val: float,
) -> tuple[float, int]:
    risk = abs(entry - stop)
    if risk <= 0.0:
        return entry, EXIT_OPEN
    if direction_code < 0:
        tp_poc = poc
        reward = entry - tp_poc
        if reward / risk >= MIN_RR_RATIO and tp_poc < entry:
            return tp_poc, EXIT_TP_POC
        tp_va = val
        reward = entry - tp_va
        if reward / risk >= MIN_RR_RATIO and tp_va < entry:
            return tp_va, EXIT_TP_VA
        return tp_va, EXIT_TP_VA
    tp_poc = poc
    reward = tp_poc - entry
    if reward / risk >= MIN_RR_RATIO and tp_poc > entry:
        return tp_poc, EXIT_TP_POC
    tp_va = vah
    reward = tp_va - entry
    if reward / risk >= MIN_RR_RATIO and tp_va > entry:
        return tp_va, EXIT_TP_VA
    return tp_va, EXIT_TP_VA


@njit(cache=True)
def detect_volatility_expansion(
    atr: np.ndarray,
    bb_width: np.ndarray,
    end_idx: int,
    lookback: int,
) -> bool:
    if end_idx < lookback + 2:
        return False
    cur_atr = atr[end_idx]
    prev_atr = atr[end_idx - 5] if end_idx >= 5 else atr[end_idx - 1]
    cur_bb = bb_width[end_idx]
    prev_bb = bb_width[end_idx - 5] if end_idx >= 5 else bb_width[end_idx - 1]
    if not np.isfinite(cur_atr) or not np.isfinite(prev_atr):
        return False
    if cur_atr > prev_atr * 1.35:
        return True
    if np.isfinite(cur_bb) and np.isfinite(prev_bb) and prev_bb > 0.0:
        if cur_bb > prev_bb * 1.35:
            return True
    window = atr[max(0, end_idx - lookback + 1) : end_idx + 1]
    pct = percentile_rank(window, cur_atr)
    return pct >= 75.0


@njit(cache=True)
def _clamp_result_r(result_r: float) -> float:
    if result_r > MAX_SIM_RESULT_R:
        return MAX_SIM_RESULT_R
    if result_r < -MAX_SIM_RESULT_R:
        return -MAX_SIM_RESULT_R
    return result_r


@njit(cache=True)
def simulate_trade_forward(
    entry_ts_ns: int,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    tp_kind: int,
    direction_code: int,
    dt_ns: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    start_idx: int,
) -> tuple[int, float, int, int, float]:
    n = dt_ns.shape[0]
    risk = abs(entry_price - stop_loss)
    if risk <= 1e-9:
        return EXIT_OPEN, entry_price, 0, 0, 0.0
    time_deadlines = (
        entry_ts_ns + np.int64(30 * 60 * 1_000_000_000),
        entry_ts_ns + np.int64(60 * 60 * 1_000_000_000),
        entry_ts_ns + np.int64(120 * 60 * 1_000_000_000),
        entry_ts_ns + np.int64(180 * 60 * 1_000_000_000),
    )
    time_codes = (EXIT_TIME_30, EXIT_TIME_60, EXIT_TIME_120, EXIT_TIME_180)
    next_time = 0
    for j in range(start_idx, n):
        ts = dt_ns[j]
        hi = high[j]
        lo = low[j]
        cl = close[j]
        if direction_code < 0:
            if hi >= stop_loss:
                mins = int(max((ts - entry_ts_ns) / 1_000_000_000 / 60.0, 0.0))
                bars = max(mins // EXEC_BAR_MINUTES, 0)
                return EXIT_SL, stop_loss, mins, bars, -1.0
            if lo <= take_profit:
                mins = int(max((ts - entry_ts_ns) / 1_000_000_000 / 60.0, 0.0))
                bars = max(mins // EXEC_BAR_MINUTES, 0)
                reward = entry_price - take_profit
                return tp_kind, take_profit, mins, bars, _clamp_result_r(reward / risk)
        else:
            if lo <= stop_loss:
                mins = int(max((ts - entry_ts_ns) / 1_000_000_000 / 60.0, 0.0))
                bars = max(mins // EXEC_BAR_MINUTES, 0)
                return EXIT_SL, stop_loss, mins, bars, -1.0
            if hi >= take_profit:
                mins = int(max((ts - entry_ts_ns) / 1_000_000_000 / 60.0, 0.0))
                bars = max(mins // EXEC_BAR_MINUTES, 0)
                reward = take_profit - entry_price
                return tp_kind, take_profit, mins, bars, _clamp_result_r(reward / risk)
        while next_time < 4 and ts >= time_deadlines[next_time]:
            exit_px = cl
            mins = int(max((ts - entry_ts_ns) / 1_000_000_000 / 60.0, 0.0))
            bars = max(mins // EXEC_BAR_MINUTES, 0)
            if direction_code < 0:
                result_r = _clamp_result_r((entry_price - exit_px) / risk)
            else:
                result_r = _clamp_result_r((exit_px - entry_price) / risk)
            code = time_codes[next_time]
            next_time += 1
            return code, exit_px, mins, bars, result_r
    if n > start_idx:
        exit_px = close[n - 1]
        mins = int(max((dt_ns[n - 1] - entry_ts_ns) / 1_000_000_000 / 60.0, 0.0))
        bars = max(mins // EXEC_BAR_MINUTES, 0)
        if direction_code < 0:
            result_r = _clamp_result_r((entry_price - exit_px) / risk)
        else:
            result_r = _clamp_result_r((exit_px - entry_price) / risk)
        return EXIT_TIME_180, exit_px, mins, bars, result_r
    return EXIT_OPEN, entry_price, 0, 0, 0.0


@njit(cache=True)
def scan_var_events_numba(
    exec_dt_ns: np.ndarray,
    exec_open: np.ndarray,
    exec_high: np.ndarray,
    exec_low: np.ndarray,
    exec_close: np.ndarray,
    exec_volume: np.ndarray,
    vp_dt_ns: np.ndarray,
    vp_open: np.ndarray,
    vp_high: np.ndarray,
    vp_low: np.ndarray,
    vp_close: np.ndarray,
    vp_volume: np.ndarray,
    atr: np.ndarray,
    bb20_width: np.ndarray,
    bb50_width: np.ndarray,
    hv: np.ndarray,
    entropy20: np.ndarray,
    entropy50: np.ndarray,
    entropy100: np.ndarray,
    hurst50: np.ndarray,
    hurst100: np.ndarray,
    pip_size: float,
    bin_step: float,
    warmup: int,
    max_events: int,
) -> tuple[np.ndarray, ...]:
    n = exec_close.shape[0]
    out_cap = max_events
    out_idx = np.zeros(out_cap, dtype=np.int64)
    out_dir = np.zeros(out_cap, dtype=np.int8)
    out_pa = np.zeros(out_cap, dtype=np.int16)
    out_vah = np.full(out_cap, np.nan)
    out_val = np.full(out_cap, np.nan)
    out_poc = np.full(out_cap, np.nan)
    out_entry = np.full(out_cap, np.nan)
    out_sl = np.full(out_cap, np.nan)
    out_tp = np.full(out_cap, np.nan)
    out_tp_kind = np.zeros(out_cap, dtype=np.int8)
    out_atr = np.full(out_cap, np.nan)
    out_atr_pct20 = np.full(out_cap, np.nan)
    out_atr_pct100 = np.full(out_cap, np.nan)
    out_atr_sess = np.full(out_cap, np.nan)
    out_bb20 = np.full(out_cap, np.nan)
    out_bb20_pct = np.full(out_cap, np.nan)
    out_bb50 = np.full(out_cap, np.nan)
    out_bb50_pct = np.full(out_cap, np.nan)
    out_hv20 = np.full(out_cap, np.nan)
    out_hv_pct = np.full(out_cap, np.nan)
    out_sess_rng = np.full(out_cap, np.nan)
    out_sess_rng_atr = np.full(out_cap, np.nan)
    out_recent5_rng = np.full(out_cap, np.nan)
    out_recent5_rng_atr = np.full(out_cap, np.nan)
    out_atr_chg = np.full(out_cap, np.nan)
    out_bb_chg = np.full(out_cap, np.nan)
    out_ent20 = np.full(out_cap, np.nan)
    out_ent50 = np.full(out_cap, np.nan)
    out_ent100 = np.full(out_cap, np.nan)
    out_ent_pct = np.full(out_cap, np.nan)
    out_ent_chg5 = np.full(out_cap, np.nan)
    out_ent_chg10 = np.full(out_cap, np.nan)
    out_h50 = np.full(out_cap, np.nan)
    out_h100 = np.full(out_cap, np.nan)
    out_vol_exp = np.zeros(out_cap, dtype=np.int8)
    out_exit_code = np.zeros(out_cap, dtype=np.int8)
    out_exit_px = np.full(out_cap, np.nan)
    out_exit_mins = np.zeros(out_cap, dtype=np.int32)
    out_bars_held = np.zeros(out_cap, dtype=np.int32)
    out_result_r = np.full(out_cap, np.nan)
    count = 0

    bars_per_day = max(int(24 * 60 / EXEC_BAR_MINUTES), 1)
    for i in range(warmup, n):
        if count >= out_cap:
            break
        cur_atr = atr[i]
        if not np.isfinite(cur_atr) or cur_atr <= 0.0:
            continue
        ts_ns = exec_dt_ns[i]
        sess_start = session_start_ns(int(ts_ns))
        vp_start = int(np.searchsorted(vp_dt_ns, sess_start, side="left"))
        vp_end = int(np.searchsorted(vp_dt_ns, ts_ns, side="right")) - 1
        if vp_end < vp_start:
            continue
        vah, val, poc = calculate_vp_profile_np(
            vp_close[vp_start : vp_end + 1],
            vp_volume[vp_start : vp_end + 1],
            bin_step,
        )
        direction_code = vp_touch_direction(
            exec_high[i],
            exec_low[i],
            exec_close[i],
            vah,
            val,
            cur_atr,
            pip_size,
        )
        if direction_code == 0:
            continue
        pa_flags = detect_pa_flags(exec_open, exec_high, exec_low, exec_close, i, direction_code)
        entry = exec_close[i]
        if direction_code < 0:
            stop = vah + SL_ATR_MULT * cur_atr
        else:
            stop = val - SL_ATR_MULT * cur_atr
        min_risk = max(pip_size * MIN_RISK_PIPS, cur_atr * 0.1)
        if abs(entry - stop) < min_risk:
            continue
        tp, tp_kind = resolve_take_profit(direction_code, entry, stop, poc, vah, val)
        sim_start = int(np.searchsorted(vp_dt_ns, ts_ns, side="left"))
        exit_code, exit_px, exit_mins, bars_held, result_r = simulate_trade_forward(
            int(ts_ns),
            entry,
            stop,
            tp,
            tp_kind,
            direction_code,
            vp_dt_ns,
            vp_high,
            vp_low,
            vp_close,
            sim_start,
        )

        atr_win20 = atr[max(0, i - bars_per_day * 20 + 1) : i + 1]
        atr_win100 = atr[max(0, i - 100 + 1) : i + 1]
        bb20_win = bb20_width[max(0, i - 100 + 1) : i + 1]
        bb50_win = bb50_width[max(0, i - 100 + 1) : i + 1]
        hv_win = hv[max(0, i - 100 + 1) : i + 1]
        ent_win = entropy100[max(0, i - 100 + 1) : i + 1]

        sess_exec_start = int(np.searchsorted(exec_dt_ns, sess_start, side="left"))
        sess_high = exec_high[i]
        sess_low = exec_low[i]
        if i >= sess_exec_start:
            for k in range(sess_exec_start, i + 1):
                if exec_high[k] > sess_high:
                    sess_high = exec_high[k]
                if exec_low[k] < sess_low:
                    sess_low = exec_low[k]
        recent_start = max(0, i - 4)
        recent_high = exec_high[i]
        recent_low = exec_low[i]
        for k in range(recent_start, i + 1):
            if exec_high[k] > recent_high:
                recent_high = exec_high[k]
            if exec_low[k] < recent_low:
                recent_low = exec_low[k]

        atr_prev5 = atr[i - 5] if i >= 5 else atr[i]
        bb_prev5 = bb20_width[i - 5] if i >= 5 else bb20_width[i]
        ent_prev5 = entropy50[i - 5] if i >= 5 else entropy50[i]
        ent_prev10 = entropy50[i - 10] if i >= 10 else entropy50[i]

        out_idx[count] = i
        out_dir[count] = direction_code
        out_pa[count] = pa_flags
        out_vah[count] = vah
        out_val[count] = val
        out_poc[count] = poc
        out_entry[count] = entry
        out_sl[count] = stop
        out_tp[count] = tp
        out_tp_kind[count] = tp_kind
        out_atr[count] = cur_atr
        out_atr_pct20[count] = percentile_rank(atr_win20, cur_atr)
        out_atr_pct100[count] = percentile_rank(atr_win100, cur_atr)
        sess_atr_sum = 0.0
        sess_atr_n = 0
        for k in range(sess_exec_start, i + 1):
            if np.isfinite(atr[k]):
                sess_atr_sum += atr[k]
                sess_atr_n += 1
        sess_atr_avg = sess_atr_sum / sess_atr_n if sess_atr_n > 0 else cur_atr
        out_atr_sess[count] = cur_atr / max(sess_atr_avg, 1e-9)
        out_bb20[count] = bb20_width[i]
        out_bb20_pct[count] = percentile_rank(bb20_win, bb20_width[i])
        out_bb50[count] = bb50_width[i]
        out_bb50_pct[count] = percentile_rank(bb50_win, bb50_width[i])
        out_hv20[count] = hv[i]
        out_hv_pct[count] = percentile_rank(hv_win, hv[i])
        out_sess_rng[count] = (sess_high - sess_low) / pip_size
        out_sess_rng_atr[count] = (sess_high - sess_low) / max(cur_atr, 1e-9)
        out_recent5_rng[count] = (recent_high - recent_low) / pip_size
        out_recent5_rng_atr[count] = (recent_high - recent_low) / max(cur_atr, 1e-9)
        out_atr_chg[count] = (cur_atr - atr_prev5) / max(abs(atr_prev5), 1e-9)
        out_bb_chg[count] = (bb20_width[i] - bb_prev5) / max(abs(bb_prev5), 1e-9)
        out_ent20[count] = entropy20[i]
        out_ent50[count] = entropy50[i]
        out_ent100[count] = entropy100[i]
        out_ent_pct[count] = percentile_rank(ent_win, entropy100[i])
        out_ent_chg5[count] = entropy50[i] - ent_prev5
        out_ent_chg10[count] = entropy50[i] - ent_prev10
        out_h50[count] = hurst50[i]
        out_h100[count] = hurst100[i]
        out_vol_exp[count] = 1 if detect_volatility_expansion(atr, bb20_width, i, 20) else 0
        out_exit_code[count] = exit_code
        out_exit_px[count] = exit_px
        out_exit_mins[count] = exit_mins
        out_bars_held[count] = bars_held
        out_result_r[count] = result_r
        count += 1

    return (
        out_idx[:count],
        out_dir[:count],
        out_pa[:count],
        out_vah[:count],
        out_val[:count],
        out_poc[:count],
        out_entry[:count],
        out_sl[:count],
        out_tp[:count],
        out_tp_kind[:count],
        out_atr[:count],
        out_atr_pct20[:count],
        out_atr_pct100[:count],
        out_atr_sess[:count],
        out_bb20[:count],
        out_bb20_pct[:count],
        out_bb50[:count],
        out_bb50_pct[:count],
        out_hv20[:count],
        out_hv_pct[:count],
        out_sess_rng[:count],
        out_sess_rng_atr[:count],
        out_recent5_rng[:count],
        out_recent5_rng_atr[:count],
        out_atr_chg[:count],
        out_bb_chg[:count],
        out_ent20[:count],
        out_ent50[:count],
        out_ent100[:count],
        out_ent_pct[:count],
        out_ent_chg5[:count],
        out_ent_chg10[:count],
        out_h50[:count],
        out_h100[:count],
        out_vol_exp[:count],
        out_exit_code[:count],
        out_exit_px[:count],
        out_exit_mins[:count],
        out_bars_held[:count],
        out_result_r[:count],
    )


def pa_flags_to_str(flags: int) -> str:
    parts: list[str] = []
    for code, label in PA_LABELS.items():
        if flags & code:
            parts.append(label)
    return "|".join(parts) if parts else "CLOSE_ONLY"


def exit_code_to_str(code: int) -> str:
    return EXIT_REASON_LABELS.get(int(code), "OPEN")


def precompute_entropy_series(close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = close.shape[0]
    e20 = np.full(n, np.nan)
    e50 = np.full(n, np.nan)
    e100 = np.full(n, np.nan)
    for i in range(n):
        e20[i] = shannon_entropy_returns(close, i, 20, ENTROPY_BINS)
        e50[i] = shannon_entropy_returns(close, i, 50, ENTROPY_BINS)
        e100[i] = shannon_entropy_returns(close, i, 100, ENTROPY_BINS)
    return e20, e50, e100


def precompute_hurst_series(close: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = close.shape[0]
    h50 = np.full(n, np.nan)
    h100 = np.full(n, np.nan)
    for i in range(n):
        h50[i] = hurst_exponent(close, i, 50)
        h100[i] = hurst_exponent(close, i, 100)
    return h50, h100
