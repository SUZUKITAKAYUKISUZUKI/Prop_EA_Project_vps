"""SMRS Phase 1 — Numba scan kernels (M1 mean-reversion grid). No pandas."""

from __future__ import annotations

import numpy as np

from strategies.scan_numba_util import njit

MA_PERIOD = 20
ATR_PERIOD = 14
ATR_PCT_WINDOW = 100
WARMUP = max(MA_PERIOD, ATR_PERIOD, ATR_PCT_WINDOW) + 5
BAR_MINUTES = 1

ENVELOPE_DEVS = np.array([0.10, 0.15, 0.20, 0.25, 0.30], dtype=np.float64)

# session_filter codes
SESSION_ALL = 0
SESSION_LONDON = 1
SESSION_NY = 2
SESSION_LONDON_NY = 3

# atr_filter codes
ATR_NONE = 0
ATR_P25 = 1
ATR_P50 = 2
ATR_RANGE = 3

# exit_logic codes
EXIT_A = 0
EXIT_B = 1
EXIT_C = 2
EXIT_D = 3  # 1.0R
EXIT_E = 4  # 1.5R
EXIT_F = 5  # 2.0R


@njit(cache=True)
def hours_from_ts_ns(ts_ns: np.ndarray) -> np.ndarray:
    n = len(ts_ns)
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        secs = ts_ns[i] // 1_000_000_000
        out[i] = int((secs % 86400) // 3600)
    return out


@njit(cache=True)
def rolling_mean(arr: np.ndarray, period: int) -> np.ndarray:
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = np.nan
    for i in range(period - 1, n):
        s = 0.0
        for k in range(i - period + 1, i + 1):
            s += arr[k]
        out[i] = s / period
    return out


@njit(cache=True)
def rolling_std(arr: np.ndarray, period: int) -> np.ndarray:
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = np.nan
    for i in range(period - 1, n):
        m = 0.0
        for k in range(i - period + 1, i + 1):
            m += arr[k]
        m /= period
        v = 0.0
        for k in range(i - period + 1, i + 1):
            d = arr[k] - m
            v += d * d
        out[i] = (v / period) ** 0.5
    return out


@njit(cache=True)
def rolling_percentile(arr: np.ndarray, window: int, q_pct: float) -> np.ndarray:
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    buf = np.empty(window, dtype=np.float64)
    for i in range(n):
        out[i] = np.nan
    for i in range(window - 1, n):
        valid = 0
        for k in range(window):
            v = arr[i - window + 1 + k]
            if np.isfinite(v):
                buf[valid] = v
                valid += 1
        if valid == 0:
            continue
        buf[:valid].sort()
        rank = (q_pct / 100.0) * (valid - 1)
        lo = int(rank)
        hi = lo + 1
        if hi >= valid:
            out[i] = buf[lo]
        else:
            w = rank - lo
            out[i] = buf[lo] * (1.0 - w) + buf[hi] * w
    return out


@njit(cache=True)
def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, max(hc, lc))
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = np.nan
    for i in range(period - 1, n):
        s = 0.0
        for k in range(i - period + 1, i + 1):
            s += tr[k]
        out[i] = s / period
    return out


@njit(cache=True)
def build_envelope_bands(ma: np.ndarray, dev_pct: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(ma)
    pct = dev_pct / 100.0
    upper = np.empty(n, dtype=np.float64)
    lower = np.empty(n, dtype=np.float64)
    for i in range(n):
        upper[i] = ma[i] * (1.0 + pct)
        lower[i] = ma[i] * (1.0 - pct)
    return upper, lower


@njit(cache=True)
def session_ok(hour: int, session_code: int) -> bool:
    if session_code == SESSION_ALL:
        return True
    if session_code == SESSION_LONDON:
        return 7 <= hour < 13
    if session_code == SESSION_NY:
        return 13 <= hour < 22
    if session_code == SESSION_LONDON_NY:
        return 7 <= hour < 22
    return False


@njit(cache=True)
def atr_ok(
    atr: float,
    atr_p25: float,
    atr_p50: float,
    atr_p75: float,
    atr_code: int,
) -> bool:
    if atr_code == ATR_NONE:
        return True
    if atr_code == ATR_P25:
        return atr >= atr_p25
    if atr_code == ATR_P50:
        return atr >= atr_p50
    if atr_code == ATR_RANGE:
        return atr >= atr_p25 and atr <= atr_p75
    return False


@njit(cache=True)
def fixed_rr(exit_code: int) -> float:
    if exit_code == EXIT_D:
        return 1.0
    if exit_code == EXIT_E:
        return 1.5
    if exit_code == EXIT_F:
        return 2.0
    return -1.0


@njit(cache=True)
def profit_factor(r: np.ndarray) -> float:
    if r.size == 0:
        return 0.0
    gw = 0.0
    gl = 0.0
    for i in range(r.size):
        if r[i] > 0.0:
            gw += r[i]
        elif r[i] < 0.0:
            gl -= r[i]
    if gl <= 0.0:
        return 1.0e9 if gw > 0.0 else 0.0
    return gw / gl


@njit(cache=True)
def max_drawdown_r(r: np.ndarray) -> float:
    if r.size == 0:
        return 0.0
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for i in range(r.size):
        equity += r[i]
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    return max_dd


@njit(cache=True)
def simulate_smrs(
    ts_ns: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    hours: np.ndarray,
    ma: np.ndarray,
    z: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    atr: np.ndarray,
    atr_p25: np.ndarray,
    atr_p50: np.ndarray,
    atr_p75: np.ndarray,
    z_threshold: float,
    session_code: int,
    atr_code: int,
    exit_code: int,
    max_hold_hours: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    warmup = WARMUP
    max_bars = max(1, max_hold_hours * 60 // BAR_MINUTES)
    cap = max(256, (n - warmup) // 3)
    profits = np.empty(cap, dtype=np.float64)
    holds = np.empty(cap, dtype=np.float64)
    ts_out = np.empty(cap, dtype=np.int64)
    count = 0
    rr = fixed_rr(exit_code)
    has_fixed_rr = rr > 0.0

    i = warmup
    while i < n - 1:
        if not session_ok(int(hours[i]), session_code):
            i += 1
            continue
        ai = atr[i]
        if not np.isfinite(ai) or ai <= 0.0:
            i += 1
            continue
        if not atr_ok(ai, atr_p25[i], atr_p50[i], atr_p75[i], atr_code):
            i += 1
            continue
        if not np.isfinite(z[i]) or not np.isfinite(lower[i]) or not np.isfinite(upper[i]):
            i += 1
            continue

        direction = 0
        if close[i] < lower[i] and z[i] <= -z_threshold:
            direction = 1
        elif close[i] > upper[i] and z[i] >= z_threshold:
            direction = -1
        if direction == 0:
            i += 1
            continue

        entry = close[i]
        if direction == 1:
            sl = entry - ai
            tp = entry + rr * ai if has_fixed_rr else 0.0
        else:
            sl = entry + ai
            tp = entry - rr * ai if has_fixed_rr else 0.0

        exit_i = i
        exit_price = entry
        exited = False
        j_end = i + 1 + max_bars
        if j_end > n:
            j_end = n

        for j in range(i + 1, j_end):
            hi = high[j]
            lo = low[j]
            if direction == 1:
                if lo <= sl:
                    exit_i = j
                    exit_price = sl
                    exited = True
                    break
                if has_fixed_rr and hi >= tp:
                    exit_i = j
                    exit_price = tp
                    exited = True
                    break
                if exit_code == EXIT_A and hi >= ma[j]:
                    exit_i = j
                    exit_price = ma[j]
                    exited = True
                    break
                if exit_code == EXIT_B:
                    target = ma[j] + 0.5 * (upper[j] - ma[j])
                    if hi >= target:
                        exit_i = j
                        exit_price = target
                        exited = True
                        break
                if exit_code == EXIT_C and z[j] >= 0.0:
                    exit_i = j
                    exit_price = close[j]
                    exited = True
                    break
            else:
                if hi >= sl:
                    exit_i = j
                    exit_price = sl
                    exited = True
                    break
                if has_fixed_rr and lo <= tp:
                    exit_i = j
                    exit_price = tp
                    exited = True
                    break
                if exit_code == EXIT_A and lo <= ma[j]:
                    exit_i = j
                    exit_price = ma[j]
                    exited = True
                    break
                if exit_code == EXIT_B:
                    target = ma[j] - 0.5 * (ma[j] - lower[j])
                    if lo <= target:
                        exit_i = j
                        exit_price = target
                        exited = True
                        break
                if exit_code == EXIT_C and z[j] <= 0.0:
                    exit_i = j
                    exit_price = close[j]
                    exited = True
                    break

        if not exited:
            exit_i = i + max_bars
            if exit_i >= n:
                exit_i = n - 1
            exit_price = close[exit_i]

        if direction == 1:
            pnl_r = (exit_price - entry) / ai
        else:
            pnl_r = (entry - exit_price) / ai
        if has_fixed_rr:
            if pnl_r < -1.0:
                pnl_r = -1.0
            elif pnl_r > rr:
                pnl_r = rr
        elif pnl_r < -1.0:
            pnl_r = -1.0

        if count >= cap:
            break
        profits[count] = pnl_r
        holds[count] = float((exit_i - i) * BAR_MINUTES)
        ts_out[count] = ts_ns[i]
        count += 1
        i = exit_i + 1

    return profits[:count], holds[:count], ts_out[:count]


# exit_reason codes for feature log
REASON_SL = 1
REASON_TP = 2
REASON_MA = 3
REASON_ENV_B = 4
REASON_Z0 = 5
REASON_TIME = 6


@njit(cache=True)
def simulate_smrs_log(
    ts_ns: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    hours: np.ndarray,
    ma: np.ndarray,
    z: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    atr: np.ndarray,
    atr_p25: np.ndarray,
    atr_p50: np.ndarray,
    atr_p75: np.ndarray,
    z_threshold: float,
    session_code: int,
    atr_code: int,
    exit_code: int,
    max_hold_hours: int,
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
]:
    """Detailed trade log for pure BT feature export."""
    n = len(close)
    warmup = WARMUP
    max_bars = max(1, max_hold_hours * 60 // BAR_MINUTES)
    cap = max(4096, (n - warmup) // 5)
    profits = np.empty(cap, dtype=np.float64)
    holds = np.empty(cap, dtype=np.float64)
    ts_entry = np.empty(cap, dtype=np.int64)
    ts_exit = np.empty(cap, dtype=np.int64)
    direction_out = np.empty(cap, dtype=np.int8)
    entry_px = np.empty(cap, dtype=np.float64)
    exit_px = np.empty(cap, dtype=np.float64)
    sl_px = np.empty(cap, dtype=np.float64)
    tp_px = np.empty(cap, dtype=np.float64)
    z_entry = np.empty(cap, dtype=np.float64)
    ma_entry = np.empty(cap, dtype=np.float64)
    upper_entry = np.empty(cap, dtype=np.float64)
    lower_entry = np.empty(cap, dtype=np.float64)
    atr_entry = np.empty(cap, dtype=np.float64)
    atr_p25_entry = np.empty(cap, dtype=np.float64)
    atr_p50_entry = np.empty(cap, dtype=np.float64)
    atr_p75_entry = np.empty(cap, dtype=np.float64)
    hour_entry = np.empty(cap, dtype=np.int32)
    bars_held = np.empty(cap, dtype=np.int32)
    exit_reason = np.empty(cap, dtype=np.int8)
    count = 0
    rr = fixed_rr(exit_code)
    has_fixed_rr = rr > 0.0

    i = warmup
    while i < n - 1:
        if not session_ok(int(hours[i]), session_code):
            i += 1
            continue
        ai = atr[i]
        if not np.isfinite(ai) or ai <= 0.0:
            i += 1
            continue
        if not atr_ok(ai, atr_p25[i], atr_p50[i], atr_p75[i], atr_code):
            i += 1
            continue
        if not np.isfinite(z[i]) or not np.isfinite(lower[i]) or not np.isfinite(upper[i]):
            i += 1
            continue

        direction = 0
        if close[i] < lower[i] and z[i] <= -z_threshold:
            direction = 1
        elif close[i] > upper[i] and z[i] >= z_threshold:
            direction = -1
        if direction == 0:
            i += 1
            continue

        entry = close[i]
        if direction == 1:
            sl = entry - ai
            tp = entry + rr * ai if has_fixed_rr else 0.0
        else:
            sl = entry + ai
            tp = entry - rr * ai if has_fixed_rr else 0.0

        exit_i = i
        exit_price = entry
        reason = REASON_TIME
        j_end = i + 1 + max_bars
        if j_end > n:
            j_end = n
        exited = False

        for j in range(i + 1, j_end):
            hi = high[j]
            lo = low[j]
            if direction == 1:
                if lo <= sl:
                    exit_i, exit_price, reason, exited = j, sl, REASON_SL, True
                    break
                if has_fixed_rr and hi >= tp:
                    exit_i, exit_price, reason, exited = j, tp, REASON_TP, True
                    break
                if exit_code == EXIT_A and hi >= ma[j]:
                    exit_i, exit_price, reason, exited = j, ma[j], REASON_MA, True
                    break
                if exit_code == EXIT_B:
                    target = ma[j] + 0.5 * (upper[j] - ma[j])
                    if hi >= target:
                        exit_i, exit_price, reason, exited = j, target, REASON_ENV_B, True
                        break
                if exit_code == EXIT_C and z[j] >= 0.0:
                    exit_i, exit_price, reason, exited = j, close[j], REASON_Z0, True
                    break
            else:
                if hi >= sl:
                    exit_i, exit_price, reason, exited = j, sl, REASON_SL, True
                    break
                if has_fixed_rr and lo <= tp:
                    exit_i, exit_price, reason, exited = j, tp, REASON_TP, True
                    break
                if exit_code == EXIT_A and lo <= ma[j]:
                    exit_i, exit_price, reason, exited = j, ma[j], REASON_MA, True
                    break
                if exit_code == EXIT_B:
                    target = ma[j] - 0.5 * (ma[j] - lower[j])
                    if lo <= target:
                        exit_i, exit_price, reason, exited = j, target, REASON_ENV_B, True
                        break
                if exit_code == EXIT_C and z[j] <= 0.0:
                    exit_i, exit_price, reason, exited = j, close[j], REASON_Z0, True
                    break

        if not exited:
            exit_i = i + max_bars
            if exit_i >= n:
                exit_i = n - 1
            exit_price = close[exit_i]
            reason = REASON_TIME

        if direction == 1:
            pnl_r = (exit_price - entry) / ai
        else:
            pnl_r = (entry - exit_price) / ai
        if has_fixed_rr:
            if pnl_r < -1.0:
                pnl_r = -1.0
            elif pnl_r > rr:
                pnl_r = rr
        elif pnl_r < -1.0:
            pnl_r = -1.0

        if count >= cap:
            break
        profits[count] = pnl_r
        holds[count] = float((exit_i - i) * BAR_MINUTES)
        ts_entry[count] = ts_ns[i]
        ts_exit[count] = ts_ns[exit_i]
        direction_out[count] = direction
        entry_px[count] = entry
        exit_px[count] = exit_price
        sl_px[count] = sl
        tp_px[count] = tp if has_fixed_rr else np.nan
        z_entry[count] = z[i]
        ma_entry[count] = ma[i]
        upper_entry[count] = upper[i]
        lower_entry[count] = lower[i]
        atr_entry[count] = ai
        atr_p25_entry[count] = atr_p25[i]
        atr_p50_entry[count] = atr_p50[i]
        atr_p75_entry[count] = atr_p75[i]
        hour_entry[count] = hours[i]
        bars_held[count] = exit_i - i
        exit_reason[count] = reason
        count += 1
        i = exit_i + 1

    sl = slice(0, count)
    return (
        profits[sl],
        holds[sl],
        ts_entry[sl],
        ts_exit[sl],
        direction_out[sl],
        entry_px[sl],
        exit_px[sl],
        sl_px[sl],
        tp_px[sl],
        z_entry[sl],
        ma_entry[sl],
        upper_entry[sl],
        lower_entry[sl],
        atr_entry[sl],
        atr_p25_entry[sl],
        atr_p50_entry[sl],
        atr_p75_entry[sl],
        hour_entry[sl],
        bars_held[sl],
        exit_reason[sl],
    )


@njit(cache=True)
def prep_pair_arrays(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ts_ns: np.ndarray,
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
]:
    """Returns hours, ma, z, atr, atr_p25, atr_p50, atr_p75, upper_all, lower_all."""
    hours = hours_from_ts_ns(ts_ns)
    ma = rolling_mean(close, MA_PERIOD)
    std = rolling_std(close, MA_PERIOD)
    n = len(close)
    z = np.empty(n, dtype=np.float64)
    for i in range(n):
        if std[i] > 0.0 and np.isfinite(std[i]):
            z[i] = (close[i] - ma[i]) / std[i]
        else:
            z[i] = np.nan
    atr = compute_atr(high, low, close, ATR_PERIOD)
    atr_p25 = rolling_percentile(atr, ATR_PCT_WINDOW, 25.0)
    atr_p50 = rolling_percentile(atr, ATR_PCT_WINDOW, 50.0)
    atr_p75 = rolling_percentile(atr, ATR_PCT_WINDOW, 75.0)

    n_dev = len(ENVELOPE_DEVS)
    upper_all = np.empty((n_dev, n), dtype=np.float64)
    lower_all = np.empty((n_dev, n), dtype=np.float64)
    for d in range(n_dev):
        u, l = build_envelope_bands(ma, ENVELOPE_DEVS[d])
        upper_all[d] = u
        lower_all[d] = l
    return hours, ma, z, atr, atr_p25, atr_p50, atr_p75, upper_all, lower_all
