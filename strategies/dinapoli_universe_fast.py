"""
DiNapoliUniverseFast — modernized DiNapoli methodology (numpy + Numba only).

Indicators: DMA, DiNapoli Stochastics, DiNapoli MACD, ZigZag.
Structure: A-B-C swings with 38.2–61.8% validation, COP/OP/XOP targets.
Signals: Single Penetration (trend pullback) and Double Repo (reversal).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from strategies.scan_numba_util import njit

SIGNAL_NONE = 0
SIGNAL_BUY = 1
SIGNAL_SELL = -1

PIVOT_PEAK = 1
PIVOT_TROUGH = -1

FIBO_RETRACE_MIN = 0.382
FIBO_RETRACE_MAX = 0.618
FIBO_COP_RATIO = 0.618
FIBO_OP_RATIO = 1.000
FIBO_XOP_RATIO = 1.618


# ---------------------------------------------------------------------------
# Numba JIT — core math
# ---------------------------------------------------------------------------
@njit(cache=True)
def compute_sma(data: np.ndarray, window: int) -> np.ndarray:
    n = data.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if window <= 0:
        return out
    for i in range(window - 1, n):
        total = 0.0
        for j in range(i - window + 1, i + 1):
            val = data[j]
            if np.isnan(val):
                total = np.nan
                break
            total += val
        if not np.isnan(total):
            out[i] = total / window
    return out


@njit(cache=True)
def compute_ema(data: np.ndarray, window: int) -> np.ndarray:
    n = data.shape[0]
    out = np.empty(n, dtype=np.float64)
    if window <= 0 or n == 0:
        for i in range(n):
            out[i] = np.nan
        return out
    alpha = 2.0 / (window + 1.0)
    out[0] = data[0]
    for i in range(1, n):
        out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1]
    return out


@njit(cache=True)
def compute_shifted_sma(data: np.ndarray, window: int, shift: int) -> np.ndarray:
    n = data.shape[0]
    sma = compute_sma(data, window)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        target = i + shift
        if target >= 0 and target < n and not np.isnan(sma[i]):
            out[target] = sma[i]
    return out


@njit(cache=True)
def compute_dma_set(close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dma_3_3 = compute_shifted_sma(close, 3, 3)
    dma_7_5 = compute_shifted_sma(close, 7, 5)
    dma_25_5 = compute_shifted_sma(close, 25, 5)
    return dma_3_3, dma_7_5, dma_25_5


@njit(cache=True)
def compute_dinapoli_stochastics(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 8,
    smooth1: int = 3,
    smooth2: int = 3,
) -> np.ndarray:
    n = close.shape[0]
    raw_k = np.full(n, np.nan, dtype=np.float64)
    for i in range(k_period - 1, n):
        hh = high[i - k_period + 1]
        ll = low[i - k_period + 1]
        for j in range(i - k_period + 2, i + 1):
            if high[j] > hh:
                hh = high[j]
            if low[j] < ll:
                ll = low[j]
        denom = hh - ll
        if denom > 0.0:
            raw_k[i] = 100.0 * (close[i] - ll) / denom
        else:
            raw_k[i] = 50.0
    smooth_a = compute_sma(raw_k, smooth1)
    smooth_b = compute_sma(smooth_a, smooth2)
    return smooth_b


@njit(cache=True)
def compute_dinapoli_macd(
    close: np.ndarray,
    fast_period: int = 8,
    slow_period: int = 17,
    signal_period: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fast = compute_ema(close, fast_period)
    slow = compute_ema(close, slow_period)
    n = close.shape[0]
    macd_line = np.empty(n, dtype=np.float64)
    for i in range(n):
        macd_line[i] = fast[i] - slow[i]
    signal_line = compute_ema(macd_line, signal_period)
    return macd_line, signal_line, macd_line - signal_line


@njit(cache=True)
def compute_zigzag(
    high: np.ndarray,
    low: np.ndarray,
    deviation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = high.shape[0]
    flags = np.zeros(n, dtype=np.int8)
    pivot_indices = np.empty(n, dtype=np.int64)
    pivot_prices = np.empty(n, dtype=np.float64)
    pivot_types = np.empty(n, dtype=np.int8)
    pivot_count = 0

    if n == 0:
        return flags, pivot_indices[:0], pivot_prices[:0], pivot_types[:0]

    trend = 1
    candidate_i = 0
    candidate_price = high[0]

    for i in range(1, n):
        if trend == 1:
            if high[i] > candidate_price:
                candidate_price = high[i]
                candidate_i = i
            if candidate_price > 0.0:
                drop = (candidate_price - low[i]) / candidate_price
            else:
                drop = 0.0
            if drop >= deviation:
                flags[candidate_i] = PIVOT_PEAK
                pivot_indices[pivot_count] = candidate_i
                pivot_prices[pivot_count] = candidate_price
                pivot_types[pivot_count] = PIVOT_PEAK
                pivot_count += 1
                trend = -1
                candidate_i = i
                candidate_price = low[i]
        else:
            if low[i] < candidate_price:
                candidate_price = low[i]
                candidate_i = i
            if candidate_price > 0.0:
                rise = (high[i] - candidate_price) / candidate_price
            else:
                rise = 0.0
            if rise >= deviation:
                flags[candidate_i] = PIVOT_TROUGH
                pivot_indices[pivot_count] = candidate_i
                pivot_prices[pivot_count] = candidate_price
                pivot_types[pivot_count] = PIVOT_TROUGH
                pivot_count += 1
                trend = 1
                candidate_i = i
                candidate_price = high[i]

    return (
        flags,
        pivot_indices[:pivot_count],
        pivot_prices[:pivot_count],
        pivot_types[:pivot_count],
    )


@njit(cache=True)
def _consecutive_closes_side(
    close: np.ndarray,
    ref: np.ndarray,
    end_idx: int,
    side: int,
    min_bars: int,
) -> bool:
    if end_idx < min_bars - 1:
        return False
    for k in range(min_bars):
        idx = end_idx - k
        if np.isnan(ref[idx]):
            return False
        if side > 0:
            if close[idx] <= ref[idx]:
                return False
        else:
            if close[idx] >= ref[idx]:
                return False
    return True


@njit(cache=True)
def _thrust_extremes(
    high: np.ndarray,
    low: np.ndarray,
    start_idx: int,
    end_idx: int,
    bullish: bool,
) -> tuple[float, float]:
    hi = high[start_idx]
    lo = low[start_idx]
    for j in range(start_idx, end_idx + 1):
        if high[j] > hi:
            hi = high[j]
        if low[j] < lo:
            lo = low[j]
    if bullish:
        return lo, hi
    return hi, lo


@njit(cache=True)
def _count_dma_crosses(
    close: np.ndarray,
    dma: np.ndarray,
    start_idx: int,
    end_idx: int,
    from_above: bool,
) -> int:
    crosses = 0
    if start_idx >= end_idx:
        return 0
    for j in range(start_idx + 1, end_idx + 1):
        if np.isnan(dma[j]) or np.isnan(dma[j - 1]):
            continue
        prev_above = close[j - 1] > dma[j - 1]
        curr_above = close[j] > dma[j]
        if from_above:
            if prev_above and not curr_above:
                crosses += 1
        else:
            if not prev_above and curr_above:
                crosses += 1
    return crosses


@njit(cache=True)
def _price_in_zone(price: float, zone_lo: float, zone_hi: float) -> bool:
    lo = zone_lo if zone_lo < zone_hi else zone_hi
    hi = zone_hi if zone_hi > zone_lo else zone_lo
    return price >= lo and price <= hi


@njit(cache=True)
def scan_signals_njit(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    dma_3_3: np.ndarray,
    macd_line: np.ndarray,
    signal_line: np.ndarray,
    struct_dir: np.ndarray,
    struct_fibo_382: np.ndarray,
    struct_fibo_618: np.ndarray,
    struct_c_idx: np.ndarray,
    struct_valid_until: np.ndarray,
    min_thrust_bars: int,
    lookback: int,
) -> np.ndarray:
    n = close.shape[0]
    signals = np.zeros(n, dtype=np.int8)
    n_struct = struct_dir.shape[0]

    for i in range(min_thrust_bars, n):
        if np.isnan(dma_3_3[i]):
            continue

        macd_bull = macd_line[i] > signal_line[i]
        macd_bear = macd_line[i] < signal_line[i]

        lb_start = 0 if i - lookback < 0 else i - lookback

        # --- Strategy 2: Double Repo (reversal) ---
        if _consecutive_closes_side(close, dma_3_3, i - 1, 1, min_thrust_bars):
            thrust_start = i - min_thrust_bars
            crosses = _count_dma_crosses(close, dma_3_3, thrust_start, i, True)
            if crosses >= 2 and close[i] < dma_3_3[i] and macd_bear:
                signals[i] = SIGNAL_SELL

        if _consecutive_closes_side(close, dma_3_3, i - 1, -1, min_thrust_bars):
            thrust_start = i - min_thrust_bars
            crosses = _count_dma_crosses(close, dma_3_3, thrust_start, i, False)
            if crosses >= 2 and close[i] > dma_3_3[i] and macd_bull:
                if signals[i] == SIGNAL_NONE:
                    signals[i] = SIGNAL_BUY

        if signals[i] != SIGNAL_NONE:
            continue

        # --- Strategy 1: Single Penetration (trend pullback) ---
        for s in range(n_struct):
            if i < struct_c_idx[s] or i > struct_valid_until[s]:
                continue
            direction = struct_dir[s]
            zone_a = struct_fibo_382[s]
            zone_b = struct_fibo_618[s]

            if direction > 0:
                if not _consecutive_closes_side(close, dma_3_3, i - 1, 1, min_thrust_bars):
                    continue
                thrust_end = i - 1
                thrust_start = thrust_end - min_thrust_bars + 1
                if thrust_start < lb_start:
                    continue
                thrust_lo, thrust_hi = _thrust_extremes(high, low, thrust_start, thrust_end, True)
                span = thrust_hi - thrust_lo
                if span <= 0.0:
                    continue
                fib_lo = thrust_hi - FIBO_RETRACE_MAX * span
                fib_hi = thrust_hi - FIBO_RETRACE_MIN * span
                crosses = _count_dma_crosses(close, dma_3_3, thrust_start, i, True)
                if crosses != 1:
                    continue
                in_thrust_fibo = _price_in_zone(close[i], fib_lo, fib_hi)
                in_abc_fibo = _price_in_zone(close[i], zone_a, zone_b)
                if (in_thrust_fibo or in_abc_fibo) and macd_bull:
                    signals[i] = SIGNAL_BUY
                    break
            else:
                if not _consecutive_closes_side(close, dma_3_3, i - 1, -1, min_thrust_bars):
                    continue
                thrust_end = i - 1
                thrust_start = thrust_end - min_thrust_bars + 1
                if thrust_start < lb_start:
                    continue
                thrust_hi, thrust_lo = _thrust_extremes(high, low, thrust_start, thrust_end, False)
                span = thrust_hi - thrust_lo
                if span <= 0.0:
                    continue
                fib_lo = thrust_lo + FIBO_RETRACE_MIN * span
                fib_hi = thrust_lo + FIBO_RETRACE_MAX * span
                crosses = _count_dma_crosses(close, dma_3_3, thrust_start, i, False)
                if crosses != 1:
                    continue
                in_thrust_fibo = _price_in_zone(close[i], fib_lo, fib_hi)
                in_abc_fibo = _price_in_zone(close[i], zone_a, zone_b)
                if (in_thrust_fibo or in_abc_fibo) and macd_bear:
                    signals[i] = SIGNAL_SELL
                    break

    return signals


# ---------------------------------------------------------------------------
# Python-side structure records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FiboStructure:
    direction: int
    a_idx: int
    b_idx: int
    c_idx: int
    a_price: float
    b_price: float
    c_price: float
    retracement: float
    fibo_382: float
    fibo_618: float
    cop: float
    op: float
    xop: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "a_idx": self.a_idx,
            "b_idx": self.b_idx,
            "c_idx": self.c_idx,
            "a_price": self.a_price,
            "b_price": self.b_price,
            "c_price": self.c_price,
            "retracement": self.retracement,
            "fibo_382": self.fibo_382,
            "fibo_618": self.fibo_618,
            "cop": self.cop,
            "op": self.op,
            "xop": self.xop,
        }


def detect_fibo_structures_from_pivots(
    pivot_indices: np.ndarray,
    pivot_prices: np.ndarray,
    pivot_types: np.ndarray,
) -> list[FiboStructure]:
    structures: list[FiboStructure] = []
    count = pivot_indices.shape[0]
    if count < 3:
        return structures

    for k in range(count - 2):
        a_idx = int(pivot_indices[k])
        b_idx = int(pivot_indices[k + 1])
        c_idx = int(pivot_indices[k + 2])
        a_type = int(pivot_types[k])
        b_type = int(pivot_types[k + 1])
        c_type = int(pivot_types[k + 2])
        a_price = float(pivot_prices[k])
        b_price = float(pivot_prices[k + 1])
        c_price = float(pivot_prices[k + 2])

        if a_type == PIVOT_TROUGH and b_type == PIVOT_PEAK and c_type == PIVOT_TROUGH:
            direction = 1
            ab = b_price - a_price
            if ab <= 0.0:
                continue
            retracement = (b_price - c_price) / ab
            fibo_382 = b_price - FIBO_RETRACE_MIN * ab
            fibo_618 = b_price - FIBO_RETRACE_MAX * ab
            cop = c_price + FIBO_COP_RATIO * ab
            op = c_price + FIBO_OP_RATIO * ab
            xop = c_price + FIBO_XOP_RATIO * ab
        elif a_type == PIVOT_PEAK and b_type == PIVOT_TROUGH and c_type == PIVOT_PEAK:
            direction = -1
            ab = a_price - b_price
            if ab <= 0.0:
                continue
            retracement = (c_price - b_price) / ab
            fibo_382 = b_price + FIBO_RETRACE_MIN * ab
            fibo_618 = b_price + FIBO_RETRACE_MAX * ab
            cop = c_price - FIBO_COP_RATIO * ab
            op = c_price - FIBO_OP_RATIO * ab
            xop = c_price - FIBO_XOP_RATIO * ab
        else:
            continue

        if retracement < FIBO_RETRACE_MIN or retracement > FIBO_RETRACE_MAX:
            continue

        structures.append(
            FiboStructure(
                direction=direction,
                a_idx=a_idx,
                b_idx=b_idx,
                c_idx=c_idx,
                a_price=a_price,
                b_price=b_price,
                c_price=c_price,
                retracement=retracement,
                fibo_382=fibo_382,
                fibo_618=fibo_618,
                cop=cop,
                op=op,
                xop=xop,
            )
        )
    return structures


def pack_structures_for_njit(
    structures: list[FiboStructure],
    n_bars: int,
    valid_horizon: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m = len(structures)
    if m == 0:
        empty_i = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float64)
        return (
            np.empty(0, dtype=np.int8),
            empty_f,
            empty_f,
            empty_i,
            empty_i,
        )
    struct_dir = np.empty(m, dtype=np.int8)
    struct_fibo_382 = np.empty(m, dtype=np.float64)
    struct_fibo_618 = np.empty(m, dtype=np.float64)
    struct_c_idx = np.empty(m, dtype=np.int64)
    struct_valid_until = np.empty(m, dtype=np.int64)
    for i, st in enumerate(structures):
        struct_dir[i] = st.direction
        struct_fibo_382[i] = st.fibo_382
        struct_fibo_618[i] = st.fibo_618
        struct_c_idx[i] = st.c_idx
        struct_valid_until[i] = min(n_bars - 1, st.c_idx + valid_horizon)
    return struct_dir, struct_fibo_382, struct_fibo_618, struct_c_idx, struct_valid_until


class DiNapoliUniverseFast:
    """Production-grade DiNapoli engine — numpy arrays in, numpy arrays out."""

    def __init__(
        self,
        zigzag_dev: float = 0.001,
        min_thrust_bars: int = 8,
        signal_lookback: int = 80,
        structure_valid_horizon: int = 120,
        stoch_k_period: int = 8,
        stoch_smooth1: int = 3,
        stoch_smooth2: int = 3,
        macd_fast: int = 8,
        macd_slow: int = 17,
        macd_signal: int = 9,
    ) -> None:
        self.zigzag_dev = float(zigzag_dev)
        self.min_thrust_bars = int(min_thrust_bars)
        self.signal_lookback = int(signal_lookback)
        self.structure_valid_horizon = int(structure_valid_horizon)
        self.stoch_k_period = int(stoch_k_period)
        self.stoch_smooth1 = int(stoch_smooth1)
        self.stoch_smooth2 = int(stoch_smooth2)
        self.macd_fast = int(macd_fast)
        self.macd_slow = int(macd_slow)
        self.macd_signal = int(macd_signal)

    def compute_all_indicators(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
    ) -> dict[str, np.ndarray]:
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        close = np.asarray(close, dtype=np.float64)
        dma_3_3, dma_7_5, dma_25_5 = compute_dma_set(close)
        stoch = compute_dinapoli_stochastics(
            high,
            low,
            close,
            self.stoch_k_period,
            self.stoch_smooth1,
            self.stoch_smooth2,
        )
        macd_line, signal_line, histogram = compute_dinapoli_macd(
            close,
            self.macd_fast,
            self.macd_slow,
            self.macd_signal,
        )
        zigzag_flags, pivot_idx, pivot_prices, pivot_types = compute_zigzag(
            high,
            low,
            self.zigzag_dev,
        )
        return {
            "dma_3_3": dma_3_3,
            "dma_7_5": dma_7_5,
            "dma_25_5": dma_25_5,
            "stochastics": stoch,
            "macd_line": macd_line,
            "macd_signal": signal_line,
            "macd_histogram": histogram,
            "zigzag_flags": zigzag_flags,
            "pivot_indices": pivot_idx,
            "pivot_prices": pivot_prices,
            "pivot_types": pivot_types,
        }

    def detect_fibo_structures(
        self,
        pivot_indices: np.ndarray,
        pivot_prices: np.ndarray,
        pivot_types: np.ndarray,
    ) -> list[FiboStructure]:
        return detect_fibo_structures_from_pivots(pivot_indices, pivot_prices, pivot_types)

    def scan_signals(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        dma_3_3: np.ndarray,
        macd_line: np.ndarray,
        macd_signal: np.ndarray,
        fibo_structures: list[FiboStructure],
    ) -> np.ndarray:
        close = np.asarray(close, dtype=np.float64)
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        dma_3_3 = np.asarray(dma_3_3, dtype=np.float64)
        macd_line = np.asarray(macd_line, dtype=np.float64)
        macd_signal = np.asarray(macd_signal, dtype=np.float64)
        packed = pack_structures_for_njit(
            fibo_structures,
            close.shape[0],
            self.structure_valid_horizon,
        )
        return scan_signals_njit(
            close,
            high,
            low,
            dma_3_3,
            macd_line,
            macd_signal,
            packed[0],
            packed[1],
            packed[2],
            packed[3],
            packed[4],
            self.min_thrust_bars,
            self.signal_lookback,
        )

    def run_pipeline(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        timestamp: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], list[FiboStructure], np.ndarray]:
        indicators = self.compute_all_indicators(high, low, close)
        structures = self.detect_fibo_structures(
            indicators["pivot_indices"],
            indicators["pivot_prices"],
            indicators["pivot_types"],
        )
        signals = self.scan_signals(
            close,
            high,
            low,
            indicators["dma_3_3"],
            indicators["macd_line"],
            indicators["macd_signal"],
            structures,
        )
        if timestamp is not None:
            indicators["timestamp"] = np.asarray(timestamp)
        return indicators, structures, signals


__all__ = [
    "DiNapoliUniverseFast",
    "FiboStructure",
    "SIGNAL_BUY",
    "SIGNAL_NONE",
    "SIGNAL_SELL",
    "compute_dinapoli_macd",
    "compute_dinapoli_stochastics",
    "compute_dma_set",
    "compute_ema",
    "compute_shifted_sma",
    "compute_sma",
    "compute_zigzag",
    "detect_fibo_structures_from_pivots",
    "pack_structures_for_njit",
    "scan_signals_njit",
]
