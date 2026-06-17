"""
strategies/dbbs_common.py — Dual Bollinger Band Squeeze (DBBS) shared types + Numba kernels.

MTF: exec M15 / structure H1 (BB) / ATR H4.
Pure BT / feature-collection phase: no candidate_score, bayes_probability=1.0 fixed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from strategies.htf_trend_analyzer import clip_as_of
from strategies.market_utils import pip_size_for_pair
from strategies.scan_numba_util import njit

STRATEGY_ABBREV = "DBBS"
STRATEGY_FULL_NAME = "Dual Bollinger Band Squeeze"
DBBSG_ABBREV = "DBBSG"
DBBSG_FULL_NAME = "Dual Bollinger Band Squeeze - GOLD"
DBBS_PAIR_PRIMARY = "EURUSD"
DBBS_PAIR_SECONDARY = "GBPUSD"
DBBS_PAIR_XAU = "XAUUSD"
DBBSG_PAIR = DBBS_PAIR_XAU  # backward-compat alias
# Production core pairs: EURUSD, GBPUSD, XAUUSD (officially adopted 2026-06)
DBBS_CORE_PAIRS = frozenset({DBBS_PAIR_PRIMARY, DBBS_PAIR_SECONDARY, DBBS_PAIR_XAU})
DBBS_EXTENDED_PAIRS = frozenset({"USDJPY", "AUDUSD", "AUDJPY", "NZDUSD"})
ALLOWED_PAIRS = frozenset({*DBBS_CORE_PAIRS, *DBBS_EXTENDED_PAIRS})
EXEC_BAR_MINUTES = 15
STRUCTURE_BAR_MINUTES = 60
DBBS_ATR_BAR_MINUTES = 240
BB_PERIOD_SHORT = 20
BB_PERIOD_LONG = 50
BB_STD_MULT = 2.0
ATR_PERIOD = 14
BB_WIDTH_LOOKBACK = 100
ADR_LOOKBACK_DAYS = 20
MIN_RR_RATIO = 1.5
MIN_RISK_PIPS = 3.0
MIN_RISK_ATR_FRAC = 0.05
MAX_SIM_RESULT_R = 50.0
DBBS_MAX_LOSS_R = 1.0
BB50_SLOPE_LOOKBACK = 3
FLAT_SLOPE_PIPS = 0.35
TOUCH_ATR_MULT = 0.3
WALK_ATR_MULT = 0.1
SQUEEZE_MAX_HOLD_H1 = 48
SQUEEZE_MIN_HOLD_H1 = 3
DBBS_BAYES_PURE_PROB = 1.0

# Bear Kill Switch V2 — default production risk control (see strategies/dbbs_bear_kill_switch.py)
BEAR_KILL_SWITCH_V2_ENABLED_DEFAULT = True
BEAR_KILL_SWITCH_V2_THRESHOLD_DEFAULT = 0.20

Direction = Literal["BUY", "SELL"]


def is_dbbs_pure_data_mode() -> bool:
    raw = os.getenv("DBBS_PURE_DATA_MODE") or os.getenv("DBB_PURE_DATA_MODE", "1")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def is_dbbs_enabled() -> bool:
    raw = os.getenv("DBBS_ENABLED", "1")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def is_dbbs_defense_mode() -> bool:
    """
    DBBS canonical production (default when not pure-data research).

    - L0-L2 / L4.5-L6: ON
    - L3.5 generic BayesEngine: OFF
    - L4 Gemini: OFF
    - Bear Kill Switch V2: ON (default)
    - 3 safety brakes: ON (Profit Cushion / Twin Brake / DD Throttling)
    """
    if is_dbbs_pure_data_mode():
        return False
    flag = os.getenv("DBBS_DEFENSE", "1").strip().lower()
    return flag in ("1", "true", "yes", "on")


def is_dbbs_defense_pure_mode() -> bool:
    """DBBS 防御レイヤー純粋モード（``DBBS_PURE_DATA_MODE=1``）。"""
    return is_dbbs_pure_data_mode()


def is_dbbs_l4_bypass() -> bool:
    """DBBS: Gemini L4 監査は常に無効（pure-data / canonical defense）。"""
    return is_dbbs_pure_data_mode() or is_dbbs_defense_mode()


def is_dbbs_generic_bayes_bypass() -> bool:
    """Generic L3.5 BayesEngine をスキップ（DBBS 本番）。"""
    return is_dbbs_pure_data_mode() or is_dbbs_defense_mode()


def configure_dbbs_defense_env() -> None:
    """Apply DBBS canonical defense defaults (idempotent)."""
    if is_dbbs_pure_data_mode():
        return
    os.environ.setdefault("DBBS_DEFENSE", "1")
    os.environ.setdefault("DBBS_BEAR_KILL_SWITCH", "1")
    os.environ.setdefault("DBBS_BEAR_KILL_SWITCH_THRESHOLD", "0.20")
    os.environ.setdefault("DBBS_PURE_DATA_MODE", "0")
    os.environ.setdefault("PROFIT_CUSHION_ENABLED", "1")
    os.environ.setdefault("TWIN_BRAKE_ENABLED", "1")
    os.environ.setdefault("DD_THROTTLING_ENABLED", "1")


class BandMatrix:
    """2本BBの状態分類（3値）"""

    SYNC = "SYNC"
    NESTED = "NESTED"
    EXPANSION = "EXPANSION"


@dataclass
class DbbsBayesFeatures:
    band_matrix: str = ""
    bb20_upper: float = 0.0
    bb20_middle: float = 0.0
    bb20_lower: float = 0.0
    bb20_width: float = 0.0
    bb20_width_atr_ratio: float = 0.0
    bb20_width_percentile: float = 0.0
    bb50_upper: float = 0.0
    bb50_middle: float = 0.0
    bb50_lower: float = 0.0
    bb50_slope: float = 0.0
    bb50_slope_direction: str = "FLAT"
    bb50_acceleration: float = 0.0
    price_vs_bb50_middle: float = 0.0
    bb20_vs_bb50_ratio: float = 0.0
    touch_zone: str = "OTHER"
    bb_walk_duration: int = 0
    golden_cross_bars_ago: int = -1
    squeeze_duration_bars: int = 0
    breakout_simultaneous: bool = False
    adr_used: float = 0.0
    adr_remaining: float = 0.0
    adr_percentile: float = 0.0
    adr_expansion_ratio: float = 0.0
    current_atr: float = 0.0
    volatility_percentile: float = 0.0
    session_type: str = "ASIA"
    minutes_from_session_open: int = 0
    outcome_label: str = ""
    result_r: float = 0.0
    mfe: float = 0.0
    mae: float = 0.0
    rr_ratio: float = 0.0
    signal_type: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "band_matrix": self.band_matrix,
            "signal_type": self.signal_type,
            "bb20_upper": self.bb20_upper,
            "bb20_middle": self.bb20_middle,
            "bb20_lower": self.bb20_lower,
            "bb20_width": self.bb20_width,
            "bb20_width_atr_ratio": self.bb20_width_atr_ratio,
            "bb20_width_percentile": self.bb20_width_percentile,
            "bb50_upper": self.bb50_upper,
            "bb50_middle": self.bb50_middle,
            "bb50_lower": self.bb50_lower,
            "bb50_slope": self.bb50_slope,
            "bb50_slope_direction": self.bb50_slope_direction,
            "bb50_acceleration": self.bb50_acceleration,
            "price_vs_bb50_middle": self.price_vs_bb50_middle,
            "bb20_vs_bb50_ratio": self.bb20_vs_bb50_ratio,
            "touch_zone": self.touch_zone,
            "bb_walk_duration": self.bb_walk_duration,
            "golden_cross_bars_ago": self.golden_cross_bars_ago,
            "squeeze_duration_bars": self.squeeze_duration_bars,
            "breakout_simultaneous": self.breakout_simultaneous,
            "adr_used": self.adr_used,
            "adr_remaining": self.adr_remaining,
            "adr_percentile": self.adr_percentile,
            "adr_expansion_ratio": self.adr_expansion_ratio,
            "current_atr": self.current_atr,
            "volatility_percentile": self.volatility_percentile,
            "session_type": self.session_type,
            "minutes_from_session_open": self.minutes_from_session_open,
            "outcome_label": self.outcome_label,
            "result_r": self.result_r,
            "mfe": self.mfe,
            "mae": self.mae,
            "rr_ratio": self.rr_ratio,
            "bayes_probability": DBBS_BAYES_PURE_PROB,
        }


DBBS_L6_EXTRA_COLUMNS: tuple[str, ...] = (
    "band_matrix",
    "signal_type",
    "bb20_width_percentile",
    "bb50_slope",
    "bb50_acceleration",
    "squeeze_duration_bars",
    "golden_cross_bars_ago",
    "bb_walk_duration",
    "touch_zone",
    "adr_percentile",
    "adr_expansion_ratio",
)

DBBS_FEATURE_COLUMNS: tuple[str, ...] = (
    "trade_id",
    "timestamp",
    "pair",
    "direction",
    *DBBS_L6_EXTRA_COLUMNS,
    "bb20_width_atr_ratio",
    "bb50_slope_direction",
    "breakout_simultaneous",
    "adr_used",
    "adr_remaining",
    "current_atr",
    "volatility_percentile",
    "session_type",
    "minutes_from_session_open",
    "bayes_probability",
    "decision_source",
    "executed",
    "trade_result",
    "profit_r",
    "result_r",
    "outcome_label",
    "mfe",
    "mae",
    "rr_ratio",
    "last_3_avg_r",
    "edge_risk_mult",
    "bear_kill_switch_active",
)


@dataclass(frozen=True)
class DbbsSetupBase:
    timestamp: pd.Timestamp
    pair: str
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    bar_index_h1: int
    bar_index_m15: int
    signal_type: str
    bayes_features: DbbsBayesFeatures
    rr_ratio: float
    bayes_probability: float = DBBS_BAYES_PURE_PROB


def compute_rr_ratio(entry: float, stop: float, take: float) -> float:
    risk = abs(entry - stop)
    if risk <= 0.0:
        return 0.0
    return abs(take - entry) / risk


def passes_min_rr(entry: float, stop: float, take: float, *, min_rr: float = MIN_RR_RATIO) -> bool:
    return compute_rr_ratio(entry, stop, take) >= min_rr


def min_risk_distance(atr: float, pip: float) -> float:
    """Minimum entry-to-stop distance for stable R-multiple simulation."""
    pip_dist = pip * MIN_RISK_PIPS
    atr_dist = atr * MIN_RISK_ATR_FRAC if atr > 0.0 else 0.0
    return max(pip_dist, atr_dist, pip)


def is_valid_stop_side(direction: Direction, entry: float, stop: float) -> bool:
    if direction == "BUY":
        return stop < entry
    return stop > entry


def is_valid_take_side(direction: Direction, entry: float, take: float) -> bool:
    if direction == "BUY":
        return take > entry
    return take < entry


def passes_entry_stop_take_geometry(
    direction: Direction,
    entry: float,
    stop: float,
    take: float,
    *,
    atr: float,
    pip: float,
    min_rr: float = MIN_RR_RATIO,
) -> bool:
    if not is_valid_stop_side(direction, entry, stop):
        return False
    if not is_valid_take_side(direction, entry, take):
        return False
    if abs(entry - stop) < min_risk_distance(atr, pip):
        return False
    return passes_min_rr(entry, stop, take, min_rr=min_rr)


def clamp_result_r(result_r: float, *, max_abs: float = MAX_SIM_RESULT_R) -> float:
    cap = max(float(max_abs), 0.0)
    return max(-cap, min(cap, float(result_r)))


def ohlcv_to_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from strategies.bt_ohlcv import lookup_ohlcv

    found = lookup_ohlcv(df)
    if found is not None:
        return found.open, found.high, found.low, found.close, found.volume
    vol = np.asarray(df["volume"], dtype=np.float64) if "volume" in df.columns else np.zeros(len(df))
    return (
        np.asarray(df["open"], dtype=np.float64),
        np.asarray(df["high"], dtype=np.float64),
        np.asarray(df["low"], dtype=np.float64),
        np.asarray(df["close"], dtype=np.float64),
        vol,
    )


def clip_arrays_as_of(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    end_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    end = max(0, min(int(end_index) + 1, len(close)))
    return open_[:end], high[:end], low[:end], close[:end], volume[:end]


@njit(cache=True)
def _sma_at(close: np.ndarray, idx: int, period: int) -> float:
    if idx + 1 < period:
        return 0.0
    total = 0.0
    for j in range(idx - period + 1, idx + 1):
        total += close[j]
    return total / period


@njit(cache=True)
def _std_at(close: np.ndarray, idx: int, period: int, mean: float) -> float:
    if idx + 1 < period:
        return 0.0
    var = 0.0
    for j in range(idx - period + 1, idx + 1):
        d = close[j] - mean
        var += d * d
    return (var / period) ** 0.5


@njit(cache=True)
def bollinger_at(close: np.ndarray, idx: int, period: int, std_mult: float) -> tuple:
    mid = _sma_at(close, idx, period)
    if mid == 0.0 and idx + 1 < period:
        return 0.0, 0.0, 0.0
    std = _std_at(close, idx, period, mid)
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower


@njit(cache=True)
def atr_at(high: np.ndarray, low: np.ndarray, close: np.ndarray, idx: int, period: int) -> float:
    if idx + 1 < period + 1:
        return 0.0
    total = 0.0
    for j in range(idx - period + 1, idx + 1):
        prev_close = close[j - 1] if j > 0 else close[j]
        tr = max(high[j] - low[j], abs(high[j] - prev_close), abs(low[j] - prev_close))
        total += tr
    return total / period


@njit(cache=True)
def percentile_in_window(values: np.ndarray, idx: int, lookback: int) -> float:
    if idx < 0:
        return 0.0
    start = max(0, idx - lookback + 1)
    current = values[idx]
    count = 0
    le = 0
    for j in range(start, idx + 1):
        v = values[j]
        if v > 0.0 or v == 0.0:
            count += 1
            if v <= current:
                le += 1
    if count <= 1:
        return 0.5
    return le / count


@njit(cache=True)
def bb50_slope_pips_at(
    bb50_middle: np.ndarray,
    idx: int,
    lookback: int,
    pip_size: float,
) -> float:
    if idx < lookback or pip_size <= 0.0:
        return 0.0
    return (bb50_middle[idx] - bb50_middle[idx - lookback]) / (lookback * pip_size)


@njit(cache=True)
def slope_direction(slope_pips: float, flat_threshold: float) -> int:
    if slope_pips > flat_threshold:
        return 1
    if slope_pips < -flat_threshold:
        return -1
    return 0


@njit(cache=True)
def is_nested_at(
    bb20_upper: float,
    bb20_lower: float,
    bb50_upper: float,
    bb50_lower: float,
) -> bool:
    return bb20_upper <= bb50_upper and bb20_lower >= bb50_lower


@njit(cache=True)
def classify_band_matrix_at(
    bb20_upper: float,
    bb20_lower: float,
    bb20_width: float,
    bb50_upper: float,
    bb50_lower: float,
    bb50_width: float,
    bb50_slope_pips: float,
    flat_threshold: float,
) -> int:
    if is_nested_at(bb20_upper, bb20_lower, bb50_upper, bb50_lower):
        return 1
    direction = slope_direction(bb50_slope_pips, flat_threshold)
    if direction == 0:
        if bb50_width > 0.0 and bb20_width / bb50_width > 1.15:
            return 2
        return 1
    if bb50_width > 0.0 and bb20_width >= bb50_width * 0.85:
        if direction > 0 and bb20_lower >= bb50_lower:
            return 0
        if direction < 0 and bb20_upper <= bb50_upper:
            return 0
    if direction == 0 and bb50_width > 0.0 and bb20_width / bb50_width > 1.15:
        return 2
    return 2 if direction == 0 else 0


@njit(cache=True)
def count_squeeze_duration(
    band_codes: np.ndarray,
    idx: int,
    nested_code: int,
) -> int:
    count = 0
    j = idx - 1
    while j >= 0:
        if band_codes[j] != nested_code:
            break
        count += 1
        j -= 1
    return count


@njit(cache=True)
def golden_cross_bars_ago_at(bb20_mid: np.ndarray, bb50_mid: np.ndarray, idx: int) -> int:
    if idx <= 0:
        return -1
    for j in range(idx, 0, -1):
        prev_diff = bb20_mid[j - 1] - bb50_mid[j - 1]
        curr_diff = bb20_mid[j] - bb50_mid[j]
        if prev_diff <= 0.0 and curr_diff > 0.0:
            return idx - j
        if prev_diff >= 0.0 and curr_diff < 0.0:
            return idx - j
    return -1


@njit(cache=True)
def bb_walk_duration_at(
    close: np.ndarray,
    bb20_upper: np.ndarray,
    bb20_lower: np.ndarray,
    atr: np.ndarray,
    idx: int,
    direction: int,
    walk_atr_mult: float,
) -> int:
    count = 0
    j = idx
    while j >= 0:
        band = bb20_upper[j] if direction > 0 else bb20_lower[j]
        tol = atr[j] * walk_atr_mult if atr[j] > 0.0 else 0.0
        if abs(close[j] - band) <= tol:
            count += 1
            j -= 1
        else:
            break
    return count


@njit(cache=True)
def breakout_simultaneous_at(
    close_val: float,
    bb20_upper: float,
    bb20_lower: float,
    bb50_upper: float,
    bb50_lower: float,
    direction: int,
) -> bool:
    if direction > 0:
        return close_val > bb20_upper and close_val > bb50_upper
    return close_val < bb20_lower and close_val < bb50_lower


@njit(cache=True)
def near_level(price: float, level: float, tol: float) -> bool:
    return abs(price - level) <= tol


@njit(cache=True)
def classify_touch_zone_buy(
    close_val: float,
    bb20_lower: float,
    bb50_middle: float,
    bb50_lower: float,
    tol: float,
) -> int:
    at_bb20 = near_level(close_val, bb20_lower, tol)
    at_bb50_mid = near_level(close_val, bb50_middle, tol)
    at_bb50_low = near_level(close_val, bb50_lower, tol)
    if at_bb20 and at_bb50_mid:
        return 0
    if at_bb20:
        return 1
    if at_bb50_mid:
        return 2
    if at_bb50_low:
        return 3
    return 4


@njit(cache=True)
def classify_touch_zone_sell(
    close_val: float,
    bb20_upper: float,
    bb50_middle: float,
    bb50_upper: float,
    tol: float,
) -> int:
    at_bb20 = near_level(close_val, bb20_upper, tol)
    at_bb50_mid = near_level(close_val, bb50_middle, tol)
    at_bb50_up = near_level(close_val, bb50_upper, tol)
    if at_bb20 and at_bb50_mid:
        return 0
    if at_bb20:
        return 1
    if at_bb50_mid:
        return 2
    if at_bb50_up:
        return 3
    return 4


TOUCH_ZONE_BUY_LABELS = (
    "BB20_LOWER_BB50_MIDDLE",
    "BB20_LOWER_ONLY",
    "BB50_MIDDLE_ONLY",
    "BB50_LOWER",
    "OTHER",
)
TOUCH_ZONE_SELL_LABELS = (
    "BB20_UPPER_BB50_MIDDLE",
    "BB20_UPPER_ONLY",
    "BB50_MIDDLE_ONLY",
    "BB50_UPPER",
    "OTHER",
)


@njit(cache=True)
def precompute_bb_series(close: np.ndarray, period: int, std_mult: float) -> tuple:
    n = len(close)
    upper = np.zeros(n, dtype=np.float64)
    middle = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    width = np.zeros(n, dtype=np.float64)
    for i in range(n):
        u, m, l = bollinger_at(close, i, period, std_mult)
        upper[i] = u
        middle[i] = m
        lower[i] = l
        width[i] = max(u - l, 0.0)
    return upper, middle, lower, width


@njit(cache=True)
def precompute_atr_series(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        out[i] = atr_at(high, low, close, i, period)
    return out


@njit(cache=True)
def precompute_band_codes(
    bb20_upper: np.ndarray,
    bb20_lower: np.ndarray,
    bb20_width: np.ndarray,
    bb50_upper: np.ndarray,
    bb50_lower: np.ndarray,
    bb50_width: np.ndarray,
    bb50_middle: np.ndarray,
    pip_size: float,
    flat_threshold: float,
    lookback: int,
) -> np.ndarray:
    n = len(bb20_upper)
    codes = np.zeros(n, dtype=np.int8)
    for i in range(n):
        slope = bb50_slope_pips_at(bb50_middle, i, lookback, pip_size) if i >= lookback else 0.0
        codes[i] = classify_band_matrix_at(
            bb20_upper[i],
            bb20_lower[i],
            bb20_width[i],
            bb50_upper[i],
            bb50_lower[i],
            bb50_width[i],
            slope,
            flat_threshold,
        )
    return codes


_BAND_CODE_TO_LABEL = {0: BandMatrix.SYNC, 1: BandMatrix.NESTED, 2: BandMatrix.EXPANSION}


def band_matrix_label(code: int) -> str:
    return _BAND_CODE_TO_LABEL.get(int(code), BandMatrix.EXPANSION)


def resolve_session(hour: int, minute: int) -> tuple[str, int]:
    if hour < 8:
        return "ASIA", hour * 60 + minute
    if hour < 13:
        return "LONDON", (hour - 8) * 60 + minute
    return "NY", (hour - 13) * 60 + minute


def compute_adr_metrics(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    day_index: np.ndarray,
    idx: int,
    atr: np.ndarray,
    lookback_days: int = ADR_LOOKBACK_DAYS,
) -> tuple[float, float, float, float]:
    if idx < 0 or idx >= len(close):
        return 0.0, 0.0, 0.5, 1.0
    current_day = day_index[idx]
    day_high = high[idx]
    day_low = low[idx]
    for j in range(idx - 1, -1, -1):
        if day_index[j] != current_day:
            break
        if high[j] > day_high:
            day_high = high[j]
        if low[j] < day_low:
            day_low = low[j]
    today_range = max(day_high - day_low, 0.0)

    ranges: list[float] = []
    atr_samples: list[float] = []
    seen_days: set[int] = set()
    j = idx
    while j >= 0 and len(seen_days) < lookback_days + 1:
        d = int(day_index[j])
        if d not in seen_days:
            seen_days.add(d)
            dh = high[j]
            dl = low[j]
            k = j - 1
            while k >= 0 and day_index[k] == d:
                if high[k] > dh:
                    dh = high[k]
                if low[k] < dl:
                    dl = low[k]
                k -= 1
            ranges.append(max(dh - dl, 0.0))
            if atr[j] > 0.0:
                atr_samples.append(float(atr[j]))
        j -= 1

    if not ranges:
        return 0.0, 0.0, 0.5, 1.0
    avg_range = float(np.mean(ranges))
    adr_used = today_range / avg_range if avg_range > 0.0 else 0.0
    adr_remaining = max(avg_range - today_range, 0.0)
    sorted_ranges = sorted(ranges)
    rank = sum(1 for r in sorted_ranges if r <= today_range)
    adr_percentile = rank / len(sorted_ranges) if sorted_ranges else 0.5
    avg_atr = float(np.mean(atr_samples)) if atr_samples else float(atr[idx])
    adr_expansion = float(atr[idx]) / avg_atr if avg_atr > 0.0 else 1.0
    return adr_used, adr_remaining, adr_percentile, adr_expansion


def day_index_from_timestamps(timestamps: np.ndarray) -> np.ndarray:
    if len(timestamps) == 0:
        return np.array([], dtype=np.int64)
    out = np.zeros(len(timestamps), dtype=np.int64)
    for i, ts in enumerate(timestamps):
        if isinstance(ts, (np.datetime64,)):
            t = pd.Timestamp(ts)
        else:
            t = pd.Timestamp(ts)
        out[i] = int(t.strftime("%Y%m%d"))
    return out


@dataclass
class DbbsStructureState:
    """H1 BB series + H4 ATR mapped onto each H1 bar index."""

    bb20_upper: np.ndarray = field(repr=False)
    bb20_middle: np.ndarray = field(repr=False)
    bb20_lower: np.ndarray = field(repr=False)
    bb20_width: np.ndarray = field(repr=False)
    bb50_upper: np.ndarray = field(repr=False)
    bb50_middle: np.ndarray = field(repr=False)
    bb50_lower: np.ndarray = field(repr=False)
    bb50_width: np.ndarray = field(repr=False)
    atr: np.ndarray = field(repr=False)  # H4 ATR(14) aligned to H1 bars
    band_codes: np.ndarray = field(repr=False)
    bb50_slope: np.ndarray = field(repr=False)
    pip_size: float = 0.0001


def map_htf_index(exec_datetime_ns: np.ndarray, htf_datetime_ns: np.ndarray) -> np.ndarray:
    """For each exec/structure bar, index of the last HTF bar with timestamp <= ts."""
    if htf_datetime_ns.size == 0:
        return np.full(exec_datetime_ns.shape[0], -1, dtype=np.int64)
    return np.searchsorted(htf_datetime_ns, exec_datetime_ns, side="right").astype(np.int64) - 1


def build_m15_to_h1_index(m15_df: pd.DataFrame, h1_df: pd.DataFrame) -> np.ndarray:
    """Vectorized M15 -> H1 bar index map (last H1 bar with datetime <= M15 datetime)."""
    m15_ns = _datetime_ns_from_df(m15_df)
    h1_ns = _datetime_ns_from_df(h1_df)
    return map_htf_index(m15_ns, h1_ns)


def _datetime_ns_from_df(df: pd.DataFrame) -> np.ndarray:
    from strategies.bt_ohlcv import as_ohlcv

    arr = as_ohlcv(df)
    return arr.datetime_ns


def resolve_h4_df(
    h1_df: pd.DataFrame,
    h4_df: pd.DataFrame | None,
    *,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return clipped H4 OHLCV; resample from H1 when native H4 is not supplied."""
    if h4_df is not None and len(h4_df) >= ATR_PERIOD + 2:
        clipped = clip_as_of(h4_df, as_of) if as_of is not None else h4_df
        return clipped
    from strategies.bt_ohlcv import BtOhlcvFrame, resample_bars_ns

    h1_clipped = clip_as_of(h1_df, as_of) if as_of is not None else h1_df
    h1_frame = BtOhlcvFrame.from_pandas(h1_clipped)
    bar_ns = int(np.timedelta64(DBBS_ATR_BAR_MINUTES, "m") / np.timedelta64(1, "ns"))
    return resample_bars_ns(h1_frame, bar_ns).to_pandas()


def compute_h4_atr_on_h1(h1_df: pd.DataFrame, h4_df: pd.DataFrame) -> np.ndarray:
    """Map H4 ATR(14) onto each H1 bar (no lookahead on H4 series)."""
    h1_ns = _datetime_ns_from_df(h1_df)
    h4_open, h4_high, h4_low, h4_close, _ = ohlcv_to_arrays(h4_df)
    atr_h4 = precompute_atr_series(h4_high, h4_low, h4_close, ATR_PERIOD)
    h4_ns = _datetime_ns_from_df(h4_df)
    mapped = np.zeros(len(h1_ns), dtype=np.float64)
    htf_idx = map_htf_index(h1_ns, h4_ns)
    for i in range(len(h1_ns)):
        j = int(htf_idx[i])
        if j >= 0 and atr_h4[j] > 0.0:
            mapped[i] = atr_h4[j]
    return mapped


def build_structure_state(
    h1_df: pd.DataFrame,
    *,
    pair: str,
    h4_df: pd.DataFrame | None = None,
    as_of: pd.Timestamp | None = None,
) -> DbbsStructureState | None:
    clipped = clip_as_of(h1_df, as_of) if as_of is not None else h1_df
    if clipped is None or len(clipped) < BB_PERIOD_LONG + BB50_SLOPE_LOOKBACK + 2:
        return None
    h4_clipped = resolve_h4_df(clipped, h4_df, as_of=as_of)
    if len(h4_clipped) < ATR_PERIOD + 2:
        return None
    open_, high, low, close, _vol = ohlcv_to_arrays(clipped)
    pip = pip_size_for_pair(pair)
    bb20_u, bb20_m, bb20_l, bb20_w = precompute_bb_series(close, BB_PERIOD_SHORT, BB_STD_MULT)
    bb50_u, bb50_m, bb50_l, bb50_w = precompute_bb_series(close, BB_PERIOD_LONG, BB_STD_MULT)
    atr_h4_on_h1 = compute_h4_atr_on_h1(clipped, h4_clipped)
    codes = precompute_band_codes(
        bb20_u, bb20_l, bb20_w, bb50_u, bb50_l, bb50_w, bb50_m, pip, FLAT_SLOPE_PIPS, BB50_SLOPE_LOOKBACK
    )
    n = len(close)
    slopes = np.zeros(n, dtype=np.float64)
    for i in range(n):
        slopes[i] = bb50_slope_pips_at(bb50_m, i, BB50_SLOPE_LOOKBACK, pip) if i >= BB50_SLOPE_LOOKBACK else 0.0
    return DbbsStructureState(
        bb20_upper=bb20_u,
        bb20_middle=bb20_m,
        bb20_lower=bb20_l,
        bb20_width=bb20_w,
        bb50_upper=bb50_u,
        bb50_middle=bb50_m,
        bb50_lower=bb50_l,
        bb50_width=bb50_w,
        atr=atr_h4_on_h1,
        band_codes=codes,
        bb50_slope=slopes,
        pip_size=pip,
    )


def build_dbbs_features_at(
    *,
    state: DbbsStructureState,
    h1_open: np.ndarray,
    h1_high: np.ndarray,
    h1_low: np.ndarray,
    h1_close: np.ndarray,
    h1_timestamps: np.ndarray,
    h1_day_index: np.ndarray,
    idx: int,
    pair: str,
    direction: Direction,
    signal_type: str,
    breakout_sim: bool | None = None,
) -> DbbsBayesFeatures:
    pip = state.pip_size
    atr_val = float(state.atr[idx])
    bb20_u = float(state.bb20_upper[idx])
    bb20_m = float(state.bb20_middle[idx])
    bb20_l = float(state.bb20_lower[idx])
    bb20_w = float(state.bb20_width[idx])
    bb50_u = float(state.bb50_upper[idx])
    bb50_m = float(state.bb50_middle[idx])
    bb50_l = float(state.bb50_lower[idx])
    bb50_w = float(state.bb50_width[idx])
    close_val = float(h1_close[idx])

    code = int(state.band_codes[idx])
    slope = float(state.bb50_slope[idx])
    dir_int = 1 if direction == "BUY" else -1
    slope_dir = slope_direction(slope, FLAT_SLOPE_PIPS)
    slope_label = "UP" if slope_dir > 0 else "DOWN" if slope_dir < 0 else "FLAT"

    accel = 0.0
    if idx >= BB50_SLOPE_LOOKBACK + 1:
        prev_slope = bb50_slope_pips_at(state.bb50_middle, idx - 1, BB50_SLOPE_LOOKBACK, pip)
        accel = slope - prev_slope

    width_pct = percentile_in_window(state.bb20_width, idx, BB_WIDTH_LOOKBACK)
    vol_pct = percentile_in_window(state.atr, idx, BB_WIDTH_LOOKBACK)
    bb20_atr_ratio = bb20_w / atr_val if atr_val > 0.0 else 0.0
    bb20_bb50_ratio = bb20_w / bb50_w if bb50_w > 0.0 else 0.0
    price_vs_mid = (close_val - bb50_m) / bb50_w if bb50_w > 0.0 else 0.0

    tol = atr_val * TOUCH_ATR_MULT if atr_val > 0.0 else pip * 3.0
    if direction == "BUY":
        tz_code = classify_touch_zone_buy(close_val, bb20_l, bb50_m, bb50_l, tol)
        touch_zone = TOUCH_ZONE_BUY_LABELS[tz_code]
    else:
        tz_code = classify_touch_zone_sell(close_val, bb20_u, bb50_m, bb50_u, tol)
        touch_zone = TOUCH_ZONE_SELL_LABELS[tz_code]

    walk = bb_walk_duration_at(h1_close, state.bb20_upper, state.bb20_lower, state.atr, idx, dir_int, WALK_ATR_MULT)
    gc_ago = golden_cross_bars_ago_at(state.bb20_middle, state.bb50_middle, idx)
    squeeze_dur = count_squeeze_duration(state.band_codes, idx, 1)

    ts = h1_timestamps[idx]
    t = pd.Timestamp(ts)
    session, mins_open = resolve_session(int(t.hour), int(t.minute))

    adr_used, adr_rem, adr_pct, adr_exp = compute_adr_metrics(
        h1_high, h1_low, h1_close, h1_day_index, idx, state.atr, ADR_LOOKBACK_DAYS
    )

    if breakout_sim is None:
        breakout_sim = breakout_simultaneous_at(close_val, bb20_u, bb20_l, bb50_u, bb50_l, dir_int)

    return DbbsBayesFeatures(
        band_matrix=band_matrix_label(code),
        bb20_upper=bb20_u,
        bb20_middle=bb20_m,
        bb20_lower=bb20_l,
        bb20_width=bb20_w,
        bb20_width_atr_ratio=bb20_atr_ratio,
        bb20_width_percentile=float(width_pct),
        bb50_upper=bb50_u,
        bb50_middle=bb50_m,
        bb50_lower=bb50_l,
        bb50_slope=slope,
        bb50_slope_direction=slope_label,
        bb50_acceleration=float(accel),
        price_vs_bb50_middle=float(price_vs_mid),
        bb20_vs_bb50_ratio=float(bb20_bb50_ratio),
        touch_zone=touch_zone,
        bb_walk_duration=int(walk),
        golden_cross_bars_ago=int(gc_ago),
        squeeze_duration_bars=int(squeeze_dur),
        breakout_simultaneous=bool(breakout_sim),
        adr_used=float(adr_used),
        adr_remaining=float(adr_rem),
        adr_percentile=float(adr_pct),
        adr_expansion_ratio=float(adr_exp),
        current_atr=atr_val,
        volatility_percentile=float(vol_pct),
        session_type=session,
        minutes_from_session_open=int(mins_open),
        signal_type=signal_type,
    )


def map_m15_index_to_h1(m15_ts: pd.Timestamp, h1_timestamps: np.ndarray) -> int:
    target = pd.Timestamp(m15_ts)
    if target.tzinfo is not None:
        target = target.tz_localize(None)
    h1_ns = h1_timestamps.astype("datetime64[ns]")
    idx = int(np.searchsorted(h1_ns, np.datetime64(target), side="right") - 1)
    return idx


def build_dbbs_feature_log_row(
    *,
    trade_id: str,
    setup: DbbsSetupBase,
    trade_result: str,
    profit_r: float,
    result_r: float | None = None,
    decision_source: str = "ALLOW",
    executed: bool = True,
    last_3_avg_r: float | None = None,
    edge_risk_mult: float | None = None,
    bear_kill_switch_active: bool | None = None,
) -> dict[str, Any]:
    row = setup.bayes_features.as_dict()
    raw_result = float(result_r if result_r is not None else profit_r)
    row.update(
        {
            "trade_id": trade_id,
            "timestamp": setup.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "pair": setup.pair,
            "direction": setup.direction,
            "decision_source": decision_source,
            "executed": executed,
            "trade_result": trade_result,
            "profit_r": round(float(profit_r), 4),
            "result_r": round(raw_result, 4),
            "outcome_label": trade_result if trade_result in ("WIN", "LOSS") else setup.bayes_features.outcome_label,
            "rr_ratio": round(float(setup.rr_ratio), 4),
        }
    )
    if last_3_avg_r is not None and np.isfinite(last_3_avg_r):
        row["last_3_avg_r"] = round(float(last_3_avg_r), 4)
    else:
        row["last_3_avg_r"] = ""
    if edge_risk_mult is not None:
        row["edge_risk_mult"] = round(float(edge_risk_mult), 4)
    if bear_kill_switch_active is not None:
        row["bear_kill_switch_active"] = bool(bear_kill_switch_active)
    return {k: row.get(k, "") for k in DBBS_FEATURE_COLUMNS}
