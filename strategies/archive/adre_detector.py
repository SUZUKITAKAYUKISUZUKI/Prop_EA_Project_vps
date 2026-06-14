"""
strategies/adre_detector.py — ADR Expansion (ADRE) Pure Discovery detection.

H1 exec/structure, 24h session, prior-day high/low breakout only.
No filters, scoring, or ML — maximize sample count for ADR edge discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from strategies.htf_trend_analyzer import clip_as_of
from strategies.market_utils import correlated_pair, pip_size_for_pair

ADR_LOOKBACK_DAYS = 20
ATR_PERIOD = 14
RR_RATIO = 2.0
SL_ATR_MULT = 1.0
Direction = Literal["BUY", "SELL"]
ADRE_PAIR_PRIMARY = "EURUSD"
ADRE_PAIR_SECONDARY = "GBPUSD"
ADRE_PAIRS = frozenset({ADRE_PAIR_PRIMARY, ADRE_PAIR_SECONDARY})
JST = timezone(timedelta(hours=9))
UTC = timezone.utc


def day_index_from_timestamps(timestamps: np.ndarray) -> np.ndarray:
    if len(timestamps) == 0:
        return np.array([], dtype=np.int64)
    out = np.zeros(len(timestamps), dtype=np.int64)
    for i, ts in enumerate(timestamps):
        t = pd.Timestamp(ts)
        out[i] = int(t.strftime("%Y%m%d"))
    return out


def compute_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = len(close)
    tr = np.zeros(n, dtype=np.float64)
    if n == 0:
        return tr
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    return tr


def compute_atr_series(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = ATR_PERIOD,
) -> np.ndarray:
    tr = compute_true_range(high, low, close)
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    if n < period:
        return atr
    atr[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def daily_range_for_day(
    high: np.ndarray,
    low: np.ndarray,
    day_index: np.ndarray,
    day: int,
    *,
    end_idx: int | None = None,
) -> float:
    end = len(high) if end_idx is None else min(end_idx, len(high) - 1) + 1
    dh = -np.inf
    dl = np.inf
    found = False
    for i in range(end):
        if int(day_index[i]) != day:
            continue
        found = True
        if high[i] > dh:
            dh = high[i]
        if low[i] < dl:
            dl = low[i]
    if not found:
        return 0.0
    return max(float(dh - dl), 0.0)


def completed_daily_ranges_before(
    high: np.ndarray,
    low: np.ndarray,
    day_index: np.ndarray,
    before_day: int,
    *,
    end_idx: int,
    lookback_days: int = ADR_LOOKBACK_DAYS,
) -> list[float]:
    """Past completed calendar days strictly before ``before_day`` (no lookahead)."""
    seen: dict[int, tuple[float, float]] = {}
    for i in range(end_idx + 1):
        d = int(day_index[i])
        if d >= before_day:
            continue
        if d not in seen:
            seen[d] = (high[i], low[i])
        else:
            dh, dl = seen[d]
            seen[d] = (max(dh, high[i]), min(dl, low[i]))
    days = sorted(seen.keys(), reverse=True)
    ranges: list[float] = []
    for d in days:
        dh, dl = seen[d]
        ranges.append(max(dh - dl, 0.0))
        if len(ranges) >= lookback_days:
            break
    return ranges


def compute_adr20_at(
    high: np.ndarray,
    low: np.ndarray,
    day_index: np.ndarray,
    idx: int,
    *,
    lookback_days: int = ADR_LOOKBACK_DAYS,
) -> float:
    if idx < 0:
        return 0.0
    current_day = int(day_index[idx])
    ranges = completed_daily_ranges_before(
        high, low, day_index, current_day, end_idx=idx, lookback_days=lookback_days
    )
    if not ranges:
        return 0.0
    return float(np.mean(ranges))


def previous_day_extremes(
    high: np.ndarray,
    low: np.ndarray,
    day_index: np.ndarray,
    idx: int,
) -> tuple[float | None, float | None]:
    if idx <= 0:
        return None, None
    current_day = int(day_index[idx])
    prev_day: int | None = None
    for j in range(idx - 1, -1, -1):
        d = int(day_index[j])
        if d < current_day:
            prev_day = d
            break
    if prev_day is None:
        return None, None
    prev_high = -np.inf
    prev_low = np.inf
    for j in range(idx):
        if int(day_index[j]) != prev_day:
            continue
        if high[j] > prev_high:
            prev_high = high[j]
        if low[j] < prev_low:
            prev_low = low[j]
    if not np.isfinite(prev_high) or not np.isfinite(prev_low):
        return None, None
    return float(prev_high), float(prev_low)


def current_day_range_at(
    high: np.ndarray,
    low: np.ndarray,
    day_index: np.ndarray,
    idx: int,
) -> tuple[float, float, float]:
    current_day = int(day_index[idx])
    day_high = high[idx]
    day_low = low[idx]
    for j in range(idx - 1, -1, -1):
        if int(day_index[j]) != current_day:
            break
        if high[j] > day_high:
            day_high = high[j]
        if low[j] < day_low:
            day_low = low[j]
    day_high = float(day_high)
    day_low = float(day_low)
    return max(day_high - day_low, 0.0), day_high, day_low


def compute_adr_metrics_at(
    high: np.ndarray,
    low: np.ndarray,
    day_index: np.ndarray,
    idx: int,
    *,
    lookback_days: int = ADR_LOOKBACK_DAYS,
) -> tuple[float, float, float, float, float, float, float]:
    """
    Returns (adr20, adr_used, adr_remaining, adr_expansion_today, current_day_range, day_high, day_low).
    """
    adr20 = compute_adr20_at(high, low, day_index, idx, lookback_days=lookback_days)
    current_range, day_high, day_low = current_day_range_at(high, low, day_index, idx)
    if adr20 <= 0.0:
        return adr20, 0.0, 1.0, 0.0, current_range, day_high, day_low
    adr_used = current_range / adr20
    adr_remaining = 1.0 - adr_used
    adr_expansion = current_range / adr20
    return adr20, adr_used, adr_remaining, adr_expansion, current_range, day_high, day_low


def session_minutes_elapsed(ts: pd.Timestamp) -> int:
    return int(ts.hour) * 60 + int(ts.minute)


def breakout_hour_jst(ts: pd.Timestamp) -> int:
    """Signal bar hour in JST (H1 CSV timestamps treated as UTC)."""
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(UTC)
    return int(stamp.tz_convert(JST).hour)


def breakout_at(
    close: float,
    prev_high: float | None,
    prev_low: float | None,
) -> tuple[Direction | None, float]:
    if prev_high is not None and close > prev_high:
        return "BUY", close - prev_high
    if prev_low is not None and close < prev_low:
        return "SELL", prev_low - close
    return None, 0.0


def find_bar_index_for_timestamp(timestamps: np.ndarray, target_ts: pd.Timestamp) -> int | None:
    from strategies.bt_ohlcv import normalize_ts_ns

    target_ns = normalize_ts_ns(target_ts)
    for i, ts in enumerate(timestamps):
        if normalize_ts_ns(pd.Timestamp(ts)) == target_ns:
            return i
    return None


def compute_adre_smt_features(
    pair: str,
    primary_dir: Direction,
    primary_dist: float,
    corr_dir: Direction | None,
    corr_dist: float,
) -> tuple[bool, float, str]:
    """Compare prior-day breakout displacement vs correlated pair at signal bar."""
    pip = pip_size_for_pair(pair)
    primary_pips = primary_dist / pip if pip > 0 else 0.0
    corr_pips = corr_dist / pip if corr_dir is not None and pip > 0 else 0.0
    corr_pair = correlated_pair(pair)

    smt_divergence = corr_dir is not None and corr_dir != primary_dir
    smt_strength = round(abs(primary_pips - corr_pips), 4)

    if primary_pips > corr_pips:
        leader_pair = pair
    elif corr_pips > primary_pips:
        leader_pair = corr_pair
    else:
        leader_pair = "NONE"

    return smt_divergence, smt_strength, leader_pair


def resolve_adre_smt_at_signal(
    *,
    pair: str,
    sig_idx: int,
    signal_ts: pd.Timestamp,
    primary_dir: Direction,
    primary_dist: float,
    corr_high: np.ndarray | None,
    corr_low: np.ndarray | None,
    corr_close: np.ndarray | None,
    corr_timestamps: np.ndarray | None,
    corr_day_index: np.ndarray | None,
) -> tuple[bool, float, str]:
    if (
        corr_high is None
        or corr_low is None
        or corr_close is None
        or corr_timestamps is None
        or corr_day_index is None
    ):
        return False, 0.0, "NONE"

    corr_idx = find_bar_index_for_timestamp(corr_timestamps, signal_ts)
    if corr_idx is None:
        return False, 0.0, "NONE"

    corr_prev_high, corr_prev_low = previous_day_extremes(
        corr_high,
        corr_low,
        corr_day_index,
        corr_idx,
    )
    corr_dir, corr_dist = breakout_at(
        float(corr_close[corr_idx]),
        corr_prev_high,
        corr_prev_low,
    )
    return compute_adre_smt_features(pair, primary_dir, primary_dist, corr_dir, corr_dist)


def compute_sl_tp(
    direction: Direction,
    entry: float,
    atr: float,
    *,
    rr: float = RR_RATIO,
    sl_atr_mult: float = SL_ATR_MULT,
) -> tuple[float, float]:
    risk = max(float(atr) * sl_atr_mult, 0.0)
    if risk <= 0.0:
        return entry, entry
    if direction == "BUY":
        sl = entry - risk
        tp = entry + risk * rr
    else:
        sl = entry + risk
        tp = entry - risk * rr
    return sl, tp


def signal_direction_at(
    close: float,
    prev_high: float | None,
    prev_low: float | None,
) -> Direction | None:
    if prev_high is not None and close > prev_high:
        return "BUY"
    if prev_low is not None and close < prev_low:
        return "SELL"
    return None


@dataclass(frozen=True)
class AdreSignalContext:
    signal_idx: int
    entry_idx: int
    direction: Direction
    adr20: float
    adr_used: float
    adr_remaining: float
    adr_expansion_today: float
    current_day_range: float
    day_high: float
    day_low: float
    session_minutes_elapsed: int
    day_of_week: int
    month: int
    atr: float
    entry_price: float
    stop_loss: float
    take_profit: float
    smt_divergence: bool = False
    smt_strength: float = 0.0
    leader_pair: str = "NONE"
    breakout_hour_jst: int = 0


def scan_adre_signals(
    *,
    pair: str,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    timestamps: np.ndarray,
    day_index: np.ndarray,
    atr: np.ndarray,
    min_warmup_days: int = ADR_LOOKBACK_DAYS + 1,
    corr_high: np.ndarray | None = None,
    corr_low: np.ndarray | None = None,
    corr_close: np.ndarray | None = None,
    corr_timestamps: np.ndarray | None = None,
    corr_day_index: np.ndarray | None = None,
) -> list[AdreSignalContext]:
    """Scan H1 bars for ADRE signals. Entry on bar after signal (next open)."""
    n = len(close)
    if n < ATR_PERIOD + 2:
        return []

    min_idx = ATR_PERIOD
    unique_days = np.unique(day_index)
    if len(unique_days) < min_warmup_days:
        return []

    signals: list[AdreSignalContext] = []
    for sig_idx in range(min_idx, n - 1):
        adr20 = compute_adr20_at(high, low, day_index, sig_idx)
        if adr20 <= 0.0:
            continue
        prev_high, prev_low = previous_day_extremes(high, low, day_index, sig_idx)
        direction = signal_direction_at(float(close[sig_idx]), prev_high, prev_low)
        if direction is None:
            continue
        _break_dir, break_dist = breakout_at(float(close[sig_idx]), prev_high, prev_low)
        assert _break_dir == direction

        entry_idx = sig_idx + 1
        atr_val = float(atr[sig_idx])
        if atr_val <= 0.0:
            continue
        entry = float(open_[entry_idx])
        sl, tp = compute_sl_tp(direction, entry, atr_val)
        if direction == "BUY" and sl >= entry:
            continue
        if direction == "SELL" and sl <= entry:
            continue

        (
            _adr20,
            adr_used,
            adr_remaining,
            adr_expansion,
            current_range,
            day_high,
            day_low,
        ) = compute_adr_metrics_at(high, low, day_index, sig_idx)

        ts = pd.Timestamp(timestamps[sig_idx])
        smt_divergence, smt_strength, leader_pair = resolve_adre_smt_at_signal(
            pair=pair,
            sig_idx=sig_idx,
            signal_ts=ts,
            primary_dir=direction,
            primary_dist=break_dist,
            corr_high=corr_high,
            corr_low=corr_low,
            corr_close=corr_close,
            corr_timestamps=corr_timestamps,
            corr_day_index=corr_day_index,
        )
        signals.append(
            AdreSignalContext(
                signal_idx=sig_idx,
                entry_idx=entry_idx,
                direction=direction,
                adr20=adr20,
                adr_used=adr_used,
                adr_remaining=adr_remaining,
                adr_expansion_today=adr_expansion,
                current_day_range=current_range,
                day_high=day_high,
                day_low=day_low,
                session_minutes_elapsed=session_minutes_elapsed(ts),
                day_of_week=int(ts.dayofweek),
                month=int(ts.month),
                atr=atr_val,
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                smt_divergence=smt_divergence,
                smt_strength=smt_strength,
                leader_pair=leader_pair,
                breakout_hour_jst=breakout_hour_jst(ts),
            )
        )
    return signals


def detect_adre_setups_from_arrays(
    *,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    timestamps: np.ndarray,
    pair: str,
    corr_high: np.ndarray | None = None,
    corr_low: np.ndarray | None = None,
    corr_close: np.ndarray | None = None,
    corr_timestamps: np.ndarray | None = None,
) -> list[dict]:
    """Return setup dict payloads (converted to AdreSetup in adre.py)."""
    if pair not in ADRE_PAIRS:
        return []
    day_index = day_index_from_timestamps(timestamps)
    corr_day_index = (
        day_index_from_timestamps(corr_timestamps)
        if corr_timestamps is not None
        else None
    )
    atr = compute_atr_series(high, low, close, ATR_PERIOD)
    contexts = scan_adre_signals(
        pair=pair,
        open_=open_,
        high=high,
        low=low,
        close=close,
        timestamps=timestamps,
        day_index=day_index,
        atr=atr,
        corr_high=corr_high,
        corr_low=corr_low,
        corr_close=corr_close,
        corr_timestamps=corr_timestamps,
        corr_day_index=corr_day_index,
    )
    out: list[dict] = []
    for ctx in contexts:
        entry_ts = pd.Timestamp(timestamps[ctx.entry_idx])
        out.append(
            {
                "timestamp": entry_ts,
                "pair": pair,
                "direction": ctx.direction,
                "entry_price": ctx.entry_price,
                "stop_loss": ctx.stop_loss,
                "take_profit": ctx.take_profit,
                "adr20": ctx.adr20,
                "adr_used": ctx.adr_used,
                "adr_remaining": ctx.adr_remaining,
                "adr_expansion_today": ctx.adr_expansion_today,
                "day_high": ctx.day_high,
                "day_low": ctx.day_low,
                "current_day_range": ctx.current_day_range,
                "session_minutes_elapsed": ctx.session_minutes_elapsed,
                "day_of_week": ctx.day_of_week,
                "month": ctx.month,
                "smt_divergence": ctx.smt_divergence,
                "smt_strength": ctx.smt_strength,
                "leader_pair": ctx.leader_pair,
                "breakout_hour_jst": ctx.breakout_hour_jst,
                "signal_idx": ctx.signal_idx,
                "entry_idx": ctx.entry_idx,
            }
        )
    return out


def detect_adre_setups_clipped(
    h1_df: pd.DataFrame,
    pair: str,
    *,
    as_of: pd.Timestamp | None = None,
    corr_h1_df: pd.DataFrame | None = None,
) -> list[dict]:
    """Apply clip_as_of before detection (no future data)."""
    from strategies.bt_ohlcv import as_ohlcv

    clipped = clip_as_of(h1_df, as_of) if as_of is not None else h1_df
    if clipped is None or len(clipped) < ATR_PERIOD + 2:
        return []
    arr = as_ohlcv(clipped)
    corr_high = corr_low = corr_close = corr_timestamps = None
    if corr_h1_df is not None:
        corr_clipped = clip_as_of(corr_h1_df, as_of) if as_of is not None else corr_h1_df
        if corr_clipped is not None and len(corr_clipped) >= ATR_PERIOD + 2:
            corr_arr = as_ohlcv(corr_clipped)
            corr_high = corr_arr.high
            corr_low = corr_arr.low
            corr_close = corr_arr.close
            corr_timestamps = corr_arr.datetime_ns
    return detect_adre_setups_from_arrays(
        open_=arr.open,
        high=arr.high,
        low=arr.low,
        close=arr.close,
        timestamps=arr.datetime_ns,
        pair=pair,
        corr_high=corr_high,
        corr_low=corr_low,
        corr_close=corr_close,
        corr_timestamps=corr_timestamps,
    )
