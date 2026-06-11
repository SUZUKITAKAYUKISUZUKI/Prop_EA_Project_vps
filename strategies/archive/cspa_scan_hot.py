"""
strategies/cspa_scan_hot.py — NumPy-only per-bar helpers for CSPA detect_setups loop.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from strategies.archive.cspa_arrays import OhlcvArrays, atr_at_index, timestamp_from_ns, body_size

if TYPE_CHECKING:
    from strategies.archive.cspa import (
        ConsolidationZone,
        ImpulseLeg,
        MomentumSignal,
        PullbackRhythm,
        Reacceleration,
        StagnationCluster,
        StagnationQuality,
        SwingPoint,
        TradeDirection,
        TrendContext,
        TrendPhase,
    )


def swings_up_to(
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    up_to_bar_index: int,
    *,
    high_bar_indices: list[int],
    low_bar_indices: list[int],
) -> tuple[list[SwingPoint], list[SwingPoint]]:
    hi = bisect.bisect_right(high_bar_indices, up_to_bar_index)
    lo = bisect.bisect_right(low_bar_indices, up_to_bar_index)
    return swing_highs[:hi], swing_lows[:lo]


def prior_correction_ratio_np(
    structure: OhlcvArrays,
    impulse: ImpulseLeg,
    phase: TrendPhase,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    *,
    high_bar_indices: list[int],
    low_bar_indices: list[int],
) -> float | None:
    highs, lows = swings_up_to(
        swing_highs,
        swing_lows,
        impulse.start_index,
        high_bar_indices=high_bar_indices,
        low_bar_indices=low_bar_indices,
    )
    if phase == "UPTREND":
        if len(highs) < 2 or len(lows) < 2:
            return None
        prev_peak = highs[-2]
        prev_trough = lows[-2]
        size = prev_peak.price - prev_trough.price
        if size <= 0:
            return None
        start = prev_trough.bar_index
        end = prev_peak.bar_index + 1
        if start >= end:
            return None
        correction_low = float(np.min(structure.low[start:end]))
        return (prev_peak.price - correction_low) / size
    if len(lows) < 2 or len(highs) < 2:
        return None
    prev_trough = lows[-2]
    prev_peak = highs[-2]
    size = prev_peak.price - prev_trough.price
    if size <= 0:
        return None
    start = prev_peak.bar_index
    end = prev_trough.bar_index + 1
    if start >= end:
        return None
    correction_high = float(np.max(structure.high[start:end]))
    return (correction_high - prev_trough.price) / size


def correction_rhythm_ok_np(
    structure: OhlcvArrays,
    impulse: ImpulseLeg,
    phase: TrendPhase,
    current_ratio: float,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    *,
    high_bar_indices: list[int],
    low_bar_indices: list[int],
    max_ratio: float,
) -> bool:
    prev = prior_correction_ratio_np(
        structure,
        impulse,
        phase,
        swing_highs,
        swing_lows,
        high_bar_indices=high_bar_indices,
        low_bar_indices=low_bar_indices,
    )
    if prev is None or prev <= 0:
        return True
    return current_ratio <= prev * max_ratio


def volatility_percentile_np(
    atr: np.ndarray,
    bar_index: int,
    *,
    lookback: int,
    atr_period: int,
) -> float:
    if bar_index < atr_period or bar_index >= len(atr):
        return 0.5
    start = max(atr_period, bar_index - lookback + 1)
    window = atr[start : bar_index + 1]
    if len(window) == 0:
        return 0.5
    current = float(window[-1])
    if current <= 0:
        return 0.5
    below = int(np.sum(window <= current))
    return round(below / len(window), 4)


def observe_overlap_ratio_np(
    high: np.ndarray,
    low: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> float:
    if end_idx <= start_idx:
        return 0.0
    start = start_idx + 1
    end = end_idx + 1
    if start >= end:
        return 0.0
    prev_high = high[start_idx : end - 1]
    prev_low = low[start_idx : end - 1]
    cur_high = high[start:end]
    cur_low = low[start:end]
    overlap = np.maximum(
        0.0,
        np.minimum(prev_high, cur_high) - np.maximum(prev_low, cur_low),
    )
    return round(float(np.mean(overlap)), 6)


def observe_correction_smoothness_np(
    high: np.ndarray,
    low: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> float:
    if end_idx <= start_idx:
        return 0.5
    ranges = high[start_idx : end_idx + 1] - low[start_idx : end_idx + 1]
    if len(ranges) < 2:
        return 0.5
    mean_r = float(np.mean(ranges))
    if mean_r <= 0:
        return 0.5
    cv = float(np.std(ranges)) / mean_r
    return round(max(0.0, 1.0 - min(cv, 1.0)), 4)


def build_pullback_rhythm_np(
    structure: OhlcvArrays,
    structure_atr: np.ndarray,
    impulse: ImpulseLeg,
    struct_idx: int,
    retrace_ratio: float,
) -> PullbackRhythm:
    from strategies.archive.cspa import (
        composite_rhythm_score,
        observe_pullback_efficiency,
    )

    duration = max(1, struct_idx - impulse.end_index)
    atr = atr_at_index(structure_atr, struct_idx)
    overlap = observe_overlap_ratio_np(
        structure.high, structure.low, impulse.end_index, struct_idx
    )
    smoothness = observe_correction_smoothness_np(
        structure.high, structure.low, impulse.end_index, struct_idx
    )
    retrace_distance = retrace_ratio * impulse.impulse_size
    efficiency = observe_pullback_efficiency(retrace_distance, duration)
    rhythm = composite_rhythm_score(overlap, smoothness, efficiency, atr)
    from strategies.archive.cspa import PullbackRhythm

    return PullbackRhythm(
        duration_bars=duration,
        retracement_depth=round(retrace_ratio, 4),
        overlap_ratio=overlap,
        pullback_efficiency=efficiency,
        correction_smoothness=smoothness,
        rhythm_score=rhythm,
    )


def _empty_momentum(bar_index: int) -> MomentumSignal:
    from datetime import datetime, timezone

    from strategies.archive.cspa import MomentumSignal

    return MomentumSignal(
        detected=False,
        trigger_type="NONE",
        bar_index=bar_index,
        timestamp=datetime.now(timezone.utc),
        entry_price=0.0,
        trigger_high=0.0,
        trigger_low=0.0,
        body_atr=0.0,
        atr=0.0,
    )


def _ts_at(trigger: OhlcvArrays, bar_index: int):
    return timestamp_from_ns(int(trigger.datetime_ns[bar_index]))


def detect_momentum_breakout_np(
    trigger: OhlcvArrays,
    trigger_atr: np.ndarray,
    bar_index: int,
    direction: TradeDirection,
    stagnation: StagnationCluster,
    *,
    min_body_atr: float,
) -> MomentumSignal:
    from strategies.archive.cspa import MomentumSignal, MomentumType

    empty = _empty_momentum(bar_index)
    if bar_index < 1 or bar_index >= trigger.length:
        return empty

    open_ = float(trigger.open[bar_index])
    close = float(trigger.close[bar_index])
    high = float(trigger.high[bar_index])
    low = float(trigger.low[bar_index])
    prev_open = float(trigger.open[bar_index - 1])
    prev_close = float(trigger.close[bar_index - 1])
    atr = atr_at_index(trigger_atr, bar_index)
    body = body_size(open_, close)
    if atr <= 0 or body < min_body_atr * atr:
        return empty

    body_atr = body / atr
    trigger_type: MomentumType = "BODY_BREAK"
    if direction == "BUY":
        if close <= open_ or close <= stagnation.zone_high:
            return empty
        if prev_close < prev_open and close > open_ and close >= prev_open:
            trigger_type = "ENGULFING"
        if min(open_, close) - low >= 2.0 * body:
            trigger_type = "PIN_BAR"
    else:
        if close >= open_ or close >= stagnation.zone_low:
            return empty
        if prev_close > prev_open and close < open_ and close <= prev_open:
            trigger_type = "ENGULFING"
        if high - max(open_, close) >= 2.0 * body:
            trigger_type = "PIN_BAR"

    return MomentumSignal(
        detected=True,
        trigger_type=trigger_type,
        bar_index=bar_index,
        timestamp=_ts_at(trigger, bar_index),
        entry_price=close,
        trigger_high=high,
        trigger_low=low,
        body_atr=round(body_atr, 4),
        atr=atr,
    )


def detect_sweep_engulfing_np(
    trigger: OhlcvArrays,
    trigger_atr: np.ndarray,
    bar_index: int,
    direction: TradeDirection,
    stagnation: StagnationCluster | None,
    *,
    min_body_atr: float,
    min_range_atr: float,
    min_outside_ratio: float,
) -> MomentumSignal:
    from strategies.archive.cspa import MomentumSignal

    empty = _empty_momentum(bar_index)
    if bar_index < 1 or bar_index >= trigger.length:
        return empty

    open_ = float(trigger.open[bar_index])
    close = float(trigger.close[bar_index])
    high = float(trigger.high[bar_index])
    low = float(trigger.low[bar_index])
    prev_high = float(trigger.high[bar_index - 1])
    prev_low = float(trigger.low[bar_index - 1])
    atr = atr_at_index(trigger_atr, bar_index)
    body = body_size(open_, close)
    if atr <= 0 or body < min_body_atr * atr:
        return empty

    bar_range = high - low
    prev_range = prev_high - prev_low
    if bar_range < min_range_atr * atr:
        return empty
    if prev_range > 0 and bar_range < prev_range * min_outside_ratio:
        return empty

    if direction == "BUY":
        if low >= prev_low or close <= prev_high or close <= open_ or high <= prev_high:
            return empty
        if stagnation is not None:
            swept_zone_low = low < stagnation.zone_low
            if not swept_zone_low and close <= stagnation.zone_high:
                return empty
    else:
        if high <= prev_high or close >= prev_low or close >= open_ or low >= prev_low:
            return empty
        if stagnation is not None:
            swept_zone_high = high > stagnation.zone_high
            if not swept_zone_high and close >= stagnation.zone_low:
                return empty

    return MomentumSignal(
        detected=True,
        trigger_type="SWEEP_ENGULFING",
        bar_index=bar_index,
        timestamp=_ts_at(trigger, bar_index),
        entry_price=close,
        trigger_high=high,
        trigger_low=low,
        body_atr=round(body / atr, 4),
        atr=atr,
    )


def resolve_momentum_trigger_np(
    trigger: OhlcvArrays,
    trigger_atr: np.ndarray,
    bar_index: int,
    direction: TradeDirection,
    stagnation: StagnationCluster,
    *,
    min_body_atr: float,
    min_range_atr: float,
    min_outside_ratio: float,
) -> MomentumSignal:
    sweep = detect_sweep_engulfing_np(
        trigger,
        trigger_atr,
        bar_index,
        direction,
        stagnation,
        min_body_atr=min_body_atr,
        min_range_atr=min_range_atr,
        min_outside_ratio=min_outside_ratio,
    )
    if sweep.detected:
        return sweep
    return detect_momentum_breakout_np(
        trigger,
        trigger_atr,
        bar_index,
        direction,
        stagnation,
        min_body_atr=min_body_atr,
    )


def scan_consolidation_zones_np(
    structure: OhlcvArrays,
    structure_atr: np.ndarray,
    up_to_bar: int,
    *,
    lookback: int,
    window: int,
    width_atr: float,
) -> list[ConsolidationZone]:
    from strategies.archive.cspa import ConsolidationZone

    start_i = max(window, up_to_bar - lookback + 1)
    zones: list[ConsolidationZone] = []
    seen_mids: set[float] = set()

    for i in range(start_i, up_to_bar + 1):
        seg_start = i - window
        if seg_start < 0:
            continue
        seg_high = structure.high[seg_start:i]
        seg_low = structure.low[seg_start:i]
        if len(seg_high) < window:
            continue
        zone_high = float(np.max(seg_high))
        zone_low = float(np.min(seg_low))
        width = zone_high - zone_low
        atr_mean = float(np.mean(structure_atr[seg_start:i]))
        if atr_mean <= 0 or width >= atr_mean * width_atr:
            continue
        zone_mid = round((zone_high + zone_low) / 2.0, 6)
        if zone_mid in seen_mids:
            continue
        seen_mids.add(zone_mid)
        zones.append(
            ConsolidationZone(
                bar_start=seg_start,
                bar_end=i - 1,
                zone_high=zone_high,
                zone_low=zone_low,
                zone_mid=zone_mid,
            )
        )
    return zones


def build_trend_context_np(
    bias: OhlcvArrays,
    bias_atr: np.ndarray,
    bias_ema50: np.ndarray,
    bias_idx: int,
    phase: TrendPhase,
    impulse: ImpulseLeg,
    struct_idx: int,
    *,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    high_bar_indices: list[int],
    low_bar_indices: list[int],
    volatility_percentile: float,
) -> TrendContext:
    from strategies.archive.cspa import TrendContext, _volatility_regime_from_percentile

    trend_age = max(0, struct_idx - impulse.start_index)
    regime = _volatility_regime_from_percentile(volatility_percentile)

    slope_component = 0.5
    breakout_component = 0.5
    imbalance_component = 0.5
    if bias_idx >= 5:
        atr = atr_at_index(bias_atr, bias_idx)
        if bias_idx >= 50 and atr > 0:
            slope_raw = float(bias_ema50[bias_idx] - bias_ema50[bias_idx - 5]) / (5.0 * atr)
            slope_component = min(max(0.5 + slope_raw, 0.0), 1.0)

        start = max(0, bias_idx - 10)
        bodies = bias.close[start : bias_idx + 1] - bias.open[start : bias_idx + 1]
        bull = float(np.sum(bodies[bodies >= 0]))
        bear = float(np.sum(-bodies[bodies < 0]))
        total = bull + bear
        if total > 0:
            imbalance_component = bull / total if phase == "UPTREND" else bear / total

        highs, lows = swings_up_to(
            swing_highs,
            swing_lows,
            bias_idx,
            high_bar_indices=high_bar_indices,
            low_bar_indices=low_bar_indices,
        )
        close = float(bias.close[bias_idx])
        if phase == "UPTREND" and highs:
            breakout_component = min(max((close - highs[-1].price) / max(atr, 1e-9) + 0.5, 0.0), 1.0)
        elif phase == "DOWNTREND" and lows:
            breakout_component = min(max((lows[-1].price - close) / max(atr, 1e-9) + 0.5, 0.0), 1.0)

    momentum_score = round(
        slope_component * 0.4 + breakout_component * 0.3 + imbalance_component * 0.3,
        4,
    )
    return TrendContext(
        direction=phase,
        momentum_score=momentum_score,
        trend_age_bars=trend_age,
        volatility_regime=regime,
    )


def observe_wick_balance_np(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> float:
    ratios: list[float] = []
    for i in range(start_idx, end_idx + 1):
        o = float(open_[i])
        c = float(close[i])
        h = float(high[i])
        l = float(low[i])
        total = h - l
        if total <= 0:
            continue
        upper = h - max(o, c)
        lower = min(o, c) - l
        ratios.append((upper + lower) / total)
    if not ratios:
        return 0.0
    return round(sum(ratios) / len(ratios), 4)


def build_stagnation_quality_np(
    trigger: OhlcvArrays,
    stagnation: StagnationCluster,
    momentum: MomentumSignal,
    direction: TradeDirection,
) -> StagnationQuality:
    from strategies.archive.cspa import (
        StagnationQuality,
        composite_stagnation_quality_score,
        observe_compression_ratio,
        observe_range_decay_score,
    )

    atr = momentum.atr if momentum.atr > 0 else 1e-9
    compression = observe_compression_ratio(stagnation.zone_high, stagnation.zone_low, atr)
    ranges = trigger.high[stagnation.start_index : stagnation.end_index + 1] - trigger.low[
        stagnation.start_index : stagnation.end_index + 1
    ]
    ranges = np.maximum(ranges, 1e-9)
    wick_balance = observe_wick_balance_np(
        trigger.open,
        trigger.high,
        trigger.low,
        trigger.close,
        stagnation.start_index,
        stagnation.end_index,
    )
    range_decay = observe_range_decay_score([float(x) for x in ranges])
    composite = composite_stagnation_quality_score(compression, range_decay, wick_balance, atr)
    return StagnationQuality(
        compression_ratio=compression,
        wick_balance=wick_balance,
        range_decay_rate=range_decay,
        composite_score=composite,
    )


def build_reacceleration_np(
    trigger: OhlcvArrays,
    stagnation: StagnationCluster,
    momentum: MomentumSignal,
    direction: TradeDirection,
    *,
    imbalance_lookback: int,
) -> Reacceleration:
    from strategies.archive.cspa import Reacceleration, composite_reacceleration_score

    idx = momentum.bar_index
    atr = momentum.atr if momentum.atr > 0 else 1e-9
    open_ = float(trigger.open[idx])
    close = float(trigger.close[idx])
    velocity = observe_breakout_velocity_from_values(open_, close, direction, atr)

    if idx + 1 < trigger.length:
        next_close = float(trigger.close[idx + 1])
        if direction == "BUY":
            follow = round(next_close - close, 6)
        else:
            follow = round(close - next_close, 6)
    else:
        follow = 0.0

    imb_start = max(0, stagnation.start_index - imbalance_lookback)
    open_seg = trigger.open[imb_start : idx + 1]
    close_seg = trigger.close[imb_start : idx + 1]
    bullish = int(np.sum(close_seg > open_seg))
    ratio = bullish / len(open_seg) if len(open_seg) else 0.5
    if direction == "SELL":
        ratio = 1.0 - ratio
    imbalance = round(ratio, 4)

    composite = composite_reacceleration_score(velocity, follow, imbalance, atr)
    return Reacceleration(
        breakout_velocity=velocity,
        follow_through=follow,
        candle_imbalance=imbalance,
        composite_score=composite,
    )


def observe_breakout_velocity_from_values(
    open_: float, close: float, direction: TradeDirection, atr: float
) -> float:
    if atr <= 0:
        return 0.0
    body = close - open_ if direction == "BUY" else open_ - close
    return round(max(body, 0.0) / atr, 4)


def find_swings_np(
    ohlcv: OhlcvArrays,
    *,
    lookback: int,
    up_to_bar_index: int | None = None,
) -> tuple[list[SwingPoint], list[SwingPoint]]:
    from strategies.archive.cspa import SwingPoint

    if lookback < 1 or ohlcv.length < lookback * 2 + 1:
        return [], []

    last_bar = ohlcv.length - 1 if up_to_bar_index is None else min(up_to_bar_index, ohlcv.length - 1)
    last_pivot = last_bar - lookback
    if last_pivot < lookback:
        return [], []

    highs = ohlcv.high
    lows = ohlcv.low
    dt_ns = ohlcv.datetime_ns
    swing_highs: list[SwingPoint] = []
    swing_lows: list[SwingPoint] = []

    for pivot in range(lookback, last_pivot + 1):
        left_h = highs[pivot - lookback : pivot]
        right_h = highs[pivot + 1 : pivot + lookback + 1]
        if highs[pivot] >= left_h.max() and highs[pivot] >= right_h.max():
            swing_highs.append(
                SwingPoint(
                    bar_index=pivot,
                    timestamp=timestamp_from_ns(int(dt_ns[pivot])),
                    price=float(highs[pivot]),
                    kind="HIGH",
                )
            )
        left_l = lows[pivot - lookback : pivot]
        right_l = lows[pivot + 1 : pivot + lookback + 1]
        if lows[pivot] <= left_l.min() and lows[pivot] <= right_l.min():
            swing_lows.append(
                SwingPoint(
                    bar_index=pivot,
                    timestamp=timestamp_from_ns(int(dt_ns[pivot])),
                    price=float(lows[pivot]),
                    kind="LOW",
                )
            )
    return swing_highs, swing_lows


def classify_bias_dow_phase_np(
    bar_index: int,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    *,
    high_bar_indices: list[int],
    low_bar_indices: list[int],
) -> TrendPhase:
    if bar_index < 0:
        return "NONE"
    highs, lows = swings_up_to(
        swing_highs,
        swing_lows,
        bar_index,
        high_bar_indices=high_bar_indices,
        low_bar_indices=low_bar_indices,
    )
    if len(highs) < 2 or len(lows) < 2:
        return "NONE"
    h1, h2 = highs[-2], highs[-1]
    l1, l2 = lows[-2], lows[-1]
    if h2.price > h1.price and l2.price > l1.price:
        return "UPTREND"
    if h2.price < h1.price and l2.price < l1.price:
        return "DOWNTREND"
    return "RANGE"


def find_latest_impulse_np(
    bar_index: int,
    phase: TrendPhase,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    *,
    high_bar_indices: list[int],
    low_bar_indices: list[int],
    min_warmup: int,
) -> ImpulseLeg | None:
    from strategies.archive.cspa import ImpulseLeg

    if phase not in ("UPTREND", "DOWNTREND") or bar_index < min_warmup:
        return None
    up_to = bar_index - 1
    highs, lows = swings_up_to(
        swing_highs,
        swing_lows,
        up_to,
        high_bar_indices=high_bar_indices,
        low_bar_indices=low_bar_indices,
    )
    if phase == "UPTREND":
        if len(highs) < 1 or len(lows) < 1:
            return None
        peak = highs[-1]
        prior_lows = [s for s in lows if s.bar_index < peak.bar_index]
        if not prior_lows:
            return None
        trough = prior_lows[-1]
        size = peak.price - trough.price
        if size <= 0:
            return None
        return ImpulseLeg(
            direction="UP",
            start_index=trough.bar_index,
            end_index=peak.bar_index,
            start_price=trough.price,
            end_price=peak.price,
            impulse_size=size,
        )
    if len(lows) < 1 or len(highs) < 1:
        return None
    trough = lows[-1]
    prior_highs = [s for s in highs if s.bar_index < trough.bar_index]
    if not prior_highs:
        return None
    peak = prior_highs[-1]
    size = peak.price - trough.price
    if size <= 0:
        return None
    return ImpulseLeg(
        direction="DOWN",
        start_index=peak.bar_index,
        end_index=trough.bar_index,
        start_price=peak.price,
        end_price=trough.price,
        impulse_size=size,
    )


def _h4_bucket_ns(dt_ns: np.ndarray) -> np.ndarray:
    sec = dt_ns // np.int64(1_000_000_000)
    hrs = sec // np.int64(3600)
    bucket_hrs = (hrs // 4) * 4
    return bucket_hrs * np.int64(3600) * np.int64(1_000_000_000)


def _resample_h1_to_h4_np(
    dt_ns: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(dt_ns) == 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty, empty, empty
    buckets = _h4_bucket_ns(dt_ns)
    uniq = np.unique(buckets)
    n = len(uniq)
    h4_dt = uniq.copy()
    h4_open = np.empty(n, dtype=np.float64)
    h4_high = np.empty(n, dtype=np.float64)
    h4_low = np.empty(n, dtype=np.float64)
    h4_close = np.empty(n, dtype=np.float64)
    for i, bucket in enumerate(uniq):
        mask = buckets == bucket
        idx = np.where(mask)[0]
        h4_open[i] = open_[idx[0]]
        h4_high[i] = high[mask].max()
        h4_low[i] = low[mask].min()
        h4_close[i] = close[idx[-1]]
    return h4_dt, h4_open, h4_high, h4_low, h4_close


def _analyze_htf_at_ns(bias: OhlcvArrays, ts_ns: int) -> str:
    from strategies.htf_trend_analyzer import (
        HTF_BAR_HOURS,
        HTF_MA_PERIOD,
        HTF_SWING_LOOKBACK,
        classify_dow_structure,
        combine_trend_signals,
    )

    h1_end = int(np.searchsorted(bias.datetime_ns, ts_ns, side="right") - 1)
    if h1_end < 0:
        return "NEUTRAL"
    h4_dt, h4_open, h4_high, h4_low, h4_close = _resample_h1_to_h4_np(
        bias.datetime_ns[: h1_end + 1],
        bias.open[: h1_end + 1],
        bias.high[: h1_end + 1],
        bias.low[: h1_end + 1],
        bias.close[: h1_end + 1],
        bias.volume[: h1_end + 1],
    )
    cutoff_ns = np.int64(ts_ns) - np.int64(HTF_BAR_HOURS * 3600 * 1_000_000_000)
    keep = h4_dt <= cutoff_ns
    if not np.any(keep):
        return "NEUTRAL"
    h4_high = h4_high[keep]
    h4_low = h4_low[keep]
    h4_close = h4_close[keep]
    if len(h4_close) < HTF_MA_PERIOD:
        ma_bias = "NEUTRAL"
    else:
        ma200 = float(np.mean(h4_close[-HTF_MA_PERIOD:]))
        last_close = float(h4_close[-1])
        if last_close > ma200:
            ma_bias = "BULL"
        elif last_close < ma200:
            ma_bias = "BEAR"
        else:
            ma_bias = "NEUTRAL"
    lookback = HTF_SWING_LOOKBACK
    swing_highs: list[float] = []
    swing_lows: list[float] = []
    if len(h4_high) >= lookback * 2 + 1:
        last_confirmable = len(h4_high) - lookback
        for i in range(lookback, last_confirmable):
            lh = h4_high[i - lookback : i]
            rh = h4_high[i + 1 : i + lookback + 1]
            if h4_high[i] >= lh.max() and h4_high[i] >= rh.max():
                swing_highs.append(float(h4_high[i]))
            ll = h4_low[i - lookback : i]
            rl = h4_low[i + 1 : i + lookback + 1]
            if h4_low[i] <= ll.min() and h4_low[i] <= rl.min():
                swing_lows.append(float(h4_low[i]))
    dow_bias = classify_dow_structure(swing_highs, swing_lows)
    return combine_trend_signals(ma_bias, dow_bias)  # type: ignore[return-value]


def precompute_htf_directions(bias: OhlcvArrays, trigger_dt_ns: np.ndarray) -> np.ndarray:
    """Legacy eager precompute — prefer ``HtfDirectionCache``."""
    out = np.empty(len(trigger_dt_ns), dtype=object)
    cache: dict[int, str] = {}
    for i, ts_ns in enumerate(trigger_dt_ns):
        h1_end = int(np.searchsorted(bias.datetime_ns, ts_ns, side="right") - 1)
        if h1_end not in cache:
            cache[h1_end] = _analyze_htf_at_ns(bias, int(ts_ns))
        out[i] = cache[h1_end]
    return out


@dataclass(slots=True)
class HtfDirectionCache:
    """Lazy HTF direction lookup keyed by H1 end index."""

    bias: OhlcvArrays
    _by_h1_end: dict[int, str] = field(default_factory=dict, repr=False)

    def direction_at_ns(self, ts_ns: int) -> str:
        h1_end = int(np.searchsorted(self.bias.datetime_ns, ts_ns, side="right") - 1)
        if h1_end in self._by_h1_end:
            return self._by_h1_end[h1_end]
        direction = _analyze_htf_at_ns(self.bias, ts_ns)
        self._by_h1_end[h1_end] = direction
        return direction


def _utc_dt_from_ns(ts_ns: int):
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=timezone.utc)


def resolve_cspa_session_type_ns(ts_ns: int) -> str:
    from strategies.archive.cspa import DATA_DST_TYPE, _cspa_session_hour_ranges

    ts = _utc_dt_from_ns(ts_ns)
    hour = int(ts.hour)
    ranges = _cspa_session_hour_ranges(ts.date(), DATA_DST_TYPE)
    if hour in ranges["LONDON"]:
        return "LONDON"
    if hour in ranges["NY"]:
        return "NY"
    if hour in ranges["ASIA"]:
        return "ASIA"
    return "OFF_HOURS"


def session_window_start_ns(ts_ns: int, session_type: str) -> int:
    from datetime import timedelta, timezone

    from strategies.archive.cspa import CSPA_SESSION_OPEN_HOUR, DATA_DST_TYPE, shift_hour

    ts = _utc_dt_from_ns(ts_ns)
    open_hour = shift_hour(ts.date(), CSPA_SESSION_OPEN_HOUR[session_type], DATA_DST_TYPE)
    start = ts.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(hours=open_hour)
    if session_type == "NY" and ts.hour < 21:
        start -= timedelta(days=1)
    if ts < start and session_type not in ("ASIA", "OFF_HOURS"):
        start -= timedelta(days=1)
    return int(start.timestamp() * 1_000_000_000)


def minutes_from_session_open_ns(ts_ns: int, session_type: str) -> int:
    from datetime import timedelta, timezone

    from strategies.archive.cspa import CSPA_SESSION_OPEN_HOUR, DATA_DST_TYPE, shift_hour

    ts = _utc_dt_from_ns(ts_ns)
    open_hour = shift_hour(ts.date(), CSPA_SESSION_OPEN_HOUR[session_type], DATA_DST_TYPE)
    session_open = ts.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
        hours=open_hour
    )
    if ts < session_open and session_type != "ASIA":
        session_open -= timedelta(days=1)
    return max(0, int((ts - session_open).total_seconds() // 60))


def calculate_vp_profile_np(
    close: np.ndarray,
    volume: np.ndarray,
    bin_step: float,
) -> dict[str, float]:
    from volume_profile_analyzer import VALUE_AREA_FRACTION

    empty = {"vah": np.nan, "val": np.nan, "poc": np.nan}
    if len(close) == 0:
        return empty
    vol = volume if len(volume) == len(close) else np.ones(len(close), dtype=np.float64)
    vol = np.clip(vol.astype(np.float64, copy=False), 0.0, None)
    if float(vol.sum()) <= 0.0:
        vol = np.ones(len(close), dtype=np.float64)
    bins = np.round(close / bin_step) * bin_step
    uniq, inv = np.unique(bins, return_inverse=True)
    vol_sum = np.bincount(inv, weights=vol)
    if len(vol_sum) == 0 or float(vol_sum.sum()) <= 0.0:
        return empty
    poc_idx = int(np.argmax(vol_sum))
    poc = float(uniq[poc_idx])
    target = float(vol_sum.sum()) * VALUE_AREA_FRACTION
    lower_idx = upper_idx = poc_idx
    current = float(vol_sum[poc_idx])
    while current < target:
        has_lower = lower_idx > 0
        has_upper = upper_idx < len(vol_sum) - 1
        if not has_lower and not has_upper:
            break
        v_lower = float(vol_sum[lower_idx - 1]) if has_lower else -1.0
        v_upper = float(vol_sum[upper_idx + 1]) if has_upper else -1.0
        if v_lower >= v_upper:
            lower_idx -= 1
            current += v_lower
        else:
            upper_idx += 1
            current += v_upper
    return {"vah": float(uniq[upper_idx]), "val": float(uniq[lower_idx]), "poc": poc}


def evaluate_cspa_vp_location_np(
    trigger: OhlcvArrays,
    momentum: MomentumSignal,
    pair: str,
    direction: TradeDirection,
    bar_index: int,
) -> tuple[bool, int, dict[str, float]]:
    from strategies.archive.cspa import (
        CSPA_VP_BUFFER_ATR_MULT,
        CSPA_VP_BUFFER_PIPS,
        CSPA_VP_SCORE_TIERS,
    )
    from strategies.market_utils import pip_size_for_pair
    from volume_profile_analyzer import SessionVolumeProfile

    session_type = resolve_cspa_session_type_ns(int(trigger.datetime_ns[bar_index]))
    if session_type == "OFF_HOURS":
        empty = {"vah": np.nan, "val": np.nan, "poc": np.nan}
        return False, 0, empty

    start_ns = session_window_start_ns(int(trigger.datetime_ns[bar_index]), session_type)
    start_i = int(np.searchsorted(trigger.datetime_ns, start_ns, side="left"))
    end_i = min(bar_index, trigger.length - 1)
    if start_i > end_i:
        empty = {"vah": np.nan, "val": np.nan, "poc": np.nan}
        return False, 0, empty

    profiler = SessionVolumeProfile.for_pair(pair)
    levels = calculate_vp_profile_np(
        trigger.close[start_i : end_i + 1],
        trigger.volume[start_i : end_i + 1],
        profiler.bin_step,
    )
    pip = pip_size_for_pair(pair)
    filter_price = momentum.trigger_low if direction == "BUY" else momentum.trigger_high
    atr_buffer = momentum.atr * CSPA_VP_BUFFER_ATR_MULT if CSPA_VP_BUFFER_ATR_MULT > 0 and momentum.atr > 0 else None
    is_allowed, location_score = profiler.evaluate_vp_location(
        direction,
        levels,  # type: ignore[arg-type]
        pip_size=pip,
        filter_price=filter_price,
        score_price=momentum.entry_price,
        buffer_pips=CSPA_VP_BUFFER_PIPS,
        buffer_atr=atr_buffer,
        score_tiers=CSPA_VP_SCORE_TIERS,
    )
    return is_allowed, int(location_score), levels


def distance_daily_extremes_np(
    adr_cache,
    struct_idx: int,
    entry_price: float,
    atr_h1: float,
) -> tuple[float, float]:
    if struct_idx < 0 or struct_idx >= len(adr_cache.highs) or atr_h1 <= 0:
        return 0.0, 0.0
    start = int(adr_cache.day_start[struct_idx])
    day_high = float(adr_cache.highs[start : struct_idx + 1].max())
    day_low = float(adr_cache.lows[start : struct_idx + 1].min())
    return (
        round((day_high - entry_price) / atr_h1, 4),
        round((entry_price - day_low) / atr_h1, 4),
    )


def distance_session_extremes_np(
    trigger: OhlcvArrays,
    ts_ns: int,
    session_type: str,
    entry_price: float,
    pair: str,
    bar_index: int,
) -> tuple[float, float]:
    from strategies.market_utils import pip_size_for_pair

    end_i = min(bar_index, trigger.length - 1)
    if end_i < 0:
        return 0.0, 0.0
    start_ns = session_window_start_ns(ts_ns, session_type)
    start_i = int(np.searchsorted(trigger.datetime_ns, start_ns, side="left"))
    if start_i > end_i:
        return 0.0, 0.0
    sess_high = float(trigger.high[start_i : end_i + 1].max())
    sess_low = float(trigger.low[start_i : end_i + 1].min())
    pip = pip_size_for_pair(pair)
    if pip <= 0:
        return 0.0, 0.0
    return round((sess_high - entry_price) / pip, 2), round((entry_price - sess_low) / pip, 2)


def tick_volume_ratio_np(
    trigger: OhlcvArrays,
    stagnation: StagnationCluster,
    momentum: MomentumSignal,
    *,
    lookback: int,
) -> float:
    idx = momentum.bar_index
    if idx < 0 or idx >= trigger.length:
        return 1.0
    breakout_vol = float(trigger.volume[idx])
    start = max(0, idx - lookback)
    baseline = trigger.volume[start:idx]
    if len(baseline) == 0:
        return 1.0
    stag_vol = trigger.volume[stagnation.start_index : stagnation.end_index + 1]
    base_mean = float(baseline.mean()) if len(baseline) else 0.0
    if base_mean <= 0:
        return 1.0
    stag_mean = float(stag_vol.mean()) if len(stag_vol) else base_mean
    denom = max(stag_mean, base_mean * 0.5, 1e-9)
    return round(breakout_vol / denom, 4)


def wick_ratio_np(trigger: OhlcvArrays, momentum: MomentumSignal) -> float:
    idx = momentum.bar_index
    if idx < 0 or idx >= trigger.length:
        return 0.0
    high = float(trigger.high[idx])
    low = float(trigger.low[idx])
    bar_range = high - low
    if bar_range <= 0:
        return 0.0
    body = abs(float(trigger.close[idx]) - float(trigger.open[idx]))
    return round(max(0.0, (bar_range - body) / bar_range), 4)


def breakout_momentum_ratio_np(trigger: OhlcvArrays, momentum: MomentumSignal) -> float:
    idx = momentum.bar_index
    if idx < 0 or idx >= trigger.length:
        return 0.0
    body = abs(float(trigger.close[idx]) - float(trigger.open[idx]))
    bar_range = float(trigger.high[idx]) - float(trigger.low[idx])
    body_share = body / bar_range if bar_range > 0 else 0.0
    return round(momentum.body_atr * body_share, 4)


def compute_adr_pair_np(adr_cache, struct_idx: int) -> tuple[float, float]:
    from strategies.archive.cspa import CSPA_ADR_LOOKBACK_DAYS

    if struct_idx < 0 or struct_idx >= len(adr_cache.highs):
        return 0.0, 1.0
    start = int(adr_cache.day_start[struct_idx])
    day_range = float(
        adr_cache.highs[start : struct_idx + 1].max() - adr_cache.lows[start : struct_idx + 1].min()
    )
    day = adr_cache.day_norm[struct_idx]
    prior_ranges: list[float] = []
    for offset in range(1, CSPA_ADR_LOOKBACK_DAYS + 1):
        prior_day = day - np.timedelta64(offset, "D")
        val = adr_cache.daily_range.get(prior_day)
        if val is not None:
            prior_ranges.append(val)
    if not prior_ranges:
        return 0.0, 1.0
    adr = sum(prior_ranges) / len(prior_ranges)
    if adr <= 0:
        return 0.0, 1.0
    adr_used = round(day_range / adr, 4)
    return adr_used, round(max(0.0, 1.0 - adr_used), 4)
