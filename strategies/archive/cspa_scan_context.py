"""
strategies/cspa_scan_context.py — Full numpy scan context (zero pandas in detect loop).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from strategies.archive.cspa import (
    ATR_PERIOD,
    CSPA_BT_SPREAD_PIPS,
    CSPA_VOLUME_LOOKBACK_BARS,
    SWING_LOOKBACK_BIAS,
    SWING_LOOKBACK_STRUCTURE,
    CspaBayesFeatures,
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
    TpMode,
    _StructureAdrCache,
    classify_dow_phase_maturity,
    _spread_percentile,
)
from strategies.archive.cspa_scan_hot import minutes_from_session_open_ns
from strategies.archive.cspa_arrays import (
    CspaScanArrays,
    OhlcvArrays,
    atr_at_index,
    build_adr_cache_np,
    build_mtf_index_maps,
    compute_atr_np,
    compute_ema_np,
)
from strategies.archive.cspa_scan_hot import (
    HtfDirectionCache,
    breakout_momentum_ratio_np,
    compute_adr_pair_np,
    distance_daily_extremes_np,
    distance_session_extremes_np,
    find_swings_np,
    resolve_cspa_session_type_ns,
    tick_volume_ratio_np,
    wick_ratio_np,
)


@dataclass(slots=True)
class CspaScanContext:
    arrays: CspaScanArrays
    adr_cache: _StructureAdrCache
    bias_swing_highs: list[SwingPoint]
    bias_swing_lows: list[SwingPoint]
    struct_swing_highs: list[SwingPoint]
    struct_swing_lows: list[SwingPoint]
    bias_high_indices: list[int]
    bias_low_indices: list[int]
    struct_high_indices: list[int]
    struct_low_indices: list[int]
    struct_high_idx: np.ndarray
    struct_low_idx: np.ndarray
    struct_high_prices: np.ndarray
    struct_low_prices: np.ndarray
    bias_high_idx: np.ndarray
    bias_low_idx: np.ndarray
    bias_high_prices: np.ndarray
    bias_low_prices: np.ndarray
    htf_cache: HtfDirectionCache
    trigger_length: int


def build_scan_context(
    trigger: Any,
    structure: Any,
    bias: Any,
) -> CspaScanContext:
    """Convert MTF frames once; detect loop uses only ``CspaScanContext``."""
    trigger_a = OhlcvArrays.from_prepared_df(trigger)
    structure_a = OhlcvArrays.from_prepared_df(structure)
    bias_a = OhlcvArrays.from_prepared_df(bias)

    trigger_atr = compute_atr_np(trigger_a.high, trigger_a.low, trigger_a.close, ATR_PERIOD)
    structure_atr = compute_atr_np(structure_a.high, structure_a.low, structure_a.close, ATR_PERIOD)
    bias_atr = compute_atr_np(bias_a.high, bias_a.low, bias_a.close, ATR_PERIOD)
    bias_ema50 = compute_ema_np(bias_a.close, 50)
    struct_idx_by_bar, bias_idx_by_bar = build_mtf_index_maps(trigger_a, structure_a, bias_a)

    arrays = CspaScanArrays(
        trigger=trigger_a,
        structure=structure_a,
        bias=bias_a,
        trigger_atr=trigger_atr,
        structure_atr=structure_atr,
        bias_atr=bias_atr,
        bias_ema50=bias_ema50,
        struct_idx_by_bar=struct_idx_by_bar,
        bias_idx_by_bar=bias_idx_by_bar,
    )

    adr_cache = build_adr_cache_np(structure_a)
    bias_swing_highs, bias_swing_lows = find_swings_np(
        bias_a, lookback=SWING_LOOKBACK_BIAS, up_to_bar_index=None
    )
    struct_swing_highs, struct_swing_lows = find_swings_np(
        structure_a, lookback=SWING_LOOKBACK_STRUCTURE, up_to_bar_index=None
    )
    htf_cache = HtfDirectionCache(bias_a)

    struct_high_idx = np.asarray([s.bar_index for s in struct_swing_highs], dtype=np.int64)
    struct_low_idx = np.asarray([s.bar_index for s in struct_swing_lows], dtype=np.int64)
    struct_high_prices = np.asarray([s.price for s in struct_swing_highs], dtype=np.float64)
    struct_low_prices = np.asarray([s.price for s in struct_swing_lows], dtype=np.float64)
    bias_high_idx = np.asarray([s.bar_index for s in bias_swing_highs], dtype=np.int64)
    bias_low_idx = np.asarray([s.bar_index for s in bias_swing_lows], dtype=np.int64)
    bias_high_prices = np.asarray([s.price for s in bias_swing_highs], dtype=np.float64)
    bias_low_prices = np.asarray([s.price for s in bias_swing_lows], dtype=np.float64)

    return CspaScanContext(
        arrays=arrays,
        adr_cache=adr_cache,
        bias_swing_highs=bias_swing_highs,
        bias_swing_lows=bias_swing_lows,
        struct_swing_highs=struct_swing_highs,
        struct_swing_lows=struct_swing_lows,
        bias_high_indices=[s.bar_index for s in bias_swing_highs],
        bias_low_indices=[s.bar_index for s in bias_swing_lows],
        struct_high_indices=[s.bar_index for s in struct_swing_highs],
        struct_low_indices=[s.bar_index for s in struct_swing_lows],
        struct_high_idx=struct_high_idx,
        struct_low_idx=struct_low_idx,
        struct_high_prices=struct_high_prices,
        struct_low_prices=struct_low_prices,
        bias_high_idx=bias_high_idx,
        bias_low_idx=bias_low_idx,
        bias_high_prices=bias_high_prices,
        bias_low_prices=bias_low_prices,
        htf_cache=htf_cache,
        trigger_length=trigger_a.length,
    )


def build_cspa_bayes_features_np(
    ctx: CspaScanContext,
    *,
    pair: str,
    bias_phase: TrendPhase,
    impulse: ImpulseLeg,
    struct_idx: int,
    bias_idx: int,
    retrace_ratio: float,
    stagnation: StagnationCluster,
    momentum: MomentumSignal,
    impulse_size_atr: float,
    trend_context: TrendContext,
    pullback_rhythm: PullbackRhythm,
    stagnation_quality: StagnationQuality,
    reacceleration: Reacceleration,
    tp_mode: TpMode,
    tp_rr_actual: float,
    structure_score: float,
    market_breath_score: float,
    volatility_percentile: float,
    vp_location_score: int,
    trigger_bar_index: int,
) -> CspaBayesFeatures:
    from strategies.archive.cspa import DEFAULT_RISK_REWARD
    from strategies.market_utils import pip_size_for_pair

    trade_dir: TradeDirection = "BUY" if bias_phase == "UPTREND" else "SELL"
    pullback_duration = max(0, struct_idx - impulse.end_index)
    trend_age_bars = max(0, struct_idx - impulse.start_index)
    pip = pip_size_for_pair(pair)
    stagnation_pips = (stagnation.zone_high - stagnation.zone_low) / pip if pip > 0 else 0.0
    h1_atr = atr_at_index(ctx.arrays.bias_atr, bias_idx)
    impulse_quality = round(min(max(impulse_size_atr / 1.5, 0.0), 1.0), 4)
    ts_ns = int(ctx.arrays.trigger.datetime_ns[trigger_bar_index])
    session_type = resolve_cspa_session_type_ns(ts_ns)
    adr_used, adr_remaining = compute_adr_pair_np(ctx.adr_cache, struct_idx)
    dist_daily_high, dist_daily_low = distance_daily_extremes_np(
        ctx.adr_cache, struct_idx, momentum.entry_price, h1_atr
    )
    dist_sess_high, dist_sess_low = distance_session_extremes_np(
        ctx.arrays.trigger,
        ts_ns,
        session_type,
        momentum.entry_price,
        pair,
        trigger_bar_index,
    )

    return CspaBayesFeatures(
        dow_phase=classify_dow_phase_maturity(bias_phase, pullback_duration),
        trend_age_bars=trend_age_bars,
        pullback_depth=round(retrace_ratio, 4),
        pullback_duration=pullback_duration,
        impulse_quality=impulse_quality,
        impulse_atr_ratio=round(impulse_size_atr, 4),
        stagnation_duration=stagnation.bar_count,
        stagnation_pips_width=round(stagnation_pips, 2),
        stagnation_compression_ratio=stagnation_quality.compression_ratio,
        stagnation_wick_balance=stagnation_quality.wick_balance,
        stagnation_range_decay_rate=stagnation_quality.range_decay_rate,
        stagnation_quality_score=stagnation_quality.composite_score,
        pullback_overlap_ratio=pullback_rhythm.overlap_ratio,
        pullback_efficiency=pullback_rhythm.pullback_efficiency,
        correction_smoothness=pullback_rhythm.correction_smoothness,
        rhythm_score=pullback_rhythm.rhythm_score,
        correction_velocity=pullback_rhythm.pullback_efficiency,
        breakout_momentum_ratio=breakout_momentum_ratio_np(ctx.arrays.trigger, momentum),
        breakout_velocity=reacceleration.breakout_velocity,
        wick_ratio=wick_ratio_np(ctx.arrays.trigger, momentum),
        reaccel_follow_through=reacceleration.follow_through,
        reaccel_candle_imbalance=reacceleration.candle_imbalance,
        reacceleration_score=reacceleration.composite_score,
        h1_momentum_score=trend_context.momentum_score,
        tp_mode=tp_mode,
        tp_rr_actual=round(tp_rr_actual, 4),
        structure_score=round(structure_score, 2),
        market_breath_score=round(market_breath_score, 2),
        current_atr_h1=round(h1_atr, 6),
        volatility_percentile=volatility_percentile,
        session_type=session_type,
        minutes_from_session_open=minutes_from_session_open_ns(ts_ns, session_type),
        adr_used=adr_used,
        adr_remaining=adr_remaining,
        distance_daily_high=dist_daily_high,
        distance_daily_low=dist_daily_low,
        distance_session_high=dist_sess_high,
        distance_session_low=dist_sess_low,
        spread=CSPA_BT_SPREAD_PIPS,
        spread_percentile=_spread_percentile(CSPA_BT_SPREAD_PIPS),
        tick_volume_ratio=tick_volume_ratio_np(
            ctx.arrays.trigger,
            stagnation,
            momentum,
            lookback=CSPA_VOLUME_LOOKBACK_BARS,
        ),
        vp_location_score=vp_location_score,
    )
