"""
strategies/cspa_scan_engine.py — CSPA scan acceleration (numpy hot path).

Phase 1: numpy arrays (``CSPA_SCAN_NUMPY=1``, default ON)
Phase 2: numba kernels (``CSPA_SCAN_NUMBA=1``)
Phase 3: joblib pair parallel — see ``cspa_scan_parallel.py``
Phase 4: year-chunk parallel — see ``cspa_scan_parallel.py``
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strategies.archive.cspa import ImpulseLeg, StagnationCluster, SwingPoint, TrendPhase
    from strategies.archive.cspa_arrays import CspaScanArrays
    from strategies.archive.cspa_scan_context import CspaScanContext


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def scan_numpy_enabled() -> bool:
    """Phase 1: use to_numpy() hot path in detect_setups."""
    return _env_flag("CSPA_SCAN_NUMPY", default=True)


def scan_numba_enabled() -> bool:
    """Phase 2: delegate hot-path kernels to ``@njit`` implementations."""
    return _env_flag("CSPA_SCAN_NUMBA", default=False)


def scan_numba_active() -> bool:
    if not scan_numba_enabled():
        return False
    from strategies.archive.cspa_scan_numba import numba_available

    return numba_available()


def scan_parallel_pairs_enabled() -> bool:
    """Phase 3: GBPUSD + EURUSD precompute in parallel (joblib)."""
    from strategies.bt_scan_parallel import scan_parallel_pairs_enabled as bt_scan_parallel_pairs_enabled

    return bt_scan_parallel_pairs_enabled(include_cspa_alias=True)


def scan_parallel_years_enabled() -> bool:
    """Phase 4: split M1 scan by calendar year (multiprocessing)."""
    return _env_flag("CSPA_SCAN_PARALLEL_YEARS", default=False)


def scan_parallel_jobs() -> int:
    """Worker count for joblib / multiprocessing (-1 = all cores)."""
    from strategies.bt_scan_parallel import scan_parallel_jobs as bt_scan_parallel_jobs

    return bt_scan_parallel_jobs()


def measure_retrace_ratio_fast(arrays: CspaScanArrays, impulse: ImpulseLeg, bar_index: int) -> float:
    from strategies.archive.cspa_arrays import measure_retrace_ratio_np

    if scan_numba_active():
        from strategies.archive.cspa_scan_numba import measure_retrace_ratio_numba

        return measure_retrace_ratio_numba(
            arrays.structure.high,
            arrays.structure.low,
            impulse.direction == "UP",
            impulse.end_index,
            impulse.end_price,
            impulse.impulse_size,
            bar_index,
        )
    return measure_retrace_ratio_np(arrays.structure, impulse, bar_index)


def m1_over_retraces_structure_fast(arrays: CspaScanArrays, trigger_index: int, impulse: ImpulseLeg) -> bool:
    from strategies.archive.cspa_arrays import m1_over_retraces_structure_np

    if scan_numba_active():
        from strategies.archive.cspa_scan_numba import m1_over_retraces_structure_numba

        fib_max = float(os.getenv("CSPA_FIB_RETRACE_MAX", "0.618"))
        return m1_over_retraces_structure_numba(
            arrays.trigger.datetime_ns,
            arrays.trigger.high,
            arrays.trigger.low,
            arrays.structure.datetime_ns,
            impulse.end_index,
            trigger_index,
            impulse.direction == "UP",
            impulse.end_price,
            impulse.impulse_size,
            fib_max,
        )
    return m1_over_retraces_structure_np(arrays.trigger, arrays.structure, trigger_index, impulse)


def detect_stagnation_cluster_fast(
    arrays: CspaScanArrays,
    end_index: int,
    direction: str,
    *,
    max_bars: int,
) -> StagnationCluster | None:
    from strategies.archive.cspa import STAGNATION_MAX_BARS
    from strategies.archive.cspa_arrays import detect_stagnation_cluster_np

    if scan_numba_active():
        from strategies.archive.cspa_scan_numba import (
            detect_stagnation_cluster_numba,
            stagnation_from_numba_tuple,
        )

        raw = detect_stagnation_cluster_numba(
            arrays.trigger.open,
            arrays.trigger.high,
            arrays.trigger.low,
            arrays.trigger.close,
            arrays.trigger_atr,
            end_index,
            direction == "BUY",
            max_bars,
        )
        return stagnation_from_numba_tuple(raw)

    return detect_stagnation_cluster_np(
        arrays.trigger,
        arrays.trigger_atr,
        end_index,
        direction,  # type: ignore[arg-type]
        max_bars=max_bars or STAGNATION_MAX_BARS,
    )


def correction_rhythm_ok_fast(
    arrays: CspaScanArrays,
    impulse: ImpulseLeg,
    phase: TrendPhase,
    current_ratio: float,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    *,
    high_bar_indices: list[int],
    low_bar_indices: list[int],
    scan_ctx: CspaScanContext | None = None,
) -> bool:
    from strategies.archive.cspa import CORRECTION_RHYTHM_MAX_RATIO
    from strategies.archive.cspa_scan_hot import correction_rhythm_ok_np

    if scan_numba_active() and scan_ctx is not None:
        from strategies.archive.cspa_scan_numba import correction_rhythm_ok_numba

        return correction_rhythm_ok_numba(
            arrays.structure.high,
            arrays.structure.low,
            phase == "UPTREND",
            impulse.start_index,
            scan_ctx.struct_high_idx,
            scan_ctx.struct_high_prices,
            scan_ctx.struct_low_idx,
            scan_ctx.struct_low_prices,
            current_ratio,
            CORRECTION_RHYTHM_MAX_RATIO,
        )

    return correction_rhythm_ok_np(
        arrays.structure,
        impulse,
        phase,
        current_ratio,
        swing_highs,
        swing_lows,
        high_bar_indices=high_bar_indices,
        low_bar_indices=low_bar_indices,
        max_ratio=CORRECTION_RHYTHM_MAX_RATIO,
    )


def prior_correction_ratio_fast(
    arrays: CspaScanArrays,
    impulse: ImpulseLeg,
    phase: TrendPhase,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    *,
    high_bar_indices: list[int],
    low_bar_indices: list[int],
    scan_ctx: CspaScanContext | None = None,
) -> float | None:
    from strategies.archive.cspa_scan_hot import prior_correction_ratio_np

    if scan_numba_active() and scan_ctx is not None:
        from strategies.archive.cspa_scan_numba import (
            prior_correction_ratio_numba,
            prior_ratio_from_numba,
        )

        raw = prior_correction_ratio_numba(
            arrays.structure.high,
            arrays.structure.low,
            phase == "UPTREND",
            impulse.start_index,
            scan_ctx.struct_high_idx,
            scan_ctx.struct_high_prices,
            scan_ctx.struct_low_idx,
            scan_ctx.struct_low_prices,
        )
        return prior_ratio_from_numba(raw)

    return prior_correction_ratio_np(
        arrays.structure,
        impulse,
        phase,
        swing_highs,
        swing_lows,
        high_bar_indices=high_bar_indices,
        low_bar_indices=low_bar_indices,
    )


def volatility_percentile_fast(arrays: CspaScanArrays, bias_idx: int) -> float:
    from strategies.archive.cspa import ATR_PERIOD, CSPA_VOLATILITY_LOOKBACK_BARS
    from strategies.archive.cspa_scan_hot import volatility_percentile_np

    if scan_numba_active():
        from strategies.archive.cspa_scan_numba import volatility_percentile_numba

        return volatility_percentile_numba(
            arrays.bias_atr,
            bias_idx,
            CSPA_VOLATILITY_LOOKBACK_BARS,
            ATR_PERIOD,
        )

    return volatility_percentile_np(
        arrays.bias_atr,
        bias_idx,
        lookback=CSPA_VOLATILITY_LOOKBACK_BARS,
        atr_period=ATR_PERIOD,
    )


def build_pullback_rhythm_fast(
    arrays: CspaScanArrays,
    impulse: ImpulseLeg,
    struct_idx: int,
    retrace_ratio: float,
):
    from strategies.archive.cspa import (
        PullbackRhythm,
        composite_rhythm_score,
        observe_pullback_efficiency,
    )
    from strategies.archive.cspa_arrays import atr_at_index
    from strategies.archive.cspa_scan_hot import build_pullback_rhythm_np

    if scan_numba_active():
        from strategies.archive.cspa_scan_numba import (
            observe_correction_smoothness_numba,
            observe_overlap_ratio_numba,
        )

        duration = max(1, struct_idx - impulse.end_index)
        atr = atr_at_index(arrays.structure_atr, struct_idx)
        overlap = observe_overlap_ratio_numba(
            arrays.structure.high,
            arrays.structure.low,
            impulse.end_index,
            struct_idx,
        )
        smoothness = observe_correction_smoothness_numba(
            arrays.structure.high,
            arrays.structure.low,
            impulse.end_index,
            struct_idx,
        )
        retrace_distance = retrace_ratio * impulse.impulse_size
        efficiency = observe_pullback_efficiency(retrace_distance, duration)
        rhythm = composite_rhythm_score(overlap, smoothness, efficiency, atr)
        return PullbackRhythm(
            duration_bars=duration,
            retracement_depth=round(retrace_ratio, 4),
            overlap_ratio=overlap,
            pullback_efficiency=efficiency,
            correction_smoothness=smoothness,
            rhythm_score=rhythm,
        )

    return build_pullback_rhythm_np(
        arrays.structure,
        arrays.structure_atr,
        impulse,
        struct_idx,
        retrace_ratio,
    )


def resolve_momentum_trigger_fast(
    arrays: CspaScanArrays,
    bar_index: int,
    direction: str,
    stagnation: StagnationCluster,
):
    from strategies.archive.cspa import (
        CSPA_FX_SWEEP_MIN_OUTSIDE_RATIO,
        CSPA_FX_SWEEP_MIN_RANGE_ATR,
        MOMENTUM_MIN_BODY_ATR,
    )
    from strategies.archive.cspa_scan_hot import resolve_momentum_trigger_np

    if scan_numba_active():
        from strategies.archive.cspa_scan_numba import (
            detect_momentum_breakout_numba,
            detect_sweep_engulfing_numba,
            momentum_from_numba_tuple,
        )

        is_buy = direction == "BUY"
        sweep = detect_sweep_engulfing_numba(
            arrays.trigger.open,
            arrays.trigger.high,
            arrays.trigger.low,
            arrays.trigger.close,
            arrays.trigger_atr,
            bar_index,
            is_buy,
            stagnation.zone_high,
            stagnation.zone_low,
            MOMENTUM_MIN_BODY_ATR,
            CSPA_FX_SWEEP_MIN_RANGE_ATR,
            CSPA_FX_SWEEP_MIN_OUTSIDE_RATIO,
        )
        if sweep[0]:
            return momentum_from_numba_tuple(arrays.trigger, bar_index, sweep)
        breakout = detect_momentum_breakout_numba(
            arrays.trigger.open,
            arrays.trigger.high,
            arrays.trigger.low,
            arrays.trigger.close,
            arrays.trigger_atr,
            bar_index,
            is_buy,
            stagnation.zone_high,
            stagnation.zone_low,
            MOMENTUM_MIN_BODY_ATR,
        )
        return momentum_from_numba_tuple(arrays.trigger, bar_index, breakout)

    return resolve_momentum_trigger_np(
        arrays.trigger,
        arrays.trigger_atr,
        bar_index,
        direction,  # type: ignore[arg-type]
        stagnation,
        min_body_atr=MOMENTUM_MIN_BODY_ATR,
        min_range_atr=CSPA_FX_SWEEP_MIN_RANGE_ATR,
        min_outside_ratio=CSPA_FX_SWEEP_MIN_OUTSIDE_RATIO,
    )


def scan_consolidation_zones_fast(arrays: CspaScanArrays, struct_idx: int):
    from strategies.archive.cspa import (
        CSPA_CONSOLIDATION_LOOKBACK,
        CSPA_CONSOLIDATION_WINDOW,
        CSPA_CONSOLIDATION_WIDTH_ATR,
    )
    from strategies.archive.cspa_scan_hot import scan_consolidation_zones_np

    return scan_consolidation_zones_np(
        arrays.structure,
        arrays.structure_atr,
        struct_idx,
        lookback=CSPA_CONSOLIDATION_LOOKBACK,
        window=CSPA_CONSOLIDATION_WINDOW,
        width_atr=CSPA_CONSOLIDATION_WIDTH_ATR,
    )


def build_trend_context_fast(
    arrays: CspaScanArrays,
    bias_idx: int,
    phase: TrendPhase,
    impulse: ImpulseLeg,
    struct_idx: int,
    volatility_percentile: float,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    *,
    high_bar_indices: list[int],
    low_bar_indices: list[int],
):
    from strategies.archive.cspa_scan_hot import build_trend_context_np

    return build_trend_context_np(
        arrays.bias,
        arrays.bias_atr,
        arrays.bias_ema50,
        bias_idx,
        phase,
        impulse,
        struct_idx,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        high_bar_indices=high_bar_indices,
        low_bar_indices=low_bar_indices,
        volatility_percentile=volatility_percentile,
    )


def build_stagnation_quality_fast(
    arrays: CspaScanArrays,
    stagnation: StagnationCluster,
    momentum,
    direction: str,
):
    from strategies.archive.cspa import (
        StagnationQuality,
        composite_stagnation_quality_score,
        observe_compression_ratio,
        observe_range_decay_score,
    )
    from strategies.archive.cspa_scan_hot import build_stagnation_quality_np

    if scan_numba_active():
        from strategies.archive.cspa_scan_numba import observe_wick_balance_numba

        wick_balance = observe_wick_balance_numba(
            arrays.trigger.open,
            arrays.trigger.high,
            arrays.trigger.low,
            arrays.trigger.close,
            stagnation.start_index,
            stagnation.end_index,
        )
        atr = momentum.atr if momentum.atr > 0 else 1e-9
        compression = observe_compression_ratio(stagnation.zone_high, stagnation.zone_low, atr)
        ranges = arrays.trigger.high[stagnation.start_index : stagnation.end_index + 1] - arrays.trigger.low[
            stagnation.start_index : stagnation.end_index + 1
        ]
        range_decay = observe_range_decay_score([float(x) for x in ranges])
        composite = composite_stagnation_quality_score(compression, range_decay, wick_balance, atr)
        return StagnationQuality(
            compression_ratio=compression,
            wick_balance=wick_balance,
            range_decay_rate=range_decay,
            composite_score=composite,
        )

    return build_stagnation_quality_np(
        arrays.trigger,
        stagnation,
        momentum,
        direction,  # type: ignore[arg-type]
    )


def build_reacceleration_fast(
    arrays: CspaScanArrays,
    stagnation: StagnationCluster,
    momentum,
    direction: str,
):
    from strategies.archive.cspa import CSPA_REACCEL_IMBALANCE_LOOKBACK
    from strategies.archive.cspa_scan_hot import build_reacceleration_np

    return build_reacceleration_np(
        arrays.trigger,
        stagnation,
        momentum,
        direction,  # type: ignore[arg-type]
        imbalance_lookback=CSPA_REACCEL_IMBALANCE_LOOKBACK,
    )


def atr_at_bar_fast(arrays: CspaScanArrays, frame: str, bar_index: int) -> float:
    from strategies.archive.cspa_arrays import atr_at_index

    if frame == "structure":
        return atr_at_index(arrays.structure_atr, bar_index)
    if frame == "bias":
        return atr_at_index(arrays.bias_atr, bar_index)
    return atr_at_index(arrays.trigger_atr, bar_index)


def evaluate_cspa_vp_location_fast(
    arrays: CspaScanArrays,
    momentum,
    pair: str,
    direction: str,
    bar_index: int,
) -> tuple[bool, int, dict]:
    from strategies.archive.cspa_scan_hot import evaluate_cspa_vp_location_np

    return evaluate_cspa_vp_location_np(
        arrays.trigger,
        momentum,
        pair,
        direction,  # type: ignore[arg-type]
        bar_index,
    )


def classify_bias_dow_phase_fast(ctx, bias_idx: int) -> str:
    from strategies.archive.cspa_scan_hot import classify_bias_dow_phase_np

    if scan_numba_active():
        from strategies.archive.cspa_scan_numba import (
            classify_bias_dow_phase_numba,
            phase_from_numba,
        )

        code = classify_bias_dow_phase_numba(
            bias_idx,
            ctx.bias_high_idx,
            ctx.bias_high_prices,
            ctx.bias_low_idx,
            ctx.bias_low_prices,
        )
        return phase_from_numba(code)

    return classify_bias_dow_phase_np(
        bias_idx,
        ctx.bias_swing_highs,
        ctx.bias_swing_lows,
        high_bar_indices=ctx.bias_high_indices,
        low_bar_indices=ctx.bias_low_indices,
    )


def find_latest_impulse_fast(ctx, struct_idx: int, phase: str):
    from strategies.archive.cspa import SWING_LOOKBACK_STRUCTURE
    from strategies.archive.cspa_scan_hot import find_latest_impulse_np

    if scan_numba_active():
        from strategies.archive.cspa_scan_numba import (
            PHASE_DOWNTREND,
            PHASE_NONE,
            PHASE_RANGE,
            PHASE_UPTREND,
            find_latest_impulse_numba,
            impulse_from_numba_tuple,
        )

        phase_codes = {
            "NONE": PHASE_NONE,
            "UPTREND": PHASE_UPTREND,
            "DOWNTREND": PHASE_DOWNTREND,
            "RANGE": PHASE_RANGE,
        }
        raw = find_latest_impulse_numba(
            struct_idx,
            phase_codes.get(phase, PHASE_NONE),
            ctx.struct_high_idx,
            ctx.struct_high_prices,
            ctx.struct_low_idx,
            ctx.struct_low_prices,
            SWING_LOOKBACK_STRUCTURE * 4,
        )
        return impulse_from_numba_tuple(raw)

    return find_latest_impulse_np(
        struct_idx,
        phase,  # type: ignore[arg-type]
        ctx.struct_swing_highs,
        ctx.struct_swing_lows,
        high_bar_indices=ctx.struct_high_indices,
        low_bar_indices=ctx.struct_low_indices,
        min_warmup=SWING_LOOKBACK_STRUCTURE * 4,
    )
