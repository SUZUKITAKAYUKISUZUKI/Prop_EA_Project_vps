"""
strategies/lgr_scan_hot.py — NumPy-only LGR precompute scan (BT / WFT hot path).

Phase 1: ``LGR_SCAN_NUMPY=1`` (default ON) — ``detect_lgr_setups_for_pair`` delegates here.
Phase 2: ``LGR_SCAN_NUMBA=1`` — session/daily pool lookups via ``lgr_scan_numba``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

from strategies.bt_ohlcv import OhlcvArrays, as_ohlcv, asof_end_index, day_norm_from_datetime_ns, slice_arrays, ts_ns_to_pd
from strategies.archive.cspa import TrendPhase, classify_dow_phase_maturity
from strategies.archive.cspa_arrays import atr_at_index, compute_atr_np
from strategies.archive.cspa_scan_hot import HtfDirectionCache, resolve_cspa_session_type_ns
from strategies.archive.liquidity_grab_detector import (
    MIN_RECOVERY_RATIO,
    MIN_SWEEP_ATR,
    STRONG_CLOSE_BODY_RATIO,
    GrabDetection,
    LiquidityPoolType,
)
from strategies.market_utils import pip_size_for_pair
from strategies.archive.reversal_feature_helpers_np import (
    adr_used_fraction_np,
    atr_at_np,
    build_bar_hours,
    compute_adr_remaining_np,
    compute_recovery_close_ratio_np,
    liquidity_distances_np,
    minutes_from_session_open_ns,
    spread_percentile_np,
    volatility_percentile_for_bar,
)

__all__ = [
    "clear_lgr_funnel_detector_level",
    "configure_lgr_detector_for_bt",
    "configure_lgr_production_detector",
    "detect_grab_at_index_np",
    "detect_lgr_setups_for_pair_np",
    "lgr_detector_mode_label",
    "lgr_scan_numba_active",
    "lgr_scan_numpy_enabled",
    "set_lgr_funnel_detector_level",
]


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def lgr_scan_numpy_enabled() -> bool:
    return _env_flag("LGR_SCAN_NUMPY", default=True)


def lgr_scan_numba_active() -> bool:
    from strategies.archive.lgr_scan_numba import lgr_scan_numba_active as _active

    return _active()


@dataclass(frozen=True)
class LgrScanContext:
    exec_arr: OhlcvArrays
    h1_arr: OhlcvArrays
    exec_atr: np.ndarray
    h1_atr: np.ndarray
    day_norm: np.ndarray
    bar_hours: np.ndarray
    htf_cache: HtfDirectionCache
    pip: float


def _grab_strength(recovery_ratio: float, sweep_pips: float, body_ratio: float) -> float:
    sweep_score = min(sweep_pips / 3.0, 1.0)
    return 0.45 * recovery_ratio + 0.35 * sweep_score + 0.20 * body_ratio


def _prior_daily_extremes(ctx: LgrScanContext, idx: int) -> tuple[float | None, float | None]:
    if lgr_scan_numba_active():
        from strategies.archive.lgr_scan_numba import prior_daily_extremes_numba

        hi, lo, ok = prior_daily_extremes_numba(
            ctx.exec_arr.high, ctx.exec_arr.low, ctx.day_norm, idx
        )
        if ok == 0:
            return None, None
        return float(hi), float(lo)

    if idx < 1:
        return None, None
    current_day = ctx.day_norm[idx]
    mask = (np.arange(idx) < idx) & (ctx.day_norm[:idx] == current_day)
    if not np.any(mask):
        return None, None
    prior = np.where(mask)[0]
    return float(np.max(ctx.exec_arr.high[prior])), float(np.min(ctx.exec_arr.low[prior]))


def _prior_session_extremes(ctx: LgrScanContext, idx: int) -> tuple[float | None, float | None]:
    if lgr_scan_numba_active():
        from strategies.archive.lgr_scan_numba import prior_session_extremes_numba, session_type_to_code

        session = resolve_cspa_session_type_ns(int(ctx.exec_arr.datetime_ns[idx]))
        hi, lo, ok = prior_session_extremes_numba(
            ctx.exec_arr.high,
            ctx.exec_arr.low,
            ctx.day_norm,
            ctx.bar_hours,
            idx,
            session_type_to_code(session),
        )
        if ok == 0:
            return None, None
        return float(hi), float(lo)

    if idx < 1:
        return None, None
    session = resolve_cspa_session_type_ns(int(ctx.exec_arr.datetime_ns[idx]))
    current_day = ctx.day_norm[idx]
    day_indices = np.where(ctx.day_norm[:idx] == current_day)[0]
    if len(day_indices) == 0:
        return None, None
    if session == "LONDON":
        sess = day_indices[(ctx.bar_hours[day_indices] >= 8) & (ctx.bar_hours[day_indices] < 17)]
    elif session == "NY":
        sess = day_indices[(ctx.bar_hours[day_indices] >= 13) & (ctx.bar_hours[day_indices] < 22)]
    elif session == "ASIA":
        sess = day_indices[ctx.bar_hours[day_indices] < 8]
    else:
        sess = day_indices
    if len(sess) == 0:
        sess = day_indices
    return float(np.max(ctx.exec_arr.high[sess])), float(np.min(ctx.exec_arr.low[sess]))


def detect_long_grab_np(
    ctx: LgrScanContext,
    idx: int,
    *,
    min_sweep_atr: float = MIN_SWEEP_ATR,
    strong_close_body_ratio: float = STRONG_CLOSE_BODY_RATIO,
    min_recovery_ratio: float = MIN_RECOVERY_RATIO,
    sl_buffer_atr: float = 0.25,
    min_rr: float = 1.5,
    atr: float | None = None,
) -> GrabDetection | None:
    arr = ctx.exec_arr
    if idx < 1 or idx + 1 >= arr.length:
        return None
    atr_val = atr if atr is not None and atr > 0 else atr_at_np(ctx.exec_atr, idx)
    if atr_val <= 0 or ctx.pip <= 0:
        return None

    bar_high = float(arr.high[idx])
    bar_low = float(arr.low[idx])
    bar_open = float(arr.open[idx])
    bar_close = float(arr.close[idx])
    span = bar_high - bar_low
    if span <= 0:
        return None

    prior_daily_high, prior_daily_low = _prior_daily_extremes(ctx, idx)
    prior_sess_high, prior_sess_low = _prior_session_extremes(ctx, idx)
    if prior_daily_high is None or prior_daily_low is None:
        return None

    if bar_high > prior_daily_high + 1e-9:
        return None
    if prior_sess_high is not None and bar_high > prior_sess_high + 1e-9:
        return None

    min_sweep = min_sweep_atr * atr_val
    pool_type: LiquidityPoolType | None = None
    pool_level: float | None = None
    if prior_sess_low is not None and bar_low < prior_sess_low - min_sweep:
        pool_type, pool_level = "SESSION_LOW", prior_sess_low
    elif bar_low < prior_daily_low - min_sweep:
        pool_type, pool_level = "DAILY_LOW", prior_daily_low
    if pool_type is None or pool_level is None:
        return None

    sweep_distance = pool_level - bar_low
    if sweep_distance < min_sweep:
        return None
    if bar_close <= pool_level:
        return None

    body_ratio = (bar_close - bar_open) / span
    if bar_close <= bar_open or body_ratio < strong_close_body_ratio:
        return None

    recovery_ratio = (bar_close - bar_low) / max(sweep_distance, 1e-9)
    if recovery_ratio < min_recovery_ratio:
        return None

    entry_idx = idx + 1
    entry_price = float(arr.open[entry_idx])
    stop_loss = bar_low - sl_buffer_atr * atr_val
    risk = entry_price - stop_loss
    if risk <= 0:
        return None
    take_profit = entry_price + min_rr * risk
    sweep_pips = sweep_distance / ctx.pip
    grab_strength = _grab_strength(recovery_ratio, sweep_pips, body_ratio)

    return GrabDetection(
        is_grabbed=True,
        direction="BUY",
        grab_price=round(bar_low, 6),
        sweep_distance_pips=round(sweep_pips, 4),
        recovery_ratio=round(recovery_ratio, 4),
        grab_strength=round(grab_strength, 4),
        liquidity_pool_type=pool_type,
        trigger_idx=idx,
        entry_idx=entry_idx,
        entry_price=round(entry_price, 6),
        stop_loss=round(stop_loss, 6),
        take_profit=round(take_profit, 6),
    )


def detect_short_grab_np(
    ctx: LgrScanContext,
    idx: int,
    *,
    min_sweep_atr: float = MIN_SWEEP_ATR,
    strong_close_body_ratio: float = STRONG_CLOSE_BODY_RATIO,
    min_recovery_ratio: float = MIN_RECOVERY_RATIO,
    sl_buffer_atr: float = 0.25,
    min_rr: float = 1.5,
    atr: float | None = None,
) -> GrabDetection | None:
    arr = ctx.exec_arr
    if idx < 1 or idx + 1 >= arr.length:
        return None
    atr_val = atr if atr is not None and atr > 0 else atr_at_np(ctx.exec_atr, idx)
    if atr_val <= 0 or ctx.pip <= 0:
        return None

    bar_high = float(arr.high[idx])
    bar_low = float(arr.low[idx])
    bar_open = float(arr.open[idx])
    bar_close = float(arr.close[idx])
    span = bar_high - bar_low
    if span <= 0:
        return None

    prior_daily_high, prior_daily_low = _prior_daily_extremes(ctx, idx)
    prior_sess_high, prior_sess_low = _prior_session_extremes(ctx, idx)
    if prior_daily_high is None or prior_daily_low is None:
        return None

    if bar_low < prior_daily_low - 1e-9:
        return None
    if prior_sess_low is not None and bar_low < prior_sess_low - 1e-9:
        return None

    min_sweep = min_sweep_atr * atr_val
    pool_type: LiquidityPoolType | None = None
    pool_level: float | None = None
    if prior_sess_high is not None and bar_high > prior_sess_high + min_sweep:
        pool_type, pool_level = "SESSION_HIGH", prior_sess_high
    elif bar_high > prior_daily_high + min_sweep:
        pool_type, pool_level = "DAILY_HIGH", prior_daily_high
    if pool_type is None or pool_level is None:
        return None

    sweep_distance = bar_high - pool_level
    if sweep_distance < min_sweep:
        return None
    if bar_close >= pool_level:
        return None

    body_ratio = (bar_open - bar_close) / span
    if bar_close >= bar_open or body_ratio < strong_close_body_ratio:
        return None

    recovery_ratio = (bar_high - bar_close) / max(sweep_distance, 1e-9)
    if recovery_ratio < min_recovery_ratio:
        return None

    entry_idx = idx + 1
    entry_price = float(arr.open[entry_idx])
    stop_loss = bar_high + sl_buffer_atr * atr_val
    risk = stop_loss - entry_price
    if risk <= 0:
        return None
    take_profit = entry_price - min_rr * risk
    sweep_pips = sweep_distance / ctx.pip
    grab_strength = _grab_strength(recovery_ratio, sweep_pips, body_ratio)

    return GrabDetection(
        is_grabbed=True,
        direction="SELL",
        grab_price=round(bar_high, 6),
        sweep_distance_pips=round(sweep_pips, 4),
        recovery_ratio=round(recovery_ratio, 4),
        grab_strength=round(grab_strength, 4),
        liquidity_pool_type=pool_type,
        trigger_idx=idx,
        entry_idx=entry_idx,
        entry_price=round(entry_price, 6),
        stop_loss=round(stop_loss, 6),
        take_profit=round(take_profit, 6),
    )


DEFAULT_MIN_WICK_RATIO = float(os.getenv("LGR_MIN_WICK_RATIO", "0.35"))


def lgr_detector_level() -> int | None:
    """0=daily update only, 1=+wick, 2=+recovery. None=full production detector."""
    raw = os.getenv("LGR_DETECTOR_LEVEL", "").strip()
    if not raw:
        return None
    level = int(raw)
    if level not in (0, 1, 2):
        raise ValueError(f"LGR_DETECTOR_LEVEL must be 0, 1, or 2 (got {level})")
    return level


def clear_lgr_funnel_detector_level() -> None:
    """Remove funnel override so the production grab detector is used."""
    os.environ.pop("LGR_DETECTOR_LEVEL", None)


def set_lgr_funnel_detector_level(level: int) -> None:
    if level not in (0, 1, 2):
        raise ValueError(f"LGR_DETECTOR_LEVEL must be 0, 1, or 2 (got {level})")
    os.environ["LGR_DETECTOR_LEVEL"] = str(level)


def configure_lgr_production_detector() -> None:
    """Ignore shell leaks; always use the full production Liquidity Grab detector."""
    clear_lgr_funnel_detector_level()


def configure_lgr_detector_for_bt(*, funnel_level: int | None) -> None:
    """Apply BT/WFT detector mode. ``funnel_level=None`` forces production."""
    if funnel_level is None:
        configure_lgr_production_detector()
    else:
        set_lgr_funnel_detector_level(funnel_level)


def lgr_detector_mode_label() -> str:
    level = lgr_detector_level()
    if level is None:
        return "production"
    return f"funnel-L{level}"


def _funnel_metrics_long(
    ctx: LgrScanContext,
    idx: int,
    prior_daily_low: float,
) -> tuple[float, float, float] | None:
    arr = ctx.exec_arr
    bar_high = float(arr.high[idx])
    bar_low = float(arr.low[idx])
    bar_open = float(arr.open[idx])
    bar_close = float(arr.close[idx])
    span = bar_high - bar_low
    if span <= 0:
        return None
    sweep_distance = prior_daily_low - bar_low
    if sweep_distance <= 0:
        return None
    wick = min(bar_open, bar_close) - bar_low
    wick_ratio = max(0.0, min(1.0, wick / span))
    recovery_ratio = (bar_close - bar_low) / sweep_distance
    body_ratio = abs((bar_close - bar_open) / span)
    return wick_ratio, recovery_ratio, body_ratio


def _funnel_metrics_short(
    ctx: LgrScanContext,
    idx: int,
    prior_daily_high: float,
) -> tuple[float, float, float] | None:
    arr = ctx.exec_arr
    bar_high = float(arr.high[idx])
    bar_low = float(arr.low[idx])
    bar_open = float(arr.open[idx])
    bar_close = float(arr.close[idx])
    span = bar_high - bar_low
    if span <= 0:
        return None
    sweep_distance = bar_high - prior_daily_high
    if sweep_distance <= 0:
        return None
    wick = bar_high - max(bar_open, bar_close)
    wick_ratio = max(0.0, min(1.0, wick / span))
    recovery_ratio = (bar_high - bar_close) / sweep_distance
    body_ratio = abs((bar_open - bar_close) / span)
    return wick_ratio, recovery_ratio, body_ratio


def _build_funnel_grab(
    ctx: LgrScanContext,
    idx: int,
    *,
    direction: str,
    pool_type: LiquidityPoolType,
    grab_price: float,
    sweep_distance: float,
    recovery_ratio: float,
    body_ratio: float,
    sl_buffer_atr: float,
    min_rr: float,
    atr_val: float,
) -> GrabDetection | None:
    arr = ctx.exec_arr
    entry_idx = idx + 1
    if entry_idx >= arr.length:
        return None
    entry_price = float(arr.open[entry_idx])
    if direction == "BUY":
        stop_loss = grab_price - sl_buffer_atr * atr_val
        risk = entry_price - stop_loss
        if risk <= 0:
            return None
        take_profit = entry_price + min_rr * risk
        trade_dir = "BUY"
    else:
        stop_loss = grab_price + sl_buffer_atr * atr_val
        risk = stop_loss - entry_price
        if risk <= 0:
            return None
        take_profit = entry_price - min_rr * risk
        trade_dir = "SELL"
    sweep_pips = sweep_distance / ctx.pip
    grab_strength = _grab_strength(recovery_ratio, sweep_pips, body_ratio)
    return GrabDetection(
        is_grabbed=True,
        direction=trade_dir,  # type: ignore[arg-type]
        grab_price=round(grab_price, 6),
        sweep_distance_pips=round(sweep_pips, 4),
        recovery_ratio=round(recovery_ratio, 4),
        grab_strength=round(grab_strength, 4),
        liquidity_pool_type=pool_type,
        trigger_idx=idx,
        entry_idx=entry_idx,
        entry_price=round(entry_price, 6),
        stop_loss=round(stop_loss, 6),
        take_profit=round(take_profit, 6),
    )


def detect_funnel_long_grab_np(
    ctx: LgrScanContext,
    idx: int,
    level: int,
    *,
    min_wick_ratio: float = DEFAULT_MIN_WICK_RATIO,
    min_recovery_ratio: float = MIN_RECOVERY_RATIO,
    sl_buffer_atr: float = 0.25,
    min_rr: float = 1.5,
    atr: float | None = None,
) -> GrabDetection | None:
    if idx < 1 or idx + 1 >= ctx.exec_arr.length:
        return None
    atr_val = atr if atr is not None and atr > 0 else atr_at_np(ctx.exec_atr, idx)
    if atr_val <= 0:
        return None
    prior_daily_high, prior_daily_low = _prior_daily_extremes(ctx, idx)
    if prior_daily_low is None:
        return None
    bar_low = float(ctx.exec_arr.low[idx])
    if bar_low >= prior_daily_low - 1e-9:
        return None
    metrics = _funnel_metrics_long(ctx, idx, prior_daily_low)
    if metrics is None:
        return None
    wick_ratio, recovery_ratio, body_ratio = metrics
    if level >= 1 and wick_ratio < min_wick_ratio:
        return None
    if level >= 2 and recovery_ratio < min_recovery_ratio:
        return None
    sweep_distance = prior_daily_low - bar_low
    return _build_funnel_grab(
        ctx,
        idx,
        direction="BUY",
        pool_type="DAILY_LOW",
        grab_price=bar_low,
        sweep_distance=sweep_distance,
        recovery_ratio=recovery_ratio,
        body_ratio=body_ratio,
        sl_buffer_atr=sl_buffer_atr,
        min_rr=min_rr,
        atr_val=atr_val,
    )


def detect_funnel_short_grab_np(
    ctx: LgrScanContext,
    idx: int,
    level: int,
    *,
    min_wick_ratio: float = DEFAULT_MIN_WICK_RATIO,
    min_recovery_ratio: float = MIN_RECOVERY_RATIO,
    sl_buffer_atr: float = 0.25,
    min_rr: float = 1.5,
    atr: float | None = None,
) -> GrabDetection | None:
    if idx < 1 or idx + 1 >= ctx.exec_arr.length:
        return None
    atr_val = atr if atr is not None and atr > 0 else atr_at_np(ctx.exec_atr, idx)
    if atr_val <= 0:
        return None
    prior_daily_high, prior_daily_low = _prior_daily_extremes(ctx, idx)
    if prior_daily_high is None:
        return None
    bar_high = float(ctx.exec_arr.high[idx])
    if bar_high <= prior_daily_high + 1e-9:
        return None
    metrics = _funnel_metrics_short(ctx, idx, prior_daily_high)
    if metrics is None:
        return None
    wick_ratio, recovery_ratio, body_ratio = metrics
    if level >= 1 and wick_ratio < min_wick_ratio:
        return None
    if level >= 2 and recovery_ratio < min_recovery_ratio:
        return None
    sweep_distance = bar_high - prior_daily_high
    return _build_funnel_grab(
        ctx,
        idx,
        direction="SELL",
        pool_type="DAILY_HIGH",
        grab_price=bar_high,
        sweep_distance=sweep_distance,
        recovery_ratio=recovery_ratio,
        body_ratio=body_ratio,
        sl_buffer_atr=sl_buffer_atr,
        min_rr=min_rr,
        atr_val=atr_val,
    )


def detect_grab_at_index_np(
    ctx: LgrScanContext,
    idx: int,
    *,
    sl_buffer_atr: float = 0.25,
    min_rr: float = 1.5,
) -> GrabDetection | None:
    atr = atr_at_np(ctx.exec_atr, idx)
    level = lgr_detector_level()
    if level is not None:
        long_grab = detect_funnel_long_grab_np(
            ctx, idx, level, sl_buffer_atr=sl_buffer_atr, min_rr=min_rr, atr=atr
        )
        short_grab = detect_funnel_short_grab_np(
            ctx, idx, level, sl_buffer_atr=sl_buffer_atr, min_rr=min_rr, atr=atr
        )
    else:
        long_grab = detect_long_grab_np(ctx, idx, sl_buffer_atr=sl_buffer_atr, min_rr=min_rr, atr=atr)
        short_grab = detect_short_grab_np(ctx, idx, sl_buffer_atr=sl_buffer_atr, min_rr=min_rr, atr=atr)
    candidates = [g for g in (long_grab, short_grab) if g is not None and g.is_grabbed]
    if not candidates:
        return None
    return max(candidates, key=lambda g: g.grab_strength)


def _directional_metrics_np(
    arr: OhlcvArrays,
    end_idx: int,
    direction: str,
    window: int = 20,
) -> dict[str, float]:
    start = max(0, end_idx - window + 1)
    if end_idx - start < 1:
        return {"positive_close_ratio": 0.0, "directional_efficiency": 0.0, "noise_ratio": 1.0}
    closes = arr.close[start : end_idx + 1].astype(np.float64)
    opens = arr.open[start : end_idx + 1].astype(np.float64)
    net = abs(float(closes[-1] - closes[0]))
    path = float(np.abs(np.diff(closes)).sum())
    de = net / path if path > 0 else 0.0
    if direction == "BUY":
        pcr = float(np.sum(closes > opens)) / len(closes)
    else:
        pcr = float(np.sum(closes < opens)) / len(closes)
    return {
        "positive_close_ratio": round(pcr, 4),
        "directional_efficiency": round(de, 4),
        "noise_ratio": round(1.0 - de, 4),
    }


def _pullback_metrics_np(
    arr: OhlcvArrays,
    trigger_idx: int,
    direction: str,
    atr: float,
    *,
    structure_lookback: int,
) -> dict[str, float | int]:
    lookback = min(structure_lookback, trigger_idx)
    if lookback < 3 or atr <= 0:
        return {"pullback_depth": 0.0, "pullback_duration": 0, "trend_age_bars": 0}
    window = slice(trigger_idx - lookback, trigger_idx + 1)
    closes = arr.close[window].astype(np.float64)
    trend_age = 0
    for i in range(len(closes) - 1, 0, -1):
        delta = closes[i] - closes[i - 1]
        if direction == "BUY" and delta > 0:
            trend_age += 1
        elif direction == "SELL" and delta < 0:
            trend_age += 1
        else:
            break
    peak = float(np.max(arr.high[window]))
    trough = float(np.min(arr.low[window]))
    depth = (peak - trough) / atr
    lows = arr.low[window]
    highs = arr.high[window]
    if direction == "BUY":
        duration = int(np.argmax(lows[::-1] == trough)) if trough in lows else 0
    else:
        duration = int(np.argmax(highs[::-1] == peak)) if peak in highs else 0
    return {
        "pullback_depth": round(float(depth), 4),
        "pullback_duration": int(duration),
        "trend_age_bars": int(trend_age),
    }


def _stagnation_metrics_np(
    arr: OhlcvArrays,
    trigger_idx: int,
    atr: float,
    pip: float,
    *,
    stagnation_lookback: int,
) -> dict[str, float | int]:
    lookback = min(stagnation_lookback, trigger_idx)
    if lookback < 2 or atr <= 0 or pip <= 0:
        return {"stagnation_duration": 0, "stagnation_pips_width": 0.0, "stagnation_compression_ratio": 0.0}
    seg = slice(trigger_idx - lookback, trigger_idx)
    width = float(np.max(arr.high[seg]) - np.min(arr.low[seg]))
    return {
        "stagnation_duration": int(lookback),
        "stagnation_pips_width": round(width / pip, 4),
        "stagnation_compression_ratio": round(width / atr, 4),
    }


def _breakout_failure_metrics_np(
    arr: OhlcvArrays,
    trigger_idx: int,
    entry_idx: int,
    direction: str,
    atr: float,
) -> dict[str, float]:
    if trigger_idx < 0 or entry_idx >= arr.length or atr <= 0:
        return {
            "breakout_velocity": 0.0,
            "breakout_momentum_ratio": 0.0,
            "wick_ratio": 0.0,
            "rejection_ratio": 0.0,
            "close_recovery_ratio": 0.0,
        }
    high = float(arr.high[trigger_idx])
    low = float(arr.low[trigger_idx])
    open_ = float(arr.open[trigger_idx])
    close = float(arr.close[trigger_idx])
    span = max(high - low, 1e-9)
    if direction == "BUY":
        wick = min(open_, close) - low
        rejection = (close - low) / span
    else:
        wick = high - max(open_, close)
        rejection = (high - close) / span
    wick_ratio = max(0.0, min(1.0, wick / span))
    span_bars = max(1, entry_idx - trigger_idx)
    move = abs(close - open_)
    breakout_velocity = move / (span_bars * atr)
    vol = max(1.0, float(arr.volume[trigger_idx]))
    close_recovery = compute_recovery_close_ratio_np(arr, trigger_idx, trigger_idx, direction=direction)
    return {
        "breakout_velocity": round(breakout_velocity, 4),
        "breakout_momentum_ratio": round(breakout_velocity * vol, 4),
        "wick_ratio": round(wick_ratio, 4),
        "rejection_ratio": round(rejection, 4),
        "close_recovery_ratio": round(close_recovery, 4),
    }


def _impulse_metrics_np(
    arr: OhlcvArrays,
    trigger_idx: int,
    direction: str,
    atr: float,
) -> dict[str, float | int]:
    lookback = min(20, trigger_idx)
    if lookback < 3 or atr <= 0:
        return {"impulse_quality": 0.0, "impulse_atr_ratio": 0.0, "trend_age_bars": 0}
    closes = arr.close[trigger_idx - lookback : trigger_idx + 1].astype(np.float64)
    net = abs(float(closes[-1] - closes[0]))
    path = float(np.abs(np.diff(closes)).sum())
    impulse_quality = net / path if path > 0 else 0.0
    trend_age = 0
    for i in range(len(closes) - 1, 0, -1):
        delta = closes[i] - closes[i - 1]
        if direction == "BUY" and delta > 0:
            trend_age += 1
        elif direction == "SELL" and delta < 0:
            trend_age += 1
        else:
            break
    return {
        "impulse_quality": round(float(impulse_quality), 4),
        "impulse_atr_ratio": round(net / atr, 4),
        "trend_age_bars": int(trend_age),
    }


def _volume_ratio_np(arr: OhlcvArrays, bar_index: int, lookback: int) -> float:
    start = max(0, bar_index - lookback)
    window = arr.volume[start:bar_index]
    if len(window) < 2:
        return 1.0
    mean = float(np.mean(window))
    if mean <= 0:
        return 1.0
    return float(arr.volume[bar_index]) / mean


def _resolve_dow_phase_np(
    ctx: LgrScanContext,
    ts_ns: int,
    direction: str,
) -> str:
    htf = ctx.htf_cache.direction_at_ns(ts_ns)
    if htf == "NEUTRAL":
        return "RANGE"
    aligned = (direction == "BUY" and htf == "BULL") or (direction == "SELL" and htf == "BEAR")
    phase: TrendPhase = "UPTREND" if direction == "BUY" else "DOWNTREND"
    if not aligned:
        phase = "RANGE"
    return classify_dow_phase_maturity(phase, correction_bars=5)


def build_lgr_features_np(
    ctx: LgrScanContext,
    grab: GrabDetection,
    *,
    atr_period: int,
    structure_lookback: int,
    stagnation_lookback: int,
    volume_lookback: int,
    volatility_lookback: int,
    spread_pips: float,
) -> Any:
    from strategies.archive.liquidity_grab_reversal import LgrFeatures, compute_lgr_score

    trigger_idx = grab.trigger_idx
    entry_idx = grab.entry_idx
    direction = grab.direction
    entry_price = grab.entry_price
    atr = atr_at_np(ctx.exec_atr, trigger_idx)
    entry_ts_ns = int(ctx.exec_arr.datetime_ns[entry_idx])

    h1_end = asof_end_index(ctx.h1_arr, entry_ts_ns)
    h1_slice = slice_arrays(ctx.h1_arr, 0, h1_end + 1) if h1_end >= 0 else ctx.h1_arr
    h1_atr_slice = ctx.h1_atr[: h1_end + 1] if h1_end >= 0 else ctx.h1_atr
    h1_idx = max(h1_slice.length - 1, 0)
    h1_atr = atr_at_np(h1_atr_slice, h1_idx) or atr

    dir_metrics = _directional_metrics_np(ctx.exec_arr, trigger_idx, direction)
    pullback = _pullback_metrics_np(
        ctx.exec_arr, trigger_idx, direction, atr, structure_lookback=structure_lookback
    )
    stagnation = _stagnation_metrics_np(
        ctx.exec_arr, trigger_idx, atr, ctx.pip, stagnation_lookback=stagnation_lookback
    )
    breakout = _breakout_failure_metrics_np(
        ctx.exec_arr, trigger_idx, entry_idx, direction, atr
    )
    impulse = _impulse_metrics_np(ctx.exec_arr, trigger_idx, direction, atr)
    liquidity = liquidity_distances_np(
        ctx.exec_arr, entry_idx, entry_price, atr, ctx.day_norm, ctx.bar_hours
    )
    vol_pct = volatility_percentile_for_bar(
        h1_atr_slice, h1_idx, lookback=volatility_lookback, atr_period=atr_period
    )
    session = resolve_cspa_session_type_ns(entry_ts_ns)  # type: ignore[assignment]
    tick_vol = _volume_ratio_np(ctx.exec_arr, trigger_idx, volume_lookback)
    lgr_score = compute_lgr_score(
        directional_efficiency=dir_metrics["directional_efficiency"],
        positive_close_ratio=dir_metrics["positive_close_ratio"],
        recovery_ratio=grab.recovery_ratio,
        sweep_distance_pips=grab.sweep_distance_pips,
        tick_volume_ratio=tick_vol,
    )
    trend_age = max(int(pullback["trend_age_bars"]), int(impulse["trend_age_bars"]))
    dow_phase = _resolve_dow_phase_np(ctx, entry_ts_ns, direction)

    return LgrFeatures(
        lgr_score=lgr_score,
        positive_close_ratio=dir_metrics["positive_close_ratio"],
        directional_efficiency=dir_metrics["directional_efficiency"],
        noise_ratio=dir_metrics["noise_ratio"],
        dow_phase=dow_phase,  # type: ignore[arg-type]
        trend_age_bars=trend_age,
        pullback_depth=float(pullback["pullback_depth"]),
        pullback_duration=int(pullback["pullback_duration"]),
        impulse_quality=float(impulse["impulse_quality"]),
        impulse_atr_ratio=float(impulse["impulse_atr_ratio"]),
        stagnation_duration=int(stagnation["stagnation_duration"]),
        stagnation_pips_width=float(stagnation["stagnation_pips_width"]),
        stagnation_compression_ratio=float(stagnation["stagnation_compression_ratio"]),
        breakout_velocity=float(breakout["breakout_velocity"]),
        breakout_momentum_ratio=float(breakout["breakout_momentum_ratio"]),
        wick_ratio=float(breakout["wick_ratio"]),
        rejection_ratio=float(breakout["rejection_ratio"]),
        close_recovery_ratio=float(breakout["close_recovery_ratio"]),
        distance_daily_high=round(float(liquidity["distance_daily_high"]), 4),
        distance_daily_low=round(float(liquidity["distance_daily_low"]), 4),
        distance_session_high=round(float(liquidity["distance_session_high"]), 4),
        distance_session_low=round(float(liquidity["distance_session_low"]), 4),
        sweep_distance_pips=grab.sweep_distance_pips,
        liquidity_pool_type=grab.liquidity_pool_type,
        current_atr_h1=round(h1_atr, 6),
        volatility_percentile=vol_pct,
        session_type=session,  # type: ignore[arg-type]
        minutes_from_session_open=minutes_from_session_open_ns(entry_ts_ns, session),
        adr_used=round(adr_used_fraction_np(ctx.exec_arr, entry_idx, atr, ctx.day_norm), 4),
        adr_remaining=round(
            compute_adr_remaining_np(ctx.exec_arr, entry_idx, atr, ctx.day_norm), 4
        ),
        spread=spread_pips,
        spread_percentile=spread_percentile_np(spread_pips),
        tick_volume_ratio=round(tick_vol, 4),
        grab_strength=grab.grab_strength,
        recovery_ratio=grab.recovery_ratio,
    )


def _grab_to_setup_np(
    ctx: LgrScanContext,
    grab: GrabDetection,
    pair: str,
    *,
    atr_period: int,
    structure_lookback: int,
    stagnation_lookback: int,
    volume_lookback: int,
    volatility_lookback: int,
    spread_pips: float,
) -> Any | None:
    from strategies.archive.liquidity_grab_reversal import LgrSetup

    atr = atr_at_np(ctx.exec_atr, grab.trigger_idx)
    if atr <= 0:
        return None
    features = build_lgr_features_np(
        ctx,
        grab,
        atr_period=atr_period,
        structure_lookback=structure_lookback,
        stagnation_lookback=stagnation_lookback,
        volume_lookback=volume_lookback,
        volatility_lookback=volatility_lookback,
        spread_pips=spread_pips,
    )
    risk = (
        grab.entry_price - grab.stop_loss
        if grab.direction == "BUY"
        else grab.stop_loss - grab.entry_price
    )
    reward = (
        grab.take_profit - grab.entry_price
        if grab.direction == "BUY"
        else grab.entry_price - grab.take_profit
    )
    rr = reward / risk if risk > 0 else 0.0
    entry_ts = ts_ns_to_pd(int(ctx.exec_arr.datetime_ns[grab.entry_idx]))
    return LgrSetup(
        timestamp=entry_ts,
        pair=pair.upper(),
        direction=grab.direction,
        entry_price=grab.entry_price,
        stop_loss=grab.stop_loss,
        take_profit=grab.take_profit,
        risk_reward=round(rr, 4),
        trigger_bar_index=grab.trigger_idx,
        entry_bar_index=grab.entry_idx,
        grab_price=grab.grab_price,
        lgr_features=features,
        candidate_score=features.lgr_score,
        liquidity_pool_type=grab.liquidity_pool_type,
    )


def build_lgr_scan_context(
    exec_arr: OhlcvArrays,
    h1_arr: OhlcvArrays,
    pair: str,
    *,
    atr_period: int,
) -> LgrScanContext:
    exec_atr = compute_atr_np(exec_arr.high, exec_arr.low, exec_arr.close, atr_period)
    h1_atr = compute_atr_np(h1_arr.high, h1_arr.low, h1_arr.close, atr_period)
    return LgrScanContext(
        exec_arr=exec_arr,
        h1_arr=h1_arr,
        exec_atr=exec_atr,
        h1_atr=h1_atr,
        day_norm=day_norm_from_datetime_ns(exec_arr.datetime_ns),
        bar_hours=build_bar_hours(exec_arr),
        htf_cache=HtfDirectionCache(h1_arr),
        pip=pip_size_for_pair(pair),
    )


def detect_lgr_setups_for_pair_np(
    exec_arr: OhlcvArrays,
    h1_arr: OhlcvArrays,
    pair: str,
    *,
    lookback_bars: int,
    max_setups_per_day: int,
    atr_period: int,
    structure_lookback: int,
    stagnation_lookback: int,
    volume_lookback: int,
    volatility_lookback: int,
    sl_buffer_atr: float,
    min_rr: float,
    spread_pips: float,
    progress_hook: Callable[[int], None] | None = None,
    resume_from_bar: int | None = None,
    initial_setups: list[Any] | None = None,
    on_checkpoint: Callable[[int, list[Any], dict[str, Any] | None], None] | None = None,
    checkpoint_every: int = 0,
) -> list[Any]:
    from strategies.archive.liquidity_grab_reversal import ALLOWED_PAIRS

    if pair.upper() not in ALLOWED_PAIRS:
        return []
    if exec_arr.length < atr_period + 5:
        return []

    logger.info(
        "LGR numpy scan %s detector=%s bars=%d",
        pair.upper(),
        lgr_detector_mode_label(),
        exec_arr.length,
    )

    ctx = build_lgr_scan_context(exec_arr, h1_arr, pair, atr_period=atr_period)
    setups: list[Any] = list(initial_setups or [])
    daily_counts: dict[tuple[str, date], int] = {}
    for setup in setups:
        key = (setup.pair, setup.timestamp.date())
        daily_counts[key] = daily_counts.get(key, 0) + 1

    min_start = max(atr_period + 2, resume_from_bar or (atr_period + 2))
    max_start = exec_arr.length - 2
    scan_from = max(min_start, exec_arr.length - lookback_bars) if lookback_bars > 0 else min_start

    for idx in range(scan_from, max_start):
        if progress_hook is not None and (idx - scan_from) % 50 == 0:
            progress_hook(50)

        grab = detect_grab_at_index_np(ctx, idx, sl_buffer_atr=sl_buffer_atr, min_rr=min_rr)
        if grab is None:
            continue

        entry_ts = ts_ns_to_pd(int(exec_arr.datetime_ns[grab.entry_idx]))
        day_key = (pair.upper(), entry_ts.date())
        if max_setups_per_day > 0 and daily_counts.get(day_key, 0) >= max_setups_per_day:
            continue

        setup = _grab_to_setup_np(
            ctx,
            grab,
            pair,
            atr_period=atr_period,
            structure_lookback=structure_lookback,
            stagnation_lookback=stagnation_lookback,
            volume_lookback=volume_lookback,
            volatility_lookback=volatility_lookback,
            spread_pips=spread_pips,
        )
        if setup is None:
            continue

        setups.append(setup)
        daily_counts[day_key] = daily_counts.get(day_key, 0) + 1

        if on_checkpoint is not None and checkpoint_every > 0 and (idx - scan_from) % checkpoint_every == 0:
            on_checkpoint(idx, setups, None)

    if progress_hook is not None:
        remainder = (max_start - scan_from) % 50
        if remainder:
            progress_hook(remainder)

    return setups


def detect_lgr_setups_from_frames(
    df: Any,
    pair: str,
    *,
    m15_df: Any | None = None,
    h1_df: Any | None = None,
    **kwargs: Any,
) -> list[Any]:
    """Convert BT/WFT frames to ``OhlcvArrays`` and run the numpy scan."""
    exec_src = m15_df if m15_df is not None else df
    h1_src = h1_df if h1_df is not None else df
    exec_arr = as_ohlcv(exec_src)
    h1_arr = as_ohlcv(h1_src)
    return detect_lgr_setups_for_pair_np(exec_arr, h1_arr, pair, **kwargs)
