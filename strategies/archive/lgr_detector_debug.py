"""
LGR Detector — 段階的デバッグファネル (Level 0〜3).

Level 0: 日足高値更新 **または** 日足安値更新
Level 1: + wick_ratio
Level 2: + recovery_ratio
Level 3: + bull/bear close (実体比率)
"""

from __future__ import annotations

from dataclasses import dataclass

from strategies.archive.lgr_scan_hot import (
    LgrScanContext,
    _prior_daily_extremes,
    atr_at_np,
    build_lgr_scan_context,
)
from strategies.archive.liquidity_grab_detector import (
    MIN_RECOVERY_RATIO,
    MIN_SWEEP_ATR,
    STRONG_CLOSE_BODY_RATIO,
)
from strategies.bt_ohlcv import OhlcvArrays, as_ohlcv

DEFAULT_MIN_WICK_RATIO = 0.35


@dataclass
class DirectionFunnelCounts:
    """LONG=日足安値更新側 / SHORT=日足高値更新側"""

    level0: int = 0
    level1: int = 0
    level2: int = 0
    level3: int = 0


@dataclass
class PairFunnelReport:
    pair: str
    bars_scanned: int
    level0_combined: int
    level0_daily_high: int
    level0_daily_low: int
    long_funnel: DirectionFunnelCounts
    short_funnel: DirectionFunnelCounts


def _bar_metrics_long(
    ctx: LgrScanContext,
    idx: int,
    prior_daily_low: float,
) -> tuple[float, float, float, bool] | None:
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
    body_ratio = (bar_close - bar_open) / span
    bull_close = bar_close > bar_open and body_ratio >= STRONG_CLOSE_BODY_RATIO
    return wick_ratio, recovery_ratio, body_ratio, bull_close


def _bar_metrics_short(
    ctx: LgrScanContext,
    idx: int,
    prior_daily_high: float,
) -> tuple[float, float, float, bool] | None:
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
    body_ratio = (bar_open - bar_close) / span
    bear_close = bar_close < bar_open and body_ratio >= STRONG_CLOSE_BODY_RATIO
    return wick_ratio, recovery_ratio, body_ratio, bear_close


def run_detector_funnel(
    exec_arr: OhlcvArrays,
    pair: str,
    *,
    h1_arr: OhlcvArrays | None = None,
    atr_period: int = 14,
    min_wick_ratio: float = DEFAULT_MIN_WICK_RATIO,
    min_recovery_ratio: float = MIN_RECOVERY_RATIO,
    scan_start: int | None = None,
    scan_end: int | None = None,
) -> PairFunnelReport:
    """
    M15 全バー（または部分範囲）で Level 0〜3 の通過件数を集計。
    """
    ctx = build_lgr_scan_context(
        exec_arr,
        h1_arr if h1_arr is not None else exec_arr,
        pair,
        atr_period=atr_period,
    )
    long_counts = DirectionFunnelCounts()
    short_counts = DirectionFunnelCounts()
    level0_high = 0
    level0_low = 0

    start = scan_start if scan_start is not None else max(atr_period + 2, 1)
    end = scan_end if scan_end is not None else exec_arr.length - 2

    for idx in range(start, end):
        prior_daily_high, prior_daily_low = _prior_daily_extremes(ctx, idx)
        if prior_daily_high is None or prior_daily_low is None:
            continue

        arr = ctx.exec_arr
        bar_high = float(arr.high[idx])
        bar_low = float(arr.low[idx])

        daily_high_update = bar_high > prior_daily_high + 1e-9
        daily_low_update = bar_low < prior_daily_low - 1e-9

        if daily_high_update:
            level0_high += 1
        if daily_low_update:
            level0_low += 1

        if daily_low_update:
            long_counts.level0 += 1
            metrics = _bar_metrics_long(ctx, idx, prior_daily_low)
            if metrics is None:
                continue
            wick_ratio, recovery_ratio, _, bull_close = metrics
            if wick_ratio >= min_wick_ratio:
                long_counts.level1 += 1
            if wick_ratio >= min_wick_ratio and recovery_ratio >= min_recovery_ratio:
                long_counts.level2 += 1
            if (
                wick_ratio >= min_wick_ratio
                and recovery_ratio >= min_recovery_ratio
                and bull_close
            ):
                long_counts.level3 += 1

        if daily_high_update:
            short_counts.level0 += 1
            metrics = _bar_metrics_short(ctx, idx, prior_daily_high)
            if metrics is None:
                continue
            wick_ratio, recovery_ratio, _, bear_close = metrics
            if wick_ratio >= min_wick_ratio:
                short_counts.level1 += 1
            if wick_ratio >= min_wick_ratio and recovery_ratio >= min_recovery_ratio:
                short_counts.level2 += 1
            if (
                wick_ratio >= min_wick_ratio
                and recovery_ratio >= min_recovery_ratio
                and bear_close
            ):
                short_counts.level3 += 1

    return PairFunnelReport(
        pair=pair.upper(),
        bars_scanned=max(end - start, 0),
        level0_combined=level0_high + level0_low,
        level0_daily_high=level0_high,
        level0_daily_low=level0_low,
        long_funnel=long_counts,
        short_funnel=short_counts,
    )


def format_funnel_report(
    report: PairFunnelReport,
    *,
    min_wick_ratio: float = DEFAULT_MIN_WICK_RATIO,
    min_recovery_ratio: float = MIN_RECOVERY_RATIO,
) -> str:
    lines = [
        f"=== {report.pair} (scanned {report.bars_scanned:,} M15 bars) ===",
        "",
        "Level 0 - daily HIGH update OR daily LOW update",
        f"  combined (high + low, double-count if both): {report.level0_combined:,}",
        f"  daily HIGH update only:                  {report.level0_daily_high:,}",
        f"  daily LOW  update only:                  {report.level0_daily_low:,}",
        "",
        "LONG funnel (daily LOW update -> wick -> recovery -> bull close)",
        f"  Level 0  daily low update:               {report.long_funnel.level0:,}",
        f"  Level 1  + wick_ratio >= {min_wick_ratio:.2f}:        {report.long_funnel.level1:,}",
        f"  Level 2  + recovery_ratio >= {min_recovery_ratio:.2f}:   {report.long_funnel.level2:,}",
        f"  Level 3  + bull close (body>={STRONG_CLOSE_BODY_RATIO:.2f}): {report.long_funnel.level3:,}",
        "",
        "SHORT funnel (daily HIGH update -> wick -> recovery -> bear close)",
        f"  Level 0  daily high update:              {report.short_funnel.level0:,}",
        f"  Level 1  + wick_ratio >= {min_wick_ratio:.2f}:        {report.short_funnel.level1:,}",
        f"  Level 2  + recovery_ratio >= {min_recovery_ratio:.2f}:   {report.short_funnel.level2:,}",
        f"  Level 3  + bear close (body>={STRONG_CLOSE_BODY_RATIO:.2f}): {report.short_funnel.level3:,}",
    ]
    return "\n".join(lines)


def run_funnel_from_frames(
    m15_frame: object,
    pair: str,
    *,
    h1_frame: object | None = None,
    **kwargs: object,
) -> PairFunnelReport:
    exec_arr = as_ohlcv(m15_frame)
    h1_arr = as_ohlcv(h1_frame) if h1_frame is not None else exec_arr
    return run_detector_funnel(exec_arr, pair, h1_arr=h1_arr, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "DirectionFunnelCounts",
    "PairFunnelReport",
    "format_funnel_report",
    "run_detector_funnel",
    "run_funnel_from_frames",
]
