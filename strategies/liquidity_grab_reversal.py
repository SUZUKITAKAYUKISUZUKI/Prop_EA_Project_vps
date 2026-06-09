"""
Liquidity Grab Reversal (LGR) — 流動性奪取 → 失敗 → 急速反転。

Pure Data Mode: L0-L6 停止、logs/lgr_features.csv へ特徴量収集。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, fields
from datetime import date
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from strategies.base import StrategyResult
from strategies.base_strategy import BaseStrategy
from strategies.cspa import classify_dow_phase_maturity, resolve_cspa_session_type
from strategies.htf_trend_analyzer import analyze_htf_trend, clip_as_of
from strategies.liquidity_grab_detector import GrabDetection, LiquidityGrabDetector
from strategies.market_utils import compute_atr, pip_size_for_pair
from strategies.reversal_feature_helpers import (
    adr_used_fraction as _adr_used_fraction,
    compute_adr_remaining as _compute_adr_remaining,
    compute_recovery_close_ratio,
    liquidity_distances as _liquidity_distances,
    minutes_from_session_open as _minutes_from_session_open,
    spread_percentile as _spread_percentile,
    volatility_percentile as _volatility_percentile,
)

logger = logging.getLogger("lgr_strategy")

SETUP_TYPE = "LIQUIDITY_GRAB_REVERSAL"
STRATEGY_ABBREV = "LGR"
STRATEGY_ID = "liquidity_grab_reversal"
STRATEGY_FULL_NAME = "Liquidity Grab Reversal"
LGR_PAIR_PRIMARY = "GBPUSD"
LGR_PAIR_SECONDARY = "EURUSD"
ALLOWED_PAIRS = frozenset({LGR_PAIR_PRIMARY, LGR_PAIR_SECONDARY})

ATR_PERIOD = int(os.getenv("LGR_ATR_PERIOD", "14"))
LOOKBACK_BARS = int(os.getenv("LGR_LOOKBACK_BARS", "300"))
MIN_RR = float(os.getenv("LGR_MIN_RR", "1.5"))
SL_BUFFER_ATR = float(os.getenv("LGR_SL_BUFFER_ATR", "0.25"))
LGR_EXEC_BAR_MINUTES = int(os.getenv("LGR_EXEC_BAR_MINUTES", "15"))
LGR_MONITOR_BAR_MINUTES = int(os.getenv("LGR_MONITOR_BAR_MINUTES", "60"))
LGR_MAX_HOLDING_HOURS = int(os.getenv("LGR_MAX_HOLDING_HOURS", "48"))
MAX_HOLDING_BARS = max(4, (LGR_MAX_HOLDING_HOURS * 60) // LGR_EXEC_BAR_MINUTES)
MAX_SETUPS_PER_DAY = int(os.getenv("LGR_MAX_SETUPS_PER_DAY", "0"))
LGR_PRODUCTION_MAX_SETUPS_PER_DAY = 1
VOLATILITY_LOOKBACK = int(os.getenv("LGR_VOLATILITY_LOOKBACK", "120"))
ADR_LOOKBACK_DAYS = int(os.getenv("LGR_ADR_LOOKBACK_DAYS", "14"))
VOLUME_LOOKBACK = int(os.getenv("LGR_VOLUME_LOOKBACK", "20"))
LGR_BT_SPREAD_PIPS = float(os.getenv("LGR_BT_SPREAD_PIPS", "1.2"))
STRUCTURE_LOOKBACK = int(os.getenv("LGR_STRUCTURE_LOOKBACK", "20"))
STAGNATION_LOOKBACK = int(os.getenv("LGR_STAGNATION_LOOKBACK", "12"))

TradeDirection = Literal["BUY", "SELL"]
LogDirection = Literal["LONG", "SHORT"]
SessionType = Literal["ASIA", "LONDON", "NY", "OFF_HOURS"]
DowPhaseLabel = Literal["EARLY_TREND", "MATURE_TREND", "RANGE"]
LiquidityPoolType = Literal["DAILY_HIGH", "DAILY_LOW", "SESSION_HIGH", "SESSION_LOW"]

LGR_FEATURE_COLUMNS: tuple[str, ...] = (
    "lgr_score",
    "positive_close_ratio",
    "directional_efficiency",
    "noise_ratio",
    "dow_phase",
    "trend_age_bars",
    "pullback_depth",
    "pullback_duration",
    "impulse_quality",
    "impulse_atr_ratio",
    "stagnation_duration",
    "stagnation_pips_width",
    "stagnation_compression_ratio",
    "breakout_velocity",
    "breakout_momentum_ratio",
    "wick_ratio",
    "rejection_ratio",
    "close_recovery_ratio",
    "distance_daily_high",
    "distance_daily_low",
    "distance_session_high",
    "distance_session_low",
    "sweep_distance_pips",
    "liquidity_pool_type",
    "current_atr_h1",
    "volatility_percentile",
    "session_type",
    "minutes_from_session_open",
    "adr_used",
    "adr_remaining",
    "spread",
    "spread_percentile",
    "tick_volume_ratio",
)


def is_lgr_pure_data_mode() -> bool:
    """特徴量収集用: L0-L6 防御・Bayes・L4 を無効化する Pure BT。"""
    return os.getenv("LGR_PURE_DATA_MODE", "0").strip().lower() in ("1", "true", "yes", "on")


def direction_to_log_label(direction: TradeDirection) -> LogDirection:
    return "LONG" if direction == "BUY" else "SHORT"


@dataclass(frozen=True)
class LgrFeatures:
    lgr_score: float = 0.0
    positive_close_ratio: float = 0.0
    directional_efficiency: float = 0.0
    noise_ratio: float = 1.0
    dow_phase: DowPhaseLabel = "RANGE"
    trend_age_bars: int = 0
    pullback_depth: float = 0.0
    pullback_duration: int = 0
    impulse_quality: float = 0.0
    impulse_atr_ratio: float = 0.0
    stagnation_duration: int = 0
    stagnation_pips_width: float = 0.0
    stagnation_compression_ratio: float = 0.0
    breakout_velocity: float = 0.0
    breakout_momentum_ratio: float = 0.0
    wick_ratio: float = 0.0
    rejection_ratio: float = 0.0
    close_recovery_ratio: float = 0.0
    distance_daily_high: float = 0.0
    distance_daily_low: float = 0.0
    distance_session_high: float = 0.0
    distance_session_low: float = 0.0
    sweep_distance_pips: float = 0.0
    liquidity_pool_type: LiquidityPoolType = "DAILY_LOW"
    current_atr_h1: float = 0.0
    volatility_percentile: float = 50.0
    session_type: SessionType = "OFF_HOURS"
    minutes_from_session_open: int = 0
    adr_used: float = 0.0
    adr_remaining: float = 1.0
    spread: float = LGR_BT_SPREAD_PIPS
    spread_percentile: float = 50.0
    tick_volume_ratio: float = 1.0
    grab_strength: float = 0.0
    recovery_ratio: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False)


@dataclass(frozen=True)
class LgrSetup:
    timestamp: pd.Timestamp
    pair: str
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    trigger_bar_index: int
    entry_bar_index: int
    grab_price: float
    lgr_features: LgrFeatures
    reason_codes: tuple[str, ...] = ()
    candidate_score: float = 0.0
    liquidity_pool_type: LiquidityPoolType = "DAILY_LOW"


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    work = df.sort_values("datetime").reset_index(drop=True).copy()
    work["datetime"] = pd.to_datetime(work["datetime"])
    if "volume" not in work.columns:
        work["volume"] = 0.0
    return work


def _atr_at(work: pd.DataFrame, bar_index: int, atr_series: pd.Series | None = None) -> float:
    if bar_index < 0 or bar_index >= len(work):
        return 0.0
    if atr_series is not None:
        val = float(atr_series.iloc[bar_index])
        return val if np.isfinite(val) and val > 0 else 0.0
    clipped = work.iloc[: bar_index + 1]
    atr = compute_atr(clipped, ATR_PERIOD)
    if bar_index >= len(atr):
        return 0.0
    val = float(atr.iloc[bar_index])
    return val if np.isfinite(val) else 0.0


def _directional_metrics(work: pd.DataFrame, end_idx: int, direction: TradeDirection, window: int = 20) -> dict[str, float]:
    start = max(0, end_idx - window + 1)
    seg = work.iloc[start : end_idx + 1]
    if len(seg) < 2:
        return {"positive_close_ratio": 0.0, "directional_efficiency": 0.0, "noise_ratio": 1.0}
    closes = seg["close"].astype(float).values
    net = abs(float(closes[-1] - closes[0]))
    path = float(np.abs(np.diff(closes)).sum())
    de = net / path if path > 0 else 0.0
    if direction == "BUY":
        bull = sum(1 for _, row in seg.iterrows() if float(row["close"]) > float(row["open"]))
        pcr = bull / len(seg)
    else:
        bear = sum(1 for _, row in seg.iterrows() if float(row["close"]) < float(row["open"]))
        pcr = bear / len(seg)
    return {
        "positive_close_ratio": round(pcr, 4),
        "directional_efficiency": round(de, 4),
        "noise_ratio": round(1.0 - de, 4),
    }


def _pullback_metrics(work: pd.DataFrame, trigger_idx: int, direction: TradeDirection, atr: float) -> dict[str, float | int]:
    lookback = min(STRUCTURE_LOOKBACK, trigger_idx)
    if lookback < 3 or atr <= 0:
        return {"pullback_depth": 0.0, "pullback_duration": 0, "trend_age_bars": 0}
    window = work.iloc[trigger_idx - lookback : trigger_idx + 1]
    closes = window["close"].astype(float).values
    trend_age = 0
    for i in range(len(closes) - 1, 0, -1):
        delta = closes[i] - closes[i - 1]
        if direction == "BUY" and delta > 0:
            trend_age += 1
        elif direction == "SELL" and delta < 0:
            trend_age += 1
        else:
            break
    peak = float(window["high"].max())
    trough = float(window["low"].min())
    depth = (peak - trough) / atr
    if direction == "BUY":
        duration = int(np.argmax(window["low"].values[::-1] == trough)) if trough in window["low"].values else 0
    else:
        duration = int(np.argmax(window["high"].values[::-1] == peak)) if peak in window["high"].values else 0
    return {
        "pullback_depth": round(float(depth), 4),
        "pullback_duration": int(duration),
        "trend_age_bars": int(trend_age),
    }


def _stagnation_metrics(work: pd.DataFrame, trigger_idx: int, atr: float, pip: float) -> dict[str, float | int]:
    lookback = min(STAGNATION_LOOKBACK, trigger_idx)
    if lookback < 2 or atr <= 0 or pip <= 0:
        return {"stagnation_duration": 0, "stagnation_pips_width": 0.0, "stagnation_compression_ratio": 0.0}
    seg = work.iloc[trigger_idx - lookback : trigger_idx]
    width = float(seg["high"].max() - seg["low"].min())
    width_pips = width / pip
    compression = width / atr if atr > 0 else 0.0
    return {
        "stagnation_duration": int(len(seg)),
        "stagnation_pips_width": round(width_pips, 4),
        "stagnation_compression_ratio": round(compression, 4),
    }


def _breakout_failure_metrics(
    work: pd.DataFrame,
    trigger_idx: int,
    entry_idx: int,
    direction: TradeDirection,
    atr: float,
) -> dict[str, float]:
    if trigger_idx < 0 or entry_idx >= len(work) or atr <= 0:
        return {
            "breakout_velocity": 0.0,
            "breakout_momentum_ratio": 0.0,
            "wick_ratio": 0.0,
            "rejection_ratio": 0.0,
            "close_recovery_ratio": 0.0,
        }
    row = work.iloc[trigger_idx]
    high = float(row["high"])
    low = float(row["low"])
    open_ = float(row["open"])
    close = float(row["close"])
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
    vol = max(1.0, float(row["volume"]))
    close_recovery = compute_recovery_close_ratio(work, trigger_idx, trigger_idx, direction=direction)
    return {
        "breakout_velocity": round(breakout_velocity, 4),
        "breakout_momentum_ratio": round(breakout_velocity * vol, 4),
        "wick_ratio": round(wick_ratio, 4),
        "rejection_ratio": round(rejection, 4),
        "close_recovery_ratio": round(close_recovery, 4),
    }


def _impulse_metrics(work: pd.DataFrame, trigger_idx: int, direction: TradeDirection, atr: float) -> dict[str, float | int]:
    lookback = min(20, trigger_idx)
    if lookback < 3 or atr <= 0:
        return {"impulse_quality": 0.0, "impulse_atr_ratio": 0.0, "trend_age_bars": 0}
    window = work.iloc[trigger_idx - lookback : trigger_idx + 1]
    closes = window["close"].astype(float).values
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


def _volume_ratio(work: pd.DataFrame, bar_index: int, lookback: int = VOLUME_LOOKBACK) -> float:
    start = max(0, bar_index - lookback)
    window = work.iloc[start:bar_index]["volume"].astype(float)
    if len(window) < 2:
        return 1.0
    mean = float(window.mean())
    if mean <= 0:
        return 1.0
    return float(work.iloc[bar_index]["volume"]) / mean


def _adr_used_fraction(work: pd.DataFrame, bar_index: int, atr: float) -> float:
    if bar_index < 1 or atr <= 0:
        return 0.0
    remaining = _compute_adr_remaining(work, bar_index, atr)
    return max(0.0, min(1.0, 1.0 - remaining))


def _sweep_distance_score(sweep_pips: float) -> float:
    return min(max(sweep_pips / 3.0, 0.0), 1.0)


def _volume_score(tick_volume_ratio: float) -> float:
    return min(max(tick_volume_ratio / 2.0, 0.0), 1.0)


def compute_lgr_score(
    *,
    directional_efficiency: float,
    positive_close_ratio: float,
    recovery_ratio: float,
    sweep_distance_pips: float,
    tick_volume_ratio: float,
) -> float:
    sweep_score = _sweep_distance_score(sweep_distance_pips)
    volume_score = _volume_score(tick_volume_ratio)
    raw = (
        0.30 * directional_efficiency
        + 0.25 * positive_close_ratio
        + 0.20 * recovery_ratio
        + 0.15 * sweep_score
        + 0.10 * volume_score
    )
    return round(raw * 100.0, 2)


def _resolve_dow_phase(h1_work: pd.DataFrame, timestamp: pd.Timestamp, direction: TradeDirection) -> DowPhaseLabel:
    from strategies.cspa import TrendPhase

    htf = analyze_htf_trend(h1_work, timestamp)
    if htf.direction == "NEUTRAL":
        return "RANGE"
    aligned = (direction == "BUY" and htf.direction == "BULL") or (direction == "SELL" and htf.direction == "BEAR")
    phase: TrendPhase = "UPTREND" if direction == "BUY" else "DOWNTREND"
    if not aligned:
        phase = "RANGE"
    return classify_dow_phase_maturity(phase, correction_bars=5)


def build_lgr_features(
    *,
    exec_work: pd.DataFrame,
    h1_work: pd.DataFrame,
    grab: GrabDetection,
    pair: str,
    atr: float,
    atr_series: pd.Series,
    spread_pips: float = LGR_BT_SPREAD_PIPS,
) -> LgrFeatures:
    trigger_idx = grab.trigger_idx
    entry_idx = grab.entry_idx
    direction = grab.direction
    entry_price = grab.entry_price
    timestamp = pd.Timestamp(exec_work.iloc[entry_idx]["datetime"])
    pip = pip_size_for_pair(pair)

    dir_metrics = _directional_metrics(exec_work, trigger_idx, direction)
    pullback = _pullback_metrics(exec_work, trigger_idx, direction, atr)
    stagnation = _stagnation_metrics(exec_work, trigger_idx, atr, pip)
    breakout = _breakout_failure_metrics(exec_work, trigger_idx, entry_idx, direction, atr)
    impulse = _impulse_metrics(exec_work, trigger_idx, direction, atr)
    liquidity = _liquidity_distances(exec_work, entry_idx, entry_price, atr)

    h1_atr_series = compute_atr(h1_work, ATR_PERIOD) if len(h1_work) >= ATR_PERIOD + 1 else atr_series
    h1_idx = len(h1_work) - 1
    h1_atr = _atr_at(h1_work, h1_idx, h1_atr_series) or atr
    vol_pct = _volatility_percentile(h1_work, h1_idx, h1_atr_series)
    session = resolve_cspa_session_type(timestamp)
    tick_vol = _volume_ratio(exec_work, trigger_idx)
    lgr_score = compute_lgr_score(
        directional_efficiency=dir_metrics["directional_efficiency"],
        positive_close_ratio=dir_metrics["positive_close_ratio"],
        recovery_ratio=grab.recovery_ratio,
        sweep_distance_pips=grab.sweep_distance_pips,
        tick_volume_ratio=tick_vol,
    )

    trend_age = max(int(pullback["trend_age_bars"]), int(impulse["trend_age_bars"]))
    dow_phase = _resolve_dow_phase(h1_work, timestamp, direction)

    return LgrFeatures(
        lgr_score=lgr_score,
        positive_close_ratio=dir_metrics["positive_close_ratio"],
        directional_efficiency=dir_metrics["directional_efficiency"],
        noise_ratio=dir_metrics["noise_ratio"],
        dow_phase=dow_phase,
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
        session_type=session,
        minutes_from_session_open=_minutes_from_session_open(timestamp, session),
        adr_used=round(_adr_used_fraction(exec_work, entry_idx, atr), 4),
        adr_remaining=round(_compute_adr_remaining(exec_work, entry_idx, atr), 4),
        spread=spread_pips,
        spread_percentile=_spread_percentile(spread_pips),
        tick_volume_ratio=round(tick_vol, 4),
        grab_strength=grab.grab_strength,
        recovery_ratio=grab.recovery_ratio,
    )


def _grab_to_setup(
    exec_work: pd.DataFrame,
    h1_work: pd.DataFrame,
    grab: GrabDetection,
    pair: str,
    atr_series: pd.Series,
) -> LgrSetup | None:
    atr = _atr_at(exec_work, grab.trigger_idx, atr_series)
    if atr <= 0:
        return None
    features = build_lgr_features(
        exec_work=exec_work,
        h1_work=h1_work,
        grab=grab,
        pair=pair,
        atr=atr,
        atr_series=atr_series,
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
    return LgrSetup(
        timestamp=pd.Timestamp(exec_work.iloc[grab.entry_idx]["datetime"]),
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


def detect_lgr_setups_for_pair(
    df: pd.DataFrame,
    pair: str,
    *,
    m15_df: pd.DataFrame | None = None,
    h1_df: pd.DataFrame | None = None,
    lookback_bars: int = LOOKBACK_BARS,
    max_setups_per_day: int = MAX_SETUPS_PER_DAY,
    progress_hook: Callable[[int], None] | None = None,
    resume_from_bar: int | None = None,
    initial_setups: list[LgrSetup] | None = None,
    on_checkpoint: Callable[[int, list[LgrSetup], dict[str, Any] | None], None] | None = None,
    checkpoint_every: int = 0,
) -> list[LgrSetup]:
    """H1 コンテキスト + M15 トリガーで LGR セットアップをスキャン。"""
    if pair.upper() not in ALLOWED_PAIRS:
        return []

    exec_work = _prepare_df(m15_df if m15_df is not None else df)
    h1_work = _prepare_df(h1_df if h1_df is not None else df)
    if len(exec_work) < ATR_PERIOD + 5:
        return []

    detector = LiquidityGrabDetector(sl_buffer_atr=SL_BUFFER_ATR, min_rr=MIN_RR)
    atr_series = compute_atr(exec_work, ATR_PERIOD)
    setups: list[LgrSetup] = list(initial_setups or [])
    daily_counts: dict[tuple[str, date], int] = {}
    for setup in setups:
        key = (setup.pair, setup.timestamp.date())
        daily_counts[key] = daily_counts.get(key, 0) + 1

    min_start = max(ATR_PERIOD + 2, resume_from_bar or (ATR_PERIOD + 2))
    max_start = len(exec_work) - 2
    scan_from = max(min_start, len(exec_work) - lookback_bars) if lookback_bars > 0 else min_start

    for idx in range(scan_from, max_start):
        if progress_hook is not None and (idx - scan_from) % 50 == 0:
            progress_hook(50)

        grab = detector.detect_at_index(exec_work, idx, pair, _atr_at(exec_work, idx, atr_series))
        if grab is None:
            continue

        ts = pd.Timestamp(exec_work.iloc[grab.entry_idx]["datetime"])
        day_key = (pair.upper(), ts.date())
        if max_setups_per_day > 0 and daily_counts.get(day_key, 0) >= max_setups_per_day:
            continue

        h1_clipped = clip_as_of(h1_work, ts)
        setup = _grab_to_setup(exec_work, h1_clipped, grab, pair, atr_series)
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


def compute_lgr_trade_excursions(
    pair_df: pd.DataFrame,
    start_index: int,
    entry: float,
    stop_loss: float,
    take_profit: float,
    *,
    max_holding_bars: int = MAX_HOLDING_BARS,
    direction: TradeDirection = "BUY",
) -> dict[str, float | str | int]:
    from strategies.bt_l5 import compute_trade_excursions_np
    from strategies.bt_ohlcv import as_ohlcv

    ohlcv = as_ohlcv(pair_df)
    return compute_trade_excursions_np(
        ohlcv,
        start_index,
        entry,
        stop_loss,
        take_profit,
        max_holding_bars=max_holding_bars,
        direction=direction,
    )


def build_lgr_feature_log_row(
    *,
    trade_id: str,
    setup: LgrSetup,
    trade_result: str,
    profit_r: float,
) -> dict[str, Any]:
    feat = setup.lgr_features.as_dict()
    return {
        "trade_id": trade_id,
        "timestamp": setup.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "pair": setup.pair,
        "direction": direction_to_log_label(setup.direction),
        "trade_result": trade_result,
        "profit_r": round(float(profit_r), 4),
        **{k: feat.get(k, "") for k in LGR_FEATURE_COLUMNS},
    }


class LiquidityGrabReversalStrategy(BaseStrategy):
    """LGR — H1 コンテキスト + M15 Liquidity Grab 反転。"""

    STRATEGY_ID = STRATEGY_ID
    SETUP_TYPE = SETUP_TYPE

    def __init__(self, weights_config: dict[str, int], mode_h1: bool = True, **kwargs: Any) -> None:
        super().__init__(weights_config, mode_h1)
        self.detector = LiquidityGrabDetector(sl_buffer_atr=SL_BUFFER_ATR, min_rr=MIN_RR)
        self.is_pure_data_mode = kwargs.get("is_lgr_pure_data_mode", is_lgr_pure_data_mode())

    @property
    def setup_type(self) -> str:
        return SETUP_TYPE

    def detect_setups(
        self,
        df: pd.DataFrame,
        pair_name: str,
        h1_df: pd.DataFrame | None = None,
    ) -> list[LgrSetup]:
        if h1_df is not None:
            return detect_lgr_setups_for_pair(h1_df, pair_name, m15_df=df)
        return detect_lgr_setups_for_pair(df, pair_name)

    def analyze_setup(
        self,
        setup: LgrSetup,
        gbp_setup: LgrSetup | None,
        eur_setup: LgrSetup | None,
        h1_gbp: pd.DataFrame,
        h1_eur: pd.DataFrame,
    ) -> StrategyResult:
        h1_ref = h1_gbp if setup.pair == LGR_PAIR_PRIMARY else h1_eur
        htf = analyze_htf_trend(h1_ref, setup.timestamp)
        raw: dict[str, Any] = {
            **setup.lgr_features.as_dict(),
            "htf_trend_direction": htf.direction,
            "candidate_score": setup.candidate_score,
        }
        if self.is_pure_data_mode or is_lgr_pure_data_mode():
            decision = "ALLOW"
        else:
            decision = "PENDING_FILTERS"
        return StrategyResult(
            is_setup=True,
            setup_type=self.setup_type,
            direction=setup.direction,
            strategy_action=decision,
            entry_price=setup.entry_price,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            candidate_score=setup.candidate_score,
            raw_features=raw,
        )


__all__ = [
    "LGR_FEATURE_COLUMNS",
    "LGR_PAIR_PRIMARY",
    "LGR_PAIR_SECONDARY",
    "LgrFeatures",
    "LgrSetup",
    "LiquidityGrabReversalStrategy",
    "SETUP_TYPE",
    "STRATEGY_ABBREV",
    "STRATEGY_FULL_NAME",
    "STRATEGY_ID",
    "ALLOWED_PAIRS",
    "MAX_SETUPS_PER_DAY",
    "LGR_EXEC_BAR_MINUTES",
    "build_lgr_feature_log_row",
    "build_lgr_features",
    "compute_lgr_score",
    "compute_lgr_trade_excursions",
    "detect_lgr_setups_for_pair",
    "direction_to_log_label",
    "is_lgr_pure_data_mode",
]
