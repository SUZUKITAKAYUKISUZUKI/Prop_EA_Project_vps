"""
strategies/lbo.py — London Break Out (LBO): Tokyo range + London breakout retest.

Pure BT / feature-collection phase — no candidate_score weighting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd

from strategies.base import StrategyResult
from strategies.base_strategy import BaseStrategy
from strategies.dbbs_common import (
    compute_adr_metrics,
    compute_rr_ratio,
    day_index_from_timestamps,
    ohlcv_to_arrays,
    passes_entry_stop_take_geometry,
    precompute_atr_series,
)
from strategies.htf_trend_analyzer import analyze_htf_trend, clip_as_of, is_counter_trend, resample_to_htf
from strategies.lbo_detector import (
    BREAKOUT_WINDOW_END_JST,
    JST_OFFSET_NS,
    LONDON_START_JST,
    MAX_RETEST_BARS,
    MIN_BREAKOUT_PIPS,
    MIN_TOKYO_RANGE_PIPS,
    NS_PER_DAY,
    NS_PER_SEC,
    TOKYO_END_JST,
    TokyoRange,
    detect_breakout_bar_core,
    detect_retest,
    detect_tokyo_range,
    jst_day_ordinal,
    jst_fields_from_ns,
    jst_hour_minute,
    london_bars_before_breakout,
    london_range_before_break_pips,
    past_tokyo_range_pips,
)
from strategies.market_utils import PIP_SIZE, calc_smt_features, pip_size_for_pair

SETUP_TYPE = "LBO"
STRATEGY_ABBREV = "LBO"
STRATEGY_FULL_NAME = "London Break Out"
ALLOWED_PAIRS = frozenset({"GBPUSD", "EURUSD"})
LBO_PAIR_PRIMARY = "EURUSD"
LBO_PAIR_SECONDARY = "GBPUSD"
EXEC_BAR_MINUTES = 15
STRUCTURE_BAR_MINUTES = 60
LBO_BAYES_PURE_PROB = 1.0
MIN_RR_RATIO = 1.5
TARGET_RR = 2.0
ADR_REJECT_THRESHOLD = 0.90
ATR_PERIOD = 14
Direction = Literal["BUY", "SELL"]

LBO_L6_EXTRA_COLUMNS: tuple[str, ...] = (
    "tokyo_range_pips",
    "tokyo_range_atr_ratio",
    "tokyo_range_percentile",
    "breakout_candle_atr_ratio",
    "breakout_momentum",
    "both_broke",
    "retest_depth_ratio",
    "retest_duration_bars",
    "retest_pa_type",
    "london_open_bars_elapsed",
    "htf_aligned",
    "smt_intensity",
    "tokyo_range_compression_score",
    "breakout_distance_from_tokyo_mid",
    "tick_volume_ratio",
    "breakout_trade_density",
    "adr_expansion_today",
    "leader_lag_bars",
    "retest_count",
    "h4_distance_from_ma",
    "d1_distance_from_ma",
)

LBO_FEATURE_COLUMNS: tuple[str, ...] = (
    "trade_id",
    "timestamp",
    "pair",
    "direction",
    *LBO_L6_EXTRA_COLUMNS,
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
)


def is_lbo_pure_data_mode() -> bool:
    raw = os.getenv("LBO_PURE_DATA_MODE", "1")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def is_lbo_enabled() -> bool:
    raw = os.getenv("LBO_ENABLED", "1")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def is_lbo_l4_bypass() -> bool:
    return is_lbo_pure_data_mode()


def is_lbo_generic_bayes_bypass() -> bool:
    return is_lbo_pure_data_mode()


def is_lbo_defense_pure_mode() -> bool:
    """LBO 防御レイヤー純粋モード（``LBO_PURE_DATA_MODE=1``）。"""
    return is_lbo_pure_data_mode()


@dataclass
class LboBayesFeatures:
    tokyo_range_pips: float = 0.0
    tokyo_range_atr_ratio: float = 0.0
    tokyo_range_percentile: float = 0.5
    tokyo_formation_bars: int = 0
    tokyo_high_low_symmetry: float = 0.0
    tokyo_range_compression_score: float = 1.0
    breakout_distance_from_tokyo_mid: float = 0.0
    breakout_candle_atr_ratio: float = 0.0
    breakout_close_distance_pips: float = 0.0
    breakout_momentum: float = 0.0
    breakout_volume_ratio: float = 1.0
    london_open_bars_elapsed: int = 0
    both_broke: bool = False
    tick_volume_ratio: float = 1.0
    breakout_trade_density: float = 0.0
    retest_depth_ratio: float = 0.0
    retest_duration_bars: int = 0
    retest_rejection_strength: float = 0.0
    retest_pa_type: str = "CLOSE_ONLY"
    retest_count: int = 0
    volatility_percentile: float = 0.5
    adr_used_at_entry: float = 0.0
    adr_remaining: float = 0.0
    adr_expansion_today: float = 1.0
    session_minutes_elapsed: int = 0
    htf_h4_direction: str = "NEUTRAL"
    htf_d1_direction: str = "NEUTRAL"
    htf_aligned: bool = False
    h4_distance_from_ma: float = 0.0
    d1_distance_from_ma: float = 0.0
    smt_intensity: float = 0.0
    leader_lag_bars: int = 0
    distance_to_daily_high: float = 0.0
    distance_to_daily_low: float = 0.0
    outcome_label: str = ""
    result_r: float = 0.0
    mfe: float = 0.0
    mae: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "tokyo_range_pips": self.tokyo_range_pips,
            "tokyo_range_atr_ratio": self.tokyo_range_atr_ratio,
            "tokyo_range_percentile": self.tokyo_range_percentile,
            "tokyo_formation_bars": self.tokyo_formation_bars,
            "tokyo_high_low_symmetry": self.tokyo_high_low_symmetry,
            "tokyo_range_compression_score": self.tokyo_range_compression_score,
            "breakout_distance_from_tokyo_mid": self.breakout_distance_from_tokyo_mid,
            "breakout_candle_atr_ratio": self.breakout_candle_atr_ratio,
            "breakout_close_distance_pips": self.breakout_close_distance_pips,
            "breakout_momentum": self.breakout_momentum,
            "breakout_volume_ratio": self.breakout_volume_ratio,
            "london_open_bars_elapsed": self.london_open_bars_elapsed,
            "both_broke": self.both_broke,
            "tick_volume_ratio": self.tick_volume_ratio,
            "breakout_trade_density": self.breakout_trade_density,
            "retest_depth_ratio": self.retest_depth_ratio,
            "retest_duration_bars": self.retest_duration_bars,
            "retest_rejection_strength": self.retest_rejection_strength,
            "retest_pa_type": self.retest_pa_type,
            "retest_count": self.retest_count,
            "volatility_percentile": self.volatility_percentile,
            "adr_used_at_entry": self.adr_used_at_entry,
            "adr_remaining": self.adr_remaining,
            "adr_expansion_today": self.adr_expansion_today,
            "session_minutes_elapsed": self.session_minutes_elapsed,
            "htf_h4_direction": self.htf_h4_direction,
            "htf_d1_direction": self.htf_d1_direction,
            "htf_aligned": self.htf_aligned,
            "h4_distance_from_ma": self.h4_distance_from_ma,
            "d1_distance_from_ma": self.d1_distance_from_ma,
            "smt_intensity": self.smt_intensity,
            "leader_lag_bars": self.leader_lag_bars,
            "distance_to_daily_high": self.distance_to_daily_high,
            "distance_to_daily_low": self.distance_to_daily_low,
            "bayes_probability": LBO_BAYES_PURE_PROB,
            "outcome_label": self.outcome_label,
            "result_r": self.result_r,
            "mfe": self.mfe,
            "mae": self.mae,
        }


@dataclass(frozen=True)
class LboSetup:
    timestamp: pd.Timestamp
    pair: str
    direction: Direction
    entry_price: float
    stop_loss: float
    take_profit: float
    tokyo_high: float
    tokyo_low: float
    tokyo_range_pips: float
    tokyo_range_atr_ratio: float
    tokyo_range_percentile: float
    tokyo_range_compression_score: float
    breakout_distance_from_tokyo_mid: float
    breakout_candle_size_pips: float
    breakout_candle_atr_ratio: float
    breakout_close_distance_pips: float
    breakout_momentum: float
    tick_volume_ratio: float
    breakout_trade_density: float
    retest_depth_pips: float
    retest_depth_ratio: float
    retest_duration_bars: int
    retest_rejection_strength: float
    retest_pa_type: str
    retest_count: int
    london_open_bars_elapsed: int
    london_range_before_break: float
    atr: float
    volatility_percentile: float
    adr_used_at_entry: float
    adr_remaining: float
    session_minutes_elapsed: int
    adr_expansion_today: float
    htf_h4_direction: str
    htf_aligned: bool
    htf_d1_direction: str
    h4_distance_from_ma: float
    d1_distance_from_ma: float
    distance_to_daily_high: float
    distance_to_daily_low: float
    smt_intensity: float
    smt_leader: str
    both_broke: bool
    leader_lag_bars: int
    bar_index_m15: int
    bar_index_h1: int
    jst_session_date: date
    breakout_bar_index: int
    bayes_features: LboBayesFeatures
    rr_ratio: float
    bayes_probability: float = LBO_BAYES_PURE_PROB
    setup_type: str = SETUP_TYPE

    @property
    def sweep_distance(self) -> float:
        return self.breakout_close_distance_pips * PIP_SIZE


def _trend_to_htf_label(direction: str) -> str:
    if direction == "BULL":
        return "UP"
    if direction == "BEAR":
        return "DOWN"
    return "FLAT"


def _is_htf_aligned(trade_dir: str, htf_dir: str) -> bool:
    if htf_dir == "FLAT" or htf_dir == "NEUTRAL":
        return False
    if trade_dir == "BUY":
        return htf_dir in ("UP", "BULL")
    return htf_dir in ("DOWN", "BEAR")


def _distance_from_ma(close: float, ma: float | None, pip: float) -> float:
    if ma is None or ma <= 0.0:
        return 0.0
    return abs(close - ma) / pip


def _compute_sl_tp(
    direction: Direction,
    entry: float,
    tokyo_high: float,
    tokyo_low: float,
    atr: float,
    pip: float,
    *,
    daily_high: float,
    daily_low: float,
) -> tuple[float, float, float] | None:
    buffer = atr * 0.3 if atr > 0.0 else pip * 3.0
    if direction == "BUY":
        stop = tokyo_high - buffer
        risk = entry - stop
        if risk <= 0.0:
            return None
        take = entry + risk * TARGET_RR
        tp_dist = (take - entry) / pip
        dist_to_daily_high = max(daily_high - entry, 0.0) / pip
        if dist_to_daily_high < tp_dist:
            return None
    else:
        stop = tokyo_low + buffer
        risk = stop - entry
        if risk <= 0.0:
            return None
        take = entry - risk * TARGET_RR
        tp_dist = (entry - take) / pip
        dist_to_daily_low = max(entry - daily_low, 0.0) / pip
        if dist_to_daily_low < tp_dist:
            return None
    rr = compute_rr_ratio(entry, stop, take)
    if rr < MIN_RR_RATIO:
        return None
    if not passes_entry_stop_take_geometry(direction, entry, stop, take, atr=atr, pip=pip, min_rr=MIN_RR_RATIO):
        return None
    return stop, take, rr


def _session_minutes_elapsed(jst_hour: int, jst_minute: int) -> int:
    if jst_hour < LONDON_START_JST:
        return 0
    return max(0, (jst_hour - LONDON_START_JST) * 60 + jst_minute)


def _build_features_for_setup(
    *,
    tokyo: TokyoRange,
    percentile: float,
    compression: float,
    mid_dist: float,
    candle_pips: float,
    atr: float,
    close_dist: float,
    momentum: float,
    retest: dict[str, Any],
    london_open_bars: int,
    vol_pct: float,
    adr_used: float,
    adr_remaining: float,
    adr_expansion: float,
    session_minutes: int,
    h4_label: str,
    d1_label: str,
    aligned: bool,
    h4_dist: float,
    d1_dist: float,
    dist_daily_high: float,
    dist_daily_low: float,
) -> LboBayesFeatures:
    symmetry = 1.0 - abs(tokyo.tokyo_high_time_ratio - tokyo.tokyo_low_time_ratio)
    return LboBayesFeatures(
        tokyo_range_pips=tokyo.range_pips,
        tokyo_range_atr_ratio=tokyo.range_atr_ratio,
        tokyo_range_percentile=percentile,
        tokyo_formation_bars=tokyo.formation_bars,
        tokyo_high_low_symmetry=symmetry,
        tokyo_range_compression_score=compression,
        breakout_distance_from_tokyo_mid=mid_dist,
        breakout_candle_atr_ratio=candle_pips * PIP_SIZE / atr if atr > 0.0 else 0.0,
        breakout_close_distance_pips=close_dist,
        breakout_momentum=momentum,
        breakout_volume_ratio=float(retest.get("tick_volume_ratio", 1.0)),
        london_open_bars_elapsed=london_open_bars,
        tick_volume_ratio=float(retest.get("tick_volume_ratio", 1.0)),
        breakout_trade_density=float(retest.get("breakout_trade_density", 0.0)),
        retest_depth_ratio=float(retest.get("retest_depth_ratio", 0.0)),
        retest_duration_bars=int(retest["retest_duration_bars"]),
        retest_rejection_strength=float(retest["retest_rejection_strength"]),
        retest_pa_type=str(retest["retest_pa_type"]),
        retest_count=int(retest.get("retest_count", 1)),
        volatility_percentile=vol_pct,
        adr_used_at_entry=adr_used,
        adr_remaining=adr_remaining,
        adr_expansion_today=adr_expansion,
        session_minutes_elapsed=session_minutes,
        htf_h4_direction=h4_label,
        htf_d1_direction=d1_label,
        htf_aligned=aligned,
        h4_distance_from_ma=h4_dist,
        d1_distance_from_ma=d1_dist,
        distance_to_daily_high=dist_daily_high,
        distance_to_daily_low=dist_daily_low,
    )


def detect_lbo_setups(
    m15_df: pd.DataFrame,
    h1_df: pd.DataFrame,
    pair: str,
    *,
    h4_df: pd.DataFrame | None = None,
) -> list[LboSetup]:
    from strategies.bt_ohlcv import BtOhlcvFrame

    if isinstance(m15_df, BtOhlcvFrame):
        m15_df = m15_df.to_pandas()
    if isinstance(h1_df, BtOhlcvFrame):
        h1_df = h1_df.to_pandas()
    if h4_df is not None and isinstance(h4_df, BtOhlcvFrame):
        h4_df = h4_df.to_pandas()

    if not is_lbo_enabled() or pair not in ALLOWED_PAIRS:
        return []
    if m15_df is None or h1_df is None or len(m15_df) < 50 or len(h1_df) < 50:
        return []

    pip = pip_size_for_pair(pair)
    m15_open, m15_high, m15_low, m15_close, m15_vol = ohlcv_to_arrays(m15_df)
    h1_open, h1_high, h1_low, h1_close, _h1_vol = ohlcv_to_arrays(h1_df)
    m15_ts_ns = np.asarray(pd.to_datetime(m15_df["datetime"]).astype(np.int64))
    h1_ts_ns = np.asarray(pd.to_datetime(h1_df["datetime"]).astype(np.int64))
    m15_jst_days, m15_jst_hours, m15_jst_mins = jst_fields_from_ns(m15_ts_ns)
    h1_jst_days, h1_jst_hours, _h1_mins = jst_fields_from_ns(h1_ts_ns)
    h1_atr = precompute_atr_series(h1_high, h1_low, h1_close, ATR_PERIOD)
    m15_atr = precompute_atr_series(m15_high, m15_low, m15_close, ATR_PERIOD)
    h1_days = day_index_from_timestamps(np.asarray(h1_df["datetime"]))

    unique_days = sorted(set(int(d) for d in m15_jst_days))
    setups: list[LboSetup] = []
    used_days: set[int] = set()
    reverse_used: set[int] = set()

    for target_day in unique_days:
        if target_day in used_days:
            continue
        session_end_ns = int(target_day) * NS_PER_DAY + TOKYO_END_JST * 3600 * NS_PER_SEC - JST_OFFSET_NS
        session_end_ts = pd.Timestamp(session_end_ns)
        tokyo = detect_tokyo_range(h1_df, session_end_ts, pip_size=pip)
        if tokyo is None or not tokyo.is_valid:
            continue

        day_indices = np.where(m15_jst_days == target_day)[0]
        if len(day_indices) == 0:
            continue

        b_idx, dir_i, close_dist, candle_pips, momentum, breakout_close = detect_breakout_bar_core(
            m15_close,
            m15_high,
            m15_low,
            m15_jst_days,
            m15_jst_hours,
            target_day,
            tokyo.high,
            tokyo.low,
            pip,
            MIN_BREAKOUT_PIPS,
        )
        if b_idx < 0 or dir_i == 0:
            continue
        direction: Direction = "BUY" if dir_i == 1 else "SELL"
        breakout_idx = int(b_idx)

        retest = detect_retest(m15_df, tokyo, breakout_idx, direction, pip_size=pip)
        if retest is None and target_day not in reverse_used:
            search_from = breakout_idx + 1
            opp_idx, opp_i, _, _, _, _ = detect_breakout_bar_core(
                m15_close[search_from:],
                m15_high[search_from:],
                m15_low[search_from:],
                m15_jst_days[search_from:],
                m15_jst_hours[search_from:],
                target_day,
                tokyo.high,
                tokyo.low,
                pip,
                MIN_BREAKOUT_PIPS,
            )
            if opp_idx >= 0 and opp_i != 0 and opp_i != dir_i:
                reverse_used.add(target_day)
                breakout_idx = search_from + int(opp_idx)
                direction = "BUY" if opp_i == 1 else "SELL"
                retest = detect_retest(m15_df, tokyo, breakout_idx, direction, pip_size=pip)

        if retest is None or retest.get("is_absorbed"):
            continue

        entry_idx = int(retest["retest_bar_index"])
        entry = float(m15_close[entry_idx])
        entry_ts = pd.Timestamp(m15_df["datetime"].iloc[entry_idx])
        h1_clip = clip_as_of(h1_df, entry_ts)
        h1_end = len(h1_clip) - 1
        atr = float(h1_atr[min(h1_end, len(h1_atr) - 1)])

        adr_used, adr_remaining, _adr_pct, adr_expansion = compute_adr_metrics(
            h1_high,
            h1_low,
            h1_close,
            h1_days,
            h1_end,
            h1_atr,
        )
        if adr_used >= ADR_REJECT_THRESHOLD:
            continue

        day_high = float(h1_high[h1_end])
        day_low = float(h1_low[h1_end])
        for j in range(h1_end, -1, -1):
            if h1_days[j] != h1_days[h1_end]:
                break
            day_high = max(day_high, float(h1_high[j]))
            day_low = min(day_low, float(h1_low[j]))

        sl_tp = _compute_sl_tp(
            direction,
            entry,
            tokyo.high,
            tokyo.low,
            atr,
            pip,
            daily_high=day_high,
            daily_low=day_low,
        )
        if sl_tp is None:
            continue
        stop, take, rr = sl_tp

        h4_clip = clip_as_of(h4_df, entry_ts) if h4_df is not None else None
        h4_result = analyze_htf_trend(h1_clip, entry_ts, htf_df=h4_clip)
        d1_df = resample_to_htf(h1_clip, "1D")
        d1_result = analyze_htf_trend(h1_clip, entry_ts, htf_df=d1_df, resample_rule="1D", bar_hours=24)

        past_ranges = past_tokyo_range_pips(h1_df, target_day, pip_size=pip)
        percentile = float(np.sum(past_ranges <= tokyo.range_pips) / len(past_ranges)) if len(past_ranges) else 0.5
        avg_past = float(np.mean(past_ranges)) if len(past_ranges) else tokyo.range_pips
        compression = tokyo.range_pips / avg_past if avg_past > 0.0 else 1.0
        tokyo_mid = (tokyo.high + tokyo.low) / 2.0
        mid_dist = abs(breakout_close - tokyo_mid) / pip

        h4_label = _trend_to_htf_label(h4_result.direction)
        d1_label = _trend_to_htf_label(d1_result.direction)
        aligned = _is_htf_aligned(direction, h4_label)

        hour, minute = jst_hour_minute(entry_ts)
        vol_pct = (
            float(np.sum(h1_atr[max(0, h1_end - 99) : h1_end + 1] <= atr) / min(h1_end + 1, 100))
            if atr > 0.0
            else 0.5
        )
        london_open_bars = london_bars_before_breakout(m15_jst_hours, m15_jst_days, target_day, breakout_idx)
        features = _build_features_for_setup(
            tokyo=tokyo,
            percentile=percentile,
            compression=compression,
            mid_dist=mid_dist,
            candle_pips=candle_pips,
            atr=atr,
            close_dist=close_dist,
            momentum=momentum,
            retest=retest,
            london_open_bars=london_open_bars,
            vol_pct=vol_pct,
            adr_used=adr_used,
            adr_remaining=adr_remaining,
            adr_expansion=adr_expansion,
            session_minutes=_session_minutes_elapsed(hour, minute),
            h4_label=h4_label,
            d1_label=d1_label,
            aligned=aligned,
            h4_dist=_distance_from_ma(entry, h4_result.ma200, pip),
            d1_dist=_distance_from_ma(entry, d1_result.ma200, pip),
            dist_daily_high=max(day_high - entry, 0.0) / pip,
            dist_daily_low=max(entry - day_low, 0.0) / pip,
        )

        setups.append(
            LboSetup(
                timestamp=entry_ts,
                pair=pair,
                direction=direction,
                entry_price=entry,
                stop_loss=stop,
                take_profit=take,
                tokyo_high=tokyo.high,
                tokyo_low=tokyo.low,
                tokyo_range_pips=tokyo.range_pips,
                tokyo_range_atr_ratio=tokyo.range_atr_ratio,
                tokyo_range_percentile=percentile,
                tokyo_range_compression_score=compression,
                breakout_distance_from_tokyo_mid=mid_dist,
                breakout_candle_size_pips=candle_pips,
                breakout_candle_atr_ratio=candle_pips * pip / atr if atr > 0.0 else 0.0,
                breakout_close_distance_pips=close_dist,
                breakout_momentum=momentum,
                tick_volume_ratio=float(retest.get("tick_volume_ratio", 1.0)),
                breakout_trade_density=float(retest.get("breakout_trade_density", 0.0)),
                retest_depth_pips=float(retest["retest_depth_pips"]),
                retest_depth_ratio=float(retest.get("retest_depth_ratio", 0.0)),
                retest_duration_bars=int(retest["retest_duration_bars"]),
                retest_rejection_strength=float(retest["retest_rejection_strength"]),
                retest_pa_type=str(retest["retest_pa_type"]),
                retest_count=int(retest.get("retest_count", 1)),
                london_open_bars_elapsed=london_open_bars,
                london_range_before_break=london_range_before_break_pips(
                    m15_high,
                    m15_low,
                    m15_jst_days,
                    m15_jst_hours,
                    target_day,
                    breakout_idx,
                    pip,
                ),
                atr=atr,
                volatility_percentile=vol_pct,
                adr_used_at_entry=adr_used,
                adr_remaining=adr_remaining,
                session_minutes_elapsed=_session_minutes_elapsed(hour, minute),
                adr_expansion_today=adr_expansion,
                htf_h4_direction=h4_label,
                htf_aligned=aligned,
                htf_d1_direction=d1_label,
                h4_distance_from_ma=_distance_from_ma(entry, h4_result.ma200, pip),
                d1_distance_from_ma=_distance_from_ma(entry, d1_result.ma200, pip),
                distance_to_daily_high=max(day_high - entry, 0.0) / pip,
                distance_to_daily_low=max(entry - day_low, 0.0) / pip,
                smt_intensity=0.0,
                smt_leader="NONE",
                both_broke=False,
                leader_lag_bars=0,
                bar_index_m15=entry_idx,
                bar_index_h1=h1_end,
                jst_session_date=tokyo.date,
                breakout_bar_index=breakout_idx,
                bayes_features=features,
                rr_ratio=rr,
            )
        )
        used_days.add(target_day)

    return setups


def build_lbo_feature_log_row(
    *,
    trade_id: str,
    setup: LboSetup,
    trade_result: str,
    profit_r: float,
    decision_source: str = "PURE_BT",
    executed: bool = True,
) -> dict[str, Any]:
    row = setup.bayes_features.as_dict()
    row.update(
        {
            "trade_id": trade_id,
            "timestamp": setup.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "pair": setup.pair,
            "direction": setup.direction,
            "decision_source": decision_source,
            "executed": executed,
            "trade_result": trade_result,
            "profit_r": profit_r,
            "rr_ratio": setup.rr_ratio,
        }
    )
    return row


class LboStrategy(BaseStrategy):
    """London Break Out — Tokyo range / London retest (feature collection)."""

    @property
    def setup_type(self) -> str:
        return SETUP_TYPE

    def detect_setups(
        self,
        df: pd.DataFrame,
        pair_name: str,
        h1_df: pd.DataFrame | None = None,
        h4_df: pd.DataFrame | None = None,
    ) -> list[LboSetup]:
        if h1_df is None:
            return []
        return detect_lbo_setups(df, h1_df, pair_name, h4_df=h4_df)

    def analyze_setup(
        self,
        setup: Any,
        gbp_setup: Any | None,
        eur_setup: Any | None,
        h1_gbp: pd.DataFrame,
        h1_eur: pd.DataFrame,
    ) -> StrategyResult:
        if not isinstance(setup, LboSetup):
            return StrategyResult(
                is_setup=False,
                setup_type=self.setup_type,
                direction="FLAT",
                strategy_action="REJECT",
            )

        raw = setup.bayes_features.as_dict()
        raw["signal_type"] = SETUP_TYPE

        if gbp_setup is not None and eur_setup is not None:
            smt = calc_smt_features(gbp_setup, eur_setup)
            raw["smt_intensity"] = smt.intensity
            raw["smt_diff"] = smt.diff
            raw["smt_leader"] = smt.leader
            if (
                isinstance(gbp_setup, LboSetup)
                and isinstance(eur_setup, LboSetup)
                and gbp_setup.jst_session_date == eur_setup.jst_session_date
                and gbp_setup.direction == eur_setup.direction
            ):
                raw["both_broke"] = True
                raw["leader_lag_bars"] = gbp_setup.breakout_bar_index - eur_setup.breakout_bar_index
            else:
                raw["both_broke"] = False
                raw["leader_lag_bars"] = 0
        else:
            raw["smt_intensity"] = setup.smt_intensity
            raw["smt_diff"] = 0.0
            raw["smt_leader"] = setup.smt_leader

        htf_ref = h1_gbp if setup.pair == "GBPUSD" else h1_eur
        htf = analyze_htf_trend(htf_ref, setup.timestamp)
        counter = is_counter_trend(setup.direction, htf.direction)
        raw["htf_counter_trend"] = counter

        action = "ALLOW" if is_lbo_pure_data_mode() else "ALLOW"
        return StrategyResult(
            is_setup=True,
            setup_type=self.setup_type,
            direction=setup.direction,
            entry_price=setup.entry_price,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            candidate_score=0.0,
            raw_features=raw,
            strategy_action=action,
        )


__all__ = [
    "LboSetup",
    "LboStrategy",
    "LboBayesFeatures",
    "SETUP_TYPE",
    "STRATEGY_ABBREV",
    "STRATEGY_FULL_NAME",
    "ALLOWED_PAIRS",
    "LBO_FEATURE_COLUMNS",
    "LBO_L6_EXTRA_COLUMNS",
    "detect_lbo_setups",
    "build_lbo_feature_log_row",
    "is_lbo_pure_data_mode",
    "is_lbo_enabled",
]
