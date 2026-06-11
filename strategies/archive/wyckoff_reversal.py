"""
Wyckoff Reversal (WR) — Spring (BUY) + Upthrust (SELL) 統合戦略。

ARCHIVED: 新戦略 Liquidity Grab Reversal (LGR) 構築に向けての発展的廃止。参照・再分析用。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from strategies.base import StrategyResult
from strategies.base_strategy import BaseStrategy
from strategies.archive.cspa import resolve_cspa_session_type
from strategies.htf_trend_analyzer import analyze_htf_trend, clip_as_of, resample_to_htf
from strategies.market_utils import compute_atr, pip_size_for_pair

logger = logging.getLogger("wyckoff_reversal")

SETUP_TYPE = "WYCKOFF_REVERSAL"
SETUP_TYPE_LEGACY = "WYCKOFF_SPRING"
STRATEGY_ABBREV = "WR"
STRATEGY_ID = "wyckoff_reversal"
STRATEGY_FULL_NAME = "Wyckoff Reversal"
WYCKOFF_PAIR_PRIMARY = "GBPUSD"
WYCKOFF_PAIR_SECONDARY = "EURUSD"
ALLOWED_PAIRS = frozenset({WYCKOFF_PAIR_PRIMARY, WYCKOFF_PAIR_SECONDARY})

LOOKBACK_BARS = int(os.getenv("WYCKOFF_LOOKBACK_BARS", "300"))
ATR_PERIOD = int(os.getenv("WYCKOFF_ATR_PERIOD", "14"))
SC_BODY_ATR_MULT = float(os.getenv("WYCKOFF_SC_BODY_ATR_MULT", "1.8"))
SC_VOLUME_ZSCORE_MIN = float(os.getenv("WYCKOFF_SC_VOLUME_ZSCORE_MIN", "2.0"))
SC_DOWNTREND_LOOKBACK = int(os.getenv("WYCKOFF_SC_DOWNTREND_LOOKBACK", "40"))
SC_VOLUME_ZSCORE_LOOKBACK = int(os.getenv("WYCKOFF_SC_VOLUME_ZSCORE_LOOKBACK", "20"))
AR_MIN_RALLY_ATR = float(os.getenv("WYCKOFF_AR_MIN_RALLY_ATR", "3.0"))
SUPPORT_BAND_ATR = float(os.getenv("WYCKOFF_SUPPORT_BAND_ATR", "0.3"))
SPRING_MIN_DEPTH_ATR = float(os.getenv("WYCKOFF_SPRING_MIN_DEPTH_ATR", "0.1"))
SPRING_MAX_DEPTH_ATR = float(os.getenv("WYCKOFF_SPRING_MAX_DEPTH_ATR", "1.5"))
UPTHRUST_MIN_HEIGHT_ATR = float(os.getenv("WYCKOFF_UPTHRUST_MIN_HEIGHT_ATR", "0.1"))
UPTHRUST_MAX_HEIGHT_ATR = float(os.getenv("WYCKOFF_UPTHRUST_MAX_HEIGHT_ATR", "1.5"))
UPTHRUST_WICK_RATIO_MIN = float(os.getenv("WYCKOFF_UPTHRUST_WICK_RATIO_MIN", "0.35"))
SPRING_VOLUME_RATIO_MIN = float(os.getenv("WYCKOFF_SPRING_VOLUME_RATIO_MIN", "1.0"))
UPTHRUST_VOLUME_RATIO_MIN = float(os.getenv("WYCKOFF_UPTHRUST_VOLUME_RATIO_MIN", "1.0"))
SL_BUFFER_ATR = float(os.getenv("WYCKOFF_SL_BUFFER_ATR", "0.3"))
MIN_RR = float(os.getenv("WYCKOFF_MIN_RR", "1.5"))
WYCKOFF_MONITOR_BAR_MINUTES = int(os.getenv("WYCKOFF_MONITOR_BAR_MINUTES", "60"))
WYCKOFF_SPRING_BAR_MINUTES = int(os.getenv("WYCKOFF_SPRING_BAR_MINUTES", "15"))
WYCKOFF_EXEC_BAR_MINUTES = WYCKOFF_SPRING_BAR_MINUTES
WYCKOFF_MAX_HOLDING_HOURS = int(os.getenv("WYCKOFF_MAX_HOLDING_HOURS", "48"))
MAX_HOLDING_BARS = max(4, (WYCKOFF_MAX_HOLDING_HOURS * 60) // WYCKOFF_EXEC_BAR_MINUTES)
# 1 日あたり最大セットアップ数（ペア別）。環境変数 WYCKOFF_MAX_SETUPS_PER_DAY で上書き可。
MAX_SETUPS_PER_DAY = int(os.getenv("WYCKOFF_MAX_SETUPS_PER_DAY", "1"))
WYCKOFF_PRODUCTION_MAX_SETUPS_PER_DAY = 1
MIN_BARS_AFTER_SC = int(os.getenv("WYCKOFF_MIN_BARS_AFTER_SC", "20"))
RECOVERY_QUALITY_BARS = int(os.getenv("WYCKOFF_RECOVERY_QUALITY_BARS", "8"))
VOLUME_LOOKBACK = int(os.getenv("WYCKOFF_VOLUME_LOOKBACK", "20"))
SESSION_VOLUME_LOOKBACK_DAYS = int(os.getenv("WYCKOFF_SESSION_VOLUME_DAYS", "5"))
VOLATILITY_LOOKBACK = int(os.getenv("WYCKOFF_VOLATILITY_LOOKBACK", "120"))
ADR_LOOKBACK_DAYS = int(os.getenv("WYCKOFF_ADR_LOOKBACK_DAYS", "14"))
WEEKLY_BARS = int(os.getenv("WYCKOFF_WEEKLY_BARS", "120"))
MONTHLY_BARS = int(os.getenv("WYCKOFF_MONTHLY_BARS", "480"))
WYCKOFF_BT_SPREAD_PIPS = float(os.getenv("WYCKOFF_BT_SPREAD_PIPS", "1.2"))


def _env_flag(name: str) -> bool | None:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return None


WS_PYRAMID_ENABLED = _env_flag("WS_PYRAMID_ENABLED") if _env_flag("WS_PYRAMID_ENABLED") is not None else True
WS_PYRAMID_MAX_LAYERS = int(os.getenv("WS_PYRAMID_MAX_LAYERS", "3"))
WS_PYRAMID_TRIGGER_R = float(os.getenv("WS_PYRAMID_TRIGGER_R", "1.0"))
WS_PYRAMID_TIME_LIMIT_BARS = int(os.getenv("WS_PYRAMID_TIME_LIMIT_BARS", "24"))


def is_wyckoff_pure_bt_mode() -> bool:
    """特徴量収集用: L0-L7 防御・カルマン・ピラミッドを無効化するピュア BT。"""
    return os.getenv("WYCKOFF_PURE_BT", "0").strip().lower() in ("1", "true", "yes", "on")


def _neutral_kalman_features() -> dict[str, float | int]:
    return {
        "kalman_velocity_at_entry": 0.0,
        "velocity_positive_bars": 0,
        "kalman_noise_ratio": 1.0,
    }


REJECT_BY_RR = "REJECT_BY_RR"
PhaseLabel = Literal["ACCUMULATION", "SPRING"]
WyckoffMacroPhase = Literal["ACCUMULATION", "MARKUP", "DISTRIBUTION", "MARKDOWN", "UNKNOWN"]
ReversalPattern = Literal["SPRING", "UPTHRUST"]
WrMode = ReversalPattern
SessionType = Literal["ASIA", "LONDON", "NY", "OFF_HOURS"]
AtrRegime = Literal["LOW", "NORMAL", "HIGH"]
OutcomeLabel = Literal["WIN", "LOSS"]
TradeDirection = Literal["BUY", "SELL"]

WYCKOFF_FEATURE_COLUMNS: tuple[str, ...] = (
    # --- 既存列（後方互換） ---
    "phase_identified",
    "support_test_count",
    "days_in_accumulation",
    "phase_b_duration",
    "range_width_atr",
    "spring_attempt_number",
    "spring_depth_atr",
    "spring_velocity",
    "spring_duration_bars",
    "support_penetration_percent",
    "spring_volume_ratio",
    "spring_recovery_time",
    "resistance_distance",
    "spring_wick_ratio",
    "range_compression",
    "spring_recovery_atr",
    "recovery_duration_bars",
    "recovery_velocity",
    "recovery_velocity_atr",
    "recovery_close_ratio",
    "recovery_acceleration",
    "kalman_velocity_at_entry",
    "velocity_positive_bars",
    "kalman_noise_ratio",
    "consecutive_higher_closes",
    "positive_close_ratio",
    "directional_efficiency",
    "noise_ratio",
    "session_type",
    "volatility_percentile",
    "atr_regime",
    "adr_remaining",
    "distance_weekly_low",
    "distance_monthly_low",
    "htf_trend_d1",
    "phase_b_ut_occurred",
    "outcome_label",
    "result_r",
    "mfe",
    "mae",
    # --- Wyckoff Reversal 拡張（ログ専用含む） ---
    "reversal_pattern",
    "wr_mode",
    "wyckoff_phase",
    "range_width",
    "range_age_bars",
    "upthrust_height",
    "recovery_speed",
    "volume_ratio",
    "volume_percentile",
    "volume_expansion",
    "distance_from_range_low",
    "distance_from_range_high",
    "position_in_range",
    "atr_ratio",
    "smt_intensity",
    "smt_diff",
    "smt_leader",
    "minutes_from_session_open",
    "adr_used",
    "distance_daily_high",
    "distance_daily_low",
    "distance_session_high",
    "distance_session_low",
    "spread",
    "spread_percentile",
    "tail_move_after_entry",
    "max_favorable_excursion",
    "max_adverse_excursion",
    "time_to_tp",
    "time_to_sl",
    "trend_age_bars",
    "impulse_quality",
    "impulse_atr_ratio",
    "stagnation_duration",
    "stagnation_width",
    "breakout_velocity",
    "breakout_momentum_ratio",
)


@dataclass(frozen=True)
class AccumulationRange:
    ps_price: float
    sc_price: float
    ar_price: float
    st_price: float
    support_level: float
    resistance_level: float
    range_width_atr: float
    test_count: int
    phase_b_ut_occurred: bool
    days_in_accumulation: int
    phase_b_duration: int
    is_valid: bool
    sc_bar_index: int
    ar_bar_index: int
    sc_timestamp: pd.Timestamp
    ar_timestamp: pd.Timestamp


@dataclass(frozen=True)
class WyckoffFeatures:
    phase_identified: PhaseLabel
    support_test_count: int
    days_in_accumulation: int
    phase_b_duration: int
    range_width_atr: float
    spring_attempt_number: int
    spring_depth_atr: float
    spring_velocity: float
    spring_duration_bars: int
    support_penetration_percent: float
    spring_volume_ratio: float
    spring_recovery_time: int
    resistance_distance: float
    spring_wick_ratio: float
    range_compression: float
    spring_recovery_atr: float
    recovery_duration_bars: int
    recovery_velocity: float
    recovery_velocity_atr: float
    recovery_close_ratio: float
    recovery_acceleration: float
    kalman_velocity_at_entry: float
    velocity_positive_bars: int
    kalman_noise_ratio: float
    consecutive_higher_closes: int
    positive_close_ratio: float
    directional_efficiency: float
    noise_ratio: float
    session_type: SessionType
    volatility_percentile: float
    atr_regime: AtrRegime
    adr_remaining: float
    distance_weekly_low: float
    distance_monthly_low: float
    htf_trend_d1: str
    phase_b_ut_occurred: bool
    outcome_label: OutcomeLabel | str = ""
    result_r: float = 0.0
    mfe: float = 0.0
    mae: float = 0.0
    # --- 拡張フィールド ---
    reversal_pattern: ReversalPattern = "SPRING"
    wr_mode: WrMode = "SPRING"
    wyckoff_phase: WyckoffMacroPhase = "UNKNOWN"
    range_width: float = 0.0
    range_age_bars: int = 0
    upthrust_height: float = 0.0
    recovery_speed: float = 0.0
    volume_ratio: float = 1.0
    volume_percentile: float = 50.0
    volume_expansion: float = 0.0
    distance_from_range_low: float = 0.0
    distance_from_range_high: float = 0.0
    position_in_range: float = 0.5
    atr_ratio: float = 0.0
    smt_intensity: float = 0.0
    smt_diff: float = 0.0
    smt_leader: str = "NONE"
    minutes_from_session_open: int = 0
    adr_used: float = 0.0
    distance_daily_high: float = 0.0
    distance_daily_low: float = 0.0
    distance_session_high: float = 0.0
    distance_session_low: float = 0.0
    spread: float = 0.0
    spread_percentile: float = 0.0
    tail_move_after_entry: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    time_to_tp: int = -1
    time_to_sl: int = -1
    trend_age_bars: int = 0
    impulse_quality: float = 0.0
    impulse_atr_ratio: float = 0.0
    stagnation_duration: int = 0
    stagnation_width: float = 0.0
    breakout_velocity: float = 0.0
    breakout_momentum_ratio: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        base = {
            "phase_identified": self.phase_identified,
            "support_test_count": self.support_test_count,
            "days_in_accumulation": self.days_in_accumulation,
            "phase_b_duration": self.phase_b_duration,
            "range_width_atr": round(self.range_width_atr, 4),
            "spring_attempt_number": self.spring_attempt_number,
            "spring_depth_atr": round(self.spring_depth_atr, 4),
            "spring_velocity": round(self.spring_velocity, 4),
            "spring_duration_bars": self.spring_duration_bars,
            "support_penetration_percent": round(self.support_penetration_percent, 4),
            "spring_volume_ratio": round(self.spring_volume_ratio, 4),
            "spring_recovery_time": self.spring_recovery_time,
            "resistance_distance": round(self.resistance_distance, 4),
            "spring_wick_ratio": round(self.spring_wick_ratio, 4),
            "range_compression": round(self.range_compression, 4),
            "spring_recovery_atr": round(self.spring_recovery_atr, 4),
            "recovery_duration_bars": self.recovery_duration_bars,
            "recovery_velocity": round(self.recovery_velocity, 4),
            "recovery_velocity_atr": round(self.recovery_velocity_atr, 4),
            "recovery_close_ratio": round(self.recovery_close_ratio, 4),
            "recovery_acceleration": round(self.recovery_acceleration, 4),
            "kalman_velocity_at_entry": round(self.kalman_velocity_at_entry, 8),
            "velocity_positive_bars": self.velocity_positive_bars,
            "kalman_noise_ratio": round(self.kalman_noise_ratio, 6),
            "consecutive_higher_closes": self.consecutive_higher_closes,
            "positive_close_ratio": round(self.positive_close_ratio, 4),
            "directional_efficiency": round(self.directional_efficiency, 4),
            "noise_ratio": round(self.noise_ratio, 4),
            "session_type": self.session_type,
            "volatility_percentile": round(self.volatility_percentile, 4),
            "atr_regime": self.atr_regime,
            "adr_remaining": round(self.adr_remaining, 4),
            "distance_weekly_low": round(self.distance_weekly_low, 4),
            "distance_monthly_low": round(self.distance_monthly_low, 4),
            "htf_trend_d1": self.htf_trend_d1,
            "phase_b_ut_occurred": self.phase_b_ut_occurred,
            "outcome_label": self.outcome_label,
            "result_r": round(self.result_r, 4),
            "mfe": round(self.mfe, 4),
            "mae": round(self.mae, 4),
            "reversal_pattern": self.reversal_pattern,
            "wr_mode": self.wr_mode,
            "wyckoff_phase": self.wyckoff_phase,
            "range_width": round(self.range_width, 6),
            "range_age_bars": self.range_age_bars,
            "upthrust_height": round(self.upthrust_height, 4),
            "recovery_speed": round(self.recovery_speed, 4),
            "volume_ratio": round(self.volume_ratio, 4),
            "volume_percentile": round(self.volume_percentile, 4),
            "volume_expansion": round(self.volume_expansion, 4),
            "distance_from_range_low": round(self.distance_from_range_low, 4),
            "distance_from_range_high": round(self.distance_from_range_high, 4),
            "position_in_range": round(self.position_in_range, 4),
            "atr_ratio": round(self.atr_ratio, 4),
            "smt_intensity": round(self.smt_intensity, 4),
            "smt_diff": round(self.smt_diff, 4),
            "smt_leader": self.smt_leader,
            "minutes_from_session_open": self.minutes_from_session_open,
            "adr_used": round(self.adr_used, 4),
            "distance_daily_high": round(self.distance_daily_high, 4),
            "distance_daily_low": round(self.distance_daily_low, 4),
            "distance_session_high": round(self.distance_session_high, 4),
            "distance_session_low": round(self.distance_session_low, 4),
            "spread": round(self.spread, 6),
            "spread_percentile": round(self.spread_percentile, 4),
            "tail_move_after_entry": round(self.tail_move_after_entry, 4),
            "max_favorable_excursion": round(self.max_favorable_excursion, 4),
            "max_adverse_excursion": round(self.max_adverse_excursion, 4),
            "time_to_tp": self.time_to_tp,
            "time_to_sl": self.time_to_sl,
            "trend_age_bars": self.trend_age_bars,
            "impulse_quality": round(self.impulse_quality, 4),
            "impulse_atr_ratio": round(self.impulse_atr_ratio, 4),
            "stagnation_duration": self.stagnation_duration,
            "stagnation_width": round(self.stagnation_width, 4),
            "breakout_velocity": round(self.breakout_velocity, 4),
            "breakout_momentum_ratio": round(self.breakout_momentum_ratio, 4),
        }
        return base

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False)


@dataclass(frozen=True)
class WyckoffReversalSimResult:
    result: str
    profit_r: float
    profit_pips: float
    holding_minutes: int
    pyramid_layers: int
    pyramid_entry_prices: list[float]
    pyramid_lot_sizes: list[float]
    final_sl_at_close: float
    peak_unrealized_r: float
    kalman_velocity_at_entry: float
    kalman_decel_exit_triggered: bool
    time_limit_exit_triggered: bool
    pyramid_rejected_reason: str
    mfe: float = 0.0
    mae: float = 0.0
    time_to_tp: int = -1
    time_to_sl: int = -1
    tail_move_after_entry: float = 0.0


WyckoffSpringSimResult = WyckoffReversalSimResult


@dataclass(frozen=True)
class ReversalSetup:
    timestamp: pd.Timestamp
    pair: str
    accumulation: AccumulationRange
    spring_depth_atr: float
    spring_velocity: float
    spring_duration_bars: int
    support_penetration_percent: float
    spring_attempt_number: int
    volume_on_spring: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    spring_bar_index: int
    recovery_bar_index: int
    wyckoff_features: WyckoffFeatures
    reason_codes: tuple[str, ...] = ()
    direction: TradeDirection = "BUY"
    reversal_pattern: ReversalPattern = "SPRING"
    upthrust_height_atr: float = 0.0
    candidate_score: float = 70.0
    ws_sim: WyckoffReversalSimResult | None = None


SpringSetup = ReversalSetup


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    work = df.sort_values("datetime").reset_index(drop=True).copy()
    work["datetime"] = pd.to_datetime(work["datetime"])
    if "volume" not in work.columns:
        work["volume"] = 0.0
    return work


def _body_size(row: pd.Series) -> float:
    return abs(float(row["close"]) - float(row["open"]))


def _is_bearish(row: pd.Series) -> bool:
    return float(row["close"]) < float(row["open"])


def _bar_hour(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).hour)


def _session_volume_samples(
    work: pd.DataFrame,
    bar_index: int,
    lookback_days: int = SESSION_VOLUME_LOOKBACK_DAYS,
) -> list[float]:
    """同一時刻帯（時）の過去数日ボリューム — ティック出来高の時間帯正規化。"""
    if bar_index < 0 or bar_index >= len(work):
        return []
    ts = pd.Timestamp(work.iloc[bar_index]["datetime"])
    target_hour = _bar_hour(ts)
    dt_series = pd.to_datetime(work["datetime"])
    day_norm = dt_series.dt.normalize()
    current_day = ts.normalize()
    samples: list[float] = []
    for offset in range(1, lookback_days + 1):
        prior_day = current_day - pd.Timedelta(days=offset)
        mask = (day_norm == prior_day) & (dt_series.dt.hour == target_hour)
        if not mask.any():
            continue
        prior_idx = np.where(mask.to_numpy())[0]
        prior_idx = prior_idx[prior_idx < bar_index]
        if len(prior_idx) == 0:
            continue
        samples.append(float(work.iloc[int(prior_idx[-1])]["volume"]))
    return samples


def _rolling_volume_zscore(work: pd.DataFrame, bar_index: int, lookback: int = SC_VOLUME_ZSCORE_LOOKBACK) -> float:
    start = max(0, bar_index - lookback)
    window = work.iloc[start:bar_index]["volume"].astype(float)
    if len(window) < 3:
        return 0.0
    val = float(work.iloc[bar_index]["volume"])
    mean = float(window.mean())
    std = float(window.std(ddof=0))
    if std <= 0:
        return 0.0
    return (val - mean) / std


def _session_volume_zscore(
    work: pd.DataFrame,
    bar_index: int,
    lookback_days: int = SESSION_VOLUME_LOOKBACK_DAYS,
) -> float:
    """SC 判定用 — 同一時間帯の過去数日平均との Z スコア（不足時は rolling にフォールバック）。"""
    samples = _session_volume_samples(work, bar_index, lookback_days)
    val = float(work.iloc[bar_index]["volume"])
    if len(samples) >= 2:
        mean = float(np.mean(samples))
        std = float(np.std(samples, ddof=0))
        if std > 0:
            return (val - mean) / std
        return 2.0 if val > mean else 0.0
    return _rolling_volume_zscore(work, bar_index)


def _in_downtrend_context(work: pd.DataFrame, sc_idx: int, lookback: int = SC_DOWNTREND_LOOKBACK) -> bool:
    """SC 直前がダウントレンド — 終値の下向き傾き + SC が局所安値圏。"""
    start = max(0, sc_idx - lookback)
    if sc_idx - start < 10:
        return False
    window = work.iloc[start:sc_idx]
    closes = window["close"].astype(float).values
    sc_low = float(work.iloc[sc_idx]["low"])
    window_low = float(window["low"].min())
    atr = _atr_at(work, sc_idx)
    near_low = sc_low <= window_low + max(0.15 * atr, 1e-9)
    x = np.arange(len(closes))
    slope = float(np.polyfit(x, closes, 1)[0])
    return slope < 0 and near_low


def _atr_at(work: pd.DataFrame, bar_index: int, atr_series: pd.Series | None = None) -> float:
    if bar_index < 0 or bar_index >= len(work):
        return 0.0
    if atr_series is not None:
        val = float(atr_series.iloc[bar_index])
        return val if np.isfinite(val) else 0.0
    clipped = work.iloc[: bar_index + 1]
    atr = compute_atr(clipped, ATR_PERIOD)
    if bar_index >= len(atr):
        return 0.0
    val = float(atr.iloc[bar_index])
    return val if np.isfinite(val) else 0.0


def _support_band(sc_price: float, atr: float) -> tuple[float, float]:
    half = SUPPORT_BAND_ATR * atr
    return sc_price - half, sc_price + half


def _touches_support_band(low: float, band_low: float, band_high: float) -> bool:
    return low <= band_high and low >= band_low - (band_high - band_low)


def _find_ps_price(work: pd.DataFrame, sc_idx: int, atr: float) -> float:
    """SC 前の参照安値 — Wyckoff 上 PS は SC 安値より上の水準として扱う。"""
    _ = atr
    start = max(0, sc_idx - 40)
    window = work.iloc[start:sc_idx]
    sc_low = float(work.iloc[sc_idx]["low"])
    if window.empty:
        return sc_low * 1.001
    ps_ref = float(window["low"].min())
    return max(ps_ref, sc_low * 1.001)


def _detect_phase_b_ut(
    work: pd.DataFrame,
    ar_idx: int,
    end_idx: int,
    resistance_level: float,
) -> bool:
    for i in range(ar_idx + 1, end_idx + 1):
        row = work.iloc[i]
        if float(row["high"]) > resistance_level and float(row["close"]) < resistance_level:
            return True
    return False


def _count_support_tests(
    work: pd.DataFrame,
    ar_idx: int,
    end_idx: int,
    band_low: float,
    band_high: float,
) -> tuple[int, float]:
    count = 0
    st_price = float(work.iloc[ar_idx]["high"])
    for i in range(ar_idx + 1, end_idx + 1):
        low = float(work.iloc[i]["low"])
        if _touches_support_band(low, band_low, band_high):
            count += 1
            st_price = min(st_price, low)
    return count, st_price


def _spread_percentile(spread_pips: float) -> float:
    ref = 3.0
    return round(min(max(spread_pips / ref, 0.0), 1.0), 4)


def _range_age_bars(accumulation: AccumulationRange, recovery_ts: pd.Timestamp) -> int:
    """AR 確定からセットアップまでの執行足換算バー数（近似）。"""
    delta_min = (pd.Timestamp(recovery_ts) - accumulation.ar_timestamp).total_seconds() / 60.0
    return max(0, int(delta_min // max(WYCKOFF_EXEC_BAR_MINUTES, 1)))


def _count_upthrust_attempts(
    h1_work: pd.DataFrame,
    accumulation: AccumulationRange,
    upthrust_ts: pd.Timestamp,
    resistance_level: float,
    atr_series: pd.Series,
) -> int:
    """H1 Phase B 上の Upthrust 試行回数（現 Upthrust より前。連続クラスターは 1 回）。"""
    if not accumulation.is_valid:
        return 1

    ar_ts = accumulation.ar_timestamp
    upthrust_ts = pd.Timestamp(upthrust_ts)
    attempts = 0
    in_cluster = False
    clipped = _prepare_df(h1_work)
    atr_series = compute_atr(clipped, ATR_PERIOD)

    for i in range(len(clipped)):
        ts = pd.Timestamp(clipped.iloc[i]["datetime"])
        if ts < ar_ts:
            continue
        if ts >= upthrust_ts:
            break

        atr = _atr_at(clipped, i, atr_series)
        if atr <= 0:
            continue

        row = clipped.iloc[i]
        high = float(row["high"])
        close = float(row["close"])
        if high <= resistance_level or close >= resistance_level:
            in_cluster = False
            continue

        height = high - resistance_level
        if height < UPTHRUST_MIN_HEIGHT_ATR * atr or height > UPTHRUST_MAX_HEIGHT_ATR * atr:
            in_cluster = False
            continue
        if _upthrust_wick_ratio(row) < UPTHRUST_WICK_RATIO_MIN:
            in_cluster = False
            continue

        if not in_cluster:
            attempts += 1
            in_cluster = True

    return max(1, attempts)


def _count_spring_attempts(
    h1_work: pd.DataFrame,
    accumulation: AccumulationRange,
    spring_ts: pd.Timestamp,
    support_level: float,
    atr_series: pd.Series,
) -> int:
    """H1 Phase B 上の Spring 試行回数（現 Spring より前。連続クラスターは 1 回）。

    M15 全バーを数えると Phase B ノイズで 30+ になりやすいため、
    蓄積監視足（H1）+ クラスタ dedupe で 5〜6 程度に収まる定義とする。
    """
    if not accumulation.is_valid:
        return 1

    ar_ts = accumulation.ar_timestamp
    spring_ts = pd.Timestamp(spring_ts)
    attempts = 0
    in_cluster = False
    clipped = _prepare_df(h1_work)
    atr_series = compute_atr(clipped, ATR_PERIOD)

    for i in range(len(clipped)):
        ts = pd.Timestamp(clipped.iloc[i]["datetime"])
        if ts < ar_ts:
            continue
        if ts >= spring_ts:
            break

        atr = _atr_at(clipped, i, atr_series)
        if atr <= 0:
            continue

        min_depth = SPRING_MIN_DEPTH_ATR * atr
        max_depth = SPRING_MAX_DEPTH_ATR * atr
        low = float(clipped.iloc[i]["low"])
        if low >= support_level:
            in_cluster = False
            continue

        penetration = support_level - low
        if penetration < min_depth or penetration > max_depth:
            in_cluster = False
            continue

        row = clipped.iloc[i]
        recovered = float(row["close"]) >= support_level
        if not recovered and i + 1 < len(clipped):
            recovered = float(clipped.iloc[i + 1]["close"]) >= support_level

        if recovered:
            if not in_cluster:
                attempts += 1
                in_cluster = True
        else:
            in_cluster = False

    return attempts + 1


def _try_build_accumulation(
    work: pd.DataFrame,
    sc_idx: int,
    atr_series: pd.Series,
    as_of_idx: int,
) -> AccumulationRange | None:
    if sc_idx >= as_of_idx - MIN_BARS_AFTER_SC or sc_idx < ATR_PERIOD:
        return None

    sc_row = work.iloc[sc_idx]
    sc_atr = _atr_at(work, sc_idx, atr_series)
    if sc_atr <= 0:
        return None

    body = _body_size(sc_row)
    if not _is_bearish(sc_row) or body < SC_BODY_ATR_MULT * sc_atr:
        return None

    vol_z = _session_volume_zscore(work, sc_idx)
    if vol_z < SC_VOLUME_ZSCORE_MIN:
        return None
    if not _in_downtrend_context(work, sc_idx):
        return None

    sc_price = float(sc_row["low"])
    sc_ts = pd.Timestamp(sc_row["datetime"])

    ar_price = sc_price
    ar_idx = sc_idx
    rally = 0.0
    for j in range(sc_idx + 1, as_of_idx + 1):
        high = float(work.iloc[j]["high"])
        ar_price = max(ar_price, high)
        rally = ar_price - sc_price
        if rally >= AR_MIN_RALLY_ATR * sc_atr:
            ar_idx = j
            break
    else:
        return None

    ar_ts = pd.Timestamp(work.iloc[ar_idx]["datetime"])
    band_low, band_high = _support_band(sc_price, sc_atr)
    support_level = sc_price
    resistance_level = ar_price
    test_count, st_price = _count_support_tests(work, ar_idx, as_of_idx, band_low, band_high)
    if test_count < 2:
        return None

    ps_price = _find_ps_price(work, sc_idx, sc_atr)
    range_width = max(resistance_level - support_level, 0.0)
    range_width_atr = range_width / sc_atr if sc_atr > 0 else 0.0
    phase_b_ut = _detect_phase_b_ut(work, ar_idx, as_of_idx, resistance_level)
    as_of_ts = pd.Timestamp(work.iloc[as_of_idx]["datetime"])
    days_in_acc = max(0, (as_of_ts.normalize() - sc_ts.normalize()).days)
    phase_b_duration = max(0, as_of_idx - ar_idx)

    return AccumulationRange(
        ps_price=round(ps_price, 6),
        sc_price=round(sc_price, 6),
        ar_price=round(ar_price, 6),
        st_price=round(st_price, 6),
        support_level=round(support_level, 6),
        resistance_level=round(resistance_level, 6),
        range_width_atr=round(range_width_atr, 4),
        test_count=test_count,
        phase_b_ut_occurred=phase_b_ut,
        days_in_accumulation=days_in_acc,
        phase_b_duration=phase_b_duration,
        is_valid=True,
        sc_bar_index=sc_idx,
        ar_bar_index=ar_idx,
        sc_timestamp=sc_ts,
        ar_timestamp=ar_ts,
    )


def detect_accumulation_range(
    df: pd.DataFrame,
    lookback_bars: int = LOOKBACK_BARS,
    atr_period: int = ATR_PERIOD,
) -> AccumulationRange | None:
    """H1 蓄積レンジを検出（評価時点 = df 末尾）。"""
    work = _prepare_df(df)
    if len(work) < atr_period + MIN_BARS_AFTER_SC + 5:
        return None

    as_of_idx = len(work) - 1
    start_idx = max(atr_period, as_of_idx - lookback_bars)
    atr_series = compute_atr(work.iloc[: as_of_idx + 1], atr_period)

    for sc_idx in range(as_of_idx - MIN_BARS_AFTER_SC, start_idx - 1, -1):
        acc = _try_build_accumulation(work, sc_idx, atr_series, as_of_idx)
        if acc is not None and acc.is_valid:
            return acc
    return None


def _volatility_percentile(work: pd.DataFrame, bar_index: int, atr_series: pd.Series) -> float:
    start = max(ATR_PERIOD, bar_index - VOLATILITY_LOOKBACK + 1)
    if bar_index < start:
        return 50.0
    window = atr_series.iloc[start : bar_index + 1].dropna()
    if window.empty:
        return 50.0
    current = float(atr_series.iloc[bar_index])
    if not np.isfinite(current):
        return 50.0
    rank = (window <= current).sum()
    return round(100.0 * rank / len(window), 4)


def _atr_regime(vol_pct: float) -> AtrRegime:
    if vol_pct < 33.0:
        return "LOW"
    if vol_pct < 66.0:
        return "NORMAL"
    return "HIGH"


def _compute_adr_remaining(work: pd.DataFrame, bar_index: int, atr: float) -> float:
    if bar_index < 1 or atr <= 0:
        return 1.0
    day_norm = pd.to_datetime(work["datetime"]).dt.normalize()
    current_day = day_norm.iloc[bar_index]
    day_mask = day_norm == current_day
    day_high = float(work.loc[day_mask, "high"].max())
    day_low = float(work.loc[day_mask, "low"].min())
    adr_used = day_high - day_low

    unique_days = day_norm.unique()
    day_pos = np.where(unique_days == np.datetime64(current_day))[0]
    if len(day_pos) == 0:
        return 1.0
    end = int(day_pos[0])
    start = max(0, end - ADR_LOOKBACK_DAYS)
    ranges: list[float] = []
    for day in unique_days[start:end]:
        mask = day_norm == day
        if int(mask.sum()) >= 4:
            ranges.append(float(work.loc[mask, "high"].max() - work.loc[mask, "low"].min()))
    if not ranges:
        return 1.0
    adr_avg = float(np.mean(ranges))
    if adr_avg <= 0:
        return 1.0
    return max(0.0, (adr_avg - adr_used) / adr_avg)


def _distance_to_rolling_low(
    work: pd.DataFrame,
    bar_index: int,
    lookback: int,
    price: float,
    atr: float,
) -> float:
    if atr <= 0 or bar_index < 0:
        return 0.0
    start = max(0, bar_index - lookback + 1)
    rolling_low = float(work.iloc[start : bar_index + 1]["low"].min())
    return (price - rolling_low) / atr


def _htf_trend_d1(work: pd.DataFrame, as_of: pd.Timestamp) -> str:
    clipped = clip_as_of(work, as_of)
    if len(clipped) < 30:
        return "NEUTRAL"
    d1 = resample_to_htf(clipped, "1D")
    if len(d1) < 20:
        return "NEUTRAL"
    result = analyze_htf_trend(clipped, as_of, htf_df=d1, bar_hours=24)
    return str(result.direction)


def _session_normalized_volume_ratio(
    work: pd.DataFrame,
    bar_index: int,
    lookback_days: int = SESSION_VOLUME_LOOKBACK_DAYS,
) -> float:
    """Spring 足の出来高 ÷ 同一時間帯の過去数日平均（ティックボリューム正規化）。"""
    val = float(work.iloc[bar_index]["volume"])
    samples = _session_volume_samples(work, bar_index, lookback_days)
    if samples:
        baseline = float(np.mean(samples))
    else:
        start = max(0, bar_index - VOLUME_LOOKBACK + 1)
        window = work.iloc[start:bar_index]["volume"].astype(float)
        baseline = float(window.mean()) if not window.empty else val
    if baseline <= 0:
        return 1.0
    return val / baseline


def _volume_ratio(work: pd.DataFrame, bar_index: int, lookback: int = VOLUME_LOOKBACK) -> float:
    return _session_normalized_volume_ratio(work, bar_index)


def _spring_recovery_time_bars(spring_idx: int, recovery_idx: int) -> int:
    """Spring 足から support 回復足までの経過バー数（H1 なら時間相当）。"""
    return max(0, recovery_idx - spring_idx)


def _resistance_distance_atr(entry_price: float, resistance_level: float, atr: float) -> float:
    """エントリーから AR 抵抗（TP 目標）までの距離 ÷ ATR。"""
    if atr <= 0:
        return 0.0
    return max(0.0, (resistance_level - entry_price) / atr)


def _upthrust_wick_ratio(row: pd.Series) -> float:
    """Upthrust 足の上ヒゲ比率 = 上ヒゲ / (高値 - 安値)。"""
    high = float(row["high"])
    low = float(row["low"])
    span = high - low
    if span <= 0:
        return 0.0
    upper_wick = high - max(float(row["open"]), float(row["close"]))
    return max(0.0, min(1.0, upper_wick / span))


def _volume_percentile(work: pd.DataFrame, bar_index: int, lookback: int = VOLUME_LOOKBACK) -> float:
    start = max(0, bar_index - lookback + 1)
    window = work.iloc[start : bar_index + 1]["volume"].astype(float)
    if window.empty:
        return 50.0
    current = float(work.iloc[bar_index]["volume"])
    rank = (window <= current).sum()
    return round(100.0 * rank / len(window), 4)


def _position_in_range(price: float, range_low: float, range_high: float) -> float:
    width = range_high - range_low
    if width <= 0:
        return 0.5
    return max(0.0, min(1.0, (price - range_low) / width))


def _minutes_from_session_open(ts: pd.Timestamp, session: SessionType) -> int:
    """セッション開始からの経過分（簡易 — ロンドン 08:00 UTC 基準）。"""
    ts = pd.Timestamp(ts)
    if session == "LONDON":
        open_ts = ts.normalize() + pd.Timedelta(hours=8)
    elif session == "NY":
        open_ts = ts.normalize() + pd.Timedelta(hours=13)
    elif session == "ASIA":
        open_ts = ts.normalize() + pd.Timedelta(hours=0)
    else:
        return 0
    if ts < open_ts:
        open_ts -= pd.Timedelta(days=1)
    return max(0, int((ts - open_ts).total_seconds() // 60))


def _adr_used_fraction(work: pd.DataFrame, bar_index: int) -> float:
    if bar_index < 1:
        return 0.0
    remaining = _compute_adr_remaining(work, bar_index, _atr_at(work, bar_index) or 1.0)
    return max(0.0, min(1.0, 1.0 - remaining))


def _liquidity_distances(
    work: pd.DataFrame,
    bar_index: int,
    price: float,
    atr: float,
) -> dict[str, float]:
    """日次 / セッション高安からの ATR 正規化距離。"""
    if bar_index < 0 or atr <= 0:
        return {
            "distance_daily_high": 0.0,
            "distance_daily_low": 0.0,
            "distance_session_high": 0.0,
            "distance_session_low": 0.0,
        }
    day_norm = pd.to_datetime(work["datetime"]).dt.normalize()
    current_day = day_norm.iloc[bar_index]
    day_mask = day_norm == current_day
    day_slice = work.loc[day_mask]
    day_high = float(day_slice["high"].max())
    day_low = float(day_slice["low"].min())
    session = resolve_cspa_session_type(pd.Timestamp(work.iloc[bar_index]["datetime"]))
    if session == "LONDON":
        hour_mask = day_slice["datetime"].apply(lambda t: 8 <= pd.Timestamp(t).hour < 17)
    elif session == "NY":
        hour_mask = day_slice["datetime"].apply(lambda t: 13 <= pd.Timestamp(t).hour < 22)
    elif session == "ASIA":
        hour_mask = day_slice["datetime"].apply(lambda t: pd.Timestamp(t).hour < 8)
    else:
        hour_mask = day_mask
    sess_slice = day_slice.loc[hour_mask] if hour_mask.any() else day_slice
    sess_high = float(sess_slice["high"].max())
    sess_low = float(sess_slice["low"].min())
    return {
        "distance_daily_high": (day_high - price) / atr,
        "distance_daily_low": (price - day_low) / atr,
        "distance_session_high": (sess_high - price) / atr,
        "distance_session_low": (price - sess_low) / atr,
    }


def classify_wyckoff_macro_phase(
    *,
    accumulation: AccumulationRange,
    entry_price: float,
    htf_trend_d1: str,
    reversal_pattern: ReversalPattern,
) -> WyckoffMacroPhase:
    """
    Wyckoff マクロフェーズの簡易ラベル（ログ専用 — 売買条件には未使用）。

    Accumulation レンジ構造 + HTF トレンド + パターン種別から推定する。
    """
    if not accumulation.is_valid:
        return "UNKNOWN"
    pos = _position_in_range(
        entry_price,
        accumulation.support_level,
        accumulation.resistance_level,
    )
    if reversal_pattern == "UPTHRUST" or accumulation.phase_b_ut_occurred:
        if htf_trend_d1 == "BEAR":
            return "DISTRIBUTION"
        return "DISTRIBUTION" if pos >= 0.55 else "ACCUMULATION"
    if htf_trend_d1 == "BULL" and pos >= 0.6:
        return "MARKUP"
    if htf_trend_d1 == "BEAR" and pos <= 0.4:
        return "MARKDOWN"
    return "ACCUMULATION"


def _learning_log_features(
    exec_work: pd.DataFrame,
    trigger_idx: int,
    recovery_idx: int,
    direction: TradeDirection,
    atr: float,
    range_width: float,
) -> dict[str, float | int]:
    """
    EVT / カルマン学習用の派生特徴量（価格系列ヒューリスティクスのみ — フィルタ未使用）。
    """
    defaults = {
        "trend_age_bars": 0,
        "impulse_quality": 0.0,
        "impulse_atr_ratio": 0.0,
        "stagnation_duration": 0,
        "stagnation_width": 0.0,
        "breakout_velocity": 0.0,
        "breakout_momentum_ratio": 0.0,
        "tail_move_after_entry": 0.0,
    }
    if trigger_idx < 0 or recovery_idx >= len(exec_work) or atr <= 0:
        return defaults

    lookback = min(20, trigger_idx)
    window = exec_work.iloc[trigger_idx - lookback : trigger_idx + 1]
    closes = window["close"].astype(float).values
    impulse_quality = 0.0
    impulse_atr = 0.0
    trend_age = 0
    if len(closes) >= 3:
        net = abs(float(closes[-1] - closes[0]))
        path = float(np.abs(np.diff(closes)).sum())
        impulse_quality = net / path if path > 0 else 0.0
        impulse_atr = net / atr
        for i in range(len(closes) - 1, 0, -1):
            delta = closes[i] - closes[i - 1]
            if direction == "BUY" and delta > 0:
                trend_age += 1
            elif direction == "SELL" and delta < 0:
                trend_age += 1
            else:
                break

    trigger_row = exec_work.iloc[trigger_idx]
    recovery_row = exec_work.iloc[recovery_idx]
    span = max(1, recovery_idx - trigger_idx)
    breakout_move = abs(float(recovery_row["close"]) - float(trigger_row["close"]))
    breakout_velocity = breakout_move / (span * atr)
    stagnation_duration = max(0, span - 1)
    stagnation_width = range_width / atr if atr > 0 else 0.0
    vol = max(1.0, float(trigger_row["volume"]))
    return {
        "trend_age_bars": int(trend_age),
        "impulse_quality": round(float(impulse_quality), 4),
        "impulse_atr_ratio": round(float(impulse_atr), 4),
        "stagnation_duration": int(stagnation_duration),
        "stagnation_width": round(float(stagnation_width), 4),
        "breakout_velocity": round(float(breakout_velocity), 4),
        "breakout_momentum_ratio": round(float(breakout_velocity * vol), 4),
        "tail_move_after_entry": round(float(breakout_move / atr), 4),
    }


def _spring_wick_ratio(row: pd.Series) -> float:
    """Spring 足の下ヒゲ比率 = 下ヒゲ / (高値 - 安値)。"""
    high = float(row["high"])
    low = float(row["low"])
    span = high - low
    if span <= 0:
        return 0.0
    lower_wick = min(float(row["open"]), float(row["close"])) - low
    return max(0.0, min(1.0, lower_wick / span))


def _range_compression(
    h1_work: pd.DataFrame,
    accumulation: AccumulationRange,
    spring_ts: pd.Timestamp,
    atr: float,
) -> float:
    """
    Phase B レンジ幅 ÷ SC–AR 全体レンジ幅（ATR 正規化比）。

    H1 上の AR〜Spring 間を Phase B として評価。
    """
    if atr <= 0 or accumulation.range_width_atr <= 0:
        return 1.0
    dt = pd.to_datetime(h1_work["datetime"])
    mask = (dt >= accumulation.ar_timestamp) & (dt <= spring_ts)
    window = h1_work.loc[mask]
    if window.empty:
        return 1.0
    phase_b_width_atr = float(window["high"].max() - window["low"].min()) / atr
    return phase_b_width_atr / accumulation.range_width_atr


def _compute_kalman_recovery_features(
    exec_work: pd.DataFrame,
    spring_idx: int,
    recovery_idx: int,
) -> dict[str, float | int]:
    """学習用カルマン特徴量プレースホルダ（フィルタ未実装 — 中立値のみ）。"""
    _ = exec_work, spring_idx, recovery_idx
    return _neutral_kalman_features()


def compute_recovery_close_ratio(
    work: pd.DataFrame,
    trigger_idx: int,
    recovery_idx: int,
    *,
    direction: TradeDirection = "BUY",
) -> float:
    """
    反発期間（トリガー足〜回復/エントリー足）の実体比率。

    BUY(Spring): 陽線実体合計 / 陰線実体合計
    SELL(Upthrust): 陰線実体合計 / 陽線実体合計
    """
    if recovery_idx < trigger_idx:
        return 0.0
    favorable = 0.0
    counter = 0.0
    for _, row in work.iloc[trigger_idx : recovery_idx + 1].iterrows():
        open_p = float(row["open"])
        close_p = float(row["close"])
        body = abs(close_p - open_p)
        if body <= 0:
            continue
        is_bull = close_p > open_p
        is_bear = close_p < open_p
        if direction == "BUY":
            if is_bull:
                favorable += body
            elif is_bear:
                counter += body
        else:
            if is_bear:
                favorable += body
            elif is_bull:
                counter += body
    if counter <= 0:
        return favorable if favorable > 0 else 0.0
    return favorable / counter


def _recovery_quality_metrics(
    work: pd.DataFrame,
    spring_idx: int,
    recovery_idx: int,
    extreme_price: float,
    atr: float,
    *,
    direction: TradeDirection = "BUY",
) -> dict[str, float | int]:
    if recovery_idx < spring_idx or atr <= 0:
        return {
            "spring_recovery_atr": 0.0,
            "recovery_duration_bars": 0,
            "recovery_velocity": 0.0,
            "recovery_velocity_atr": 0.0,
            "recovery_close_ratio": 0.0,
            "recovery_acceleration": 0.0,
            "consecutive_higher_closes": 0,
            "positive_close_ratio": 0.0,
            "directional_efficiency": 0.0,
            "noise_ratio": 1.0,
        }

    recovery_close = float(work.iloc[recovery_idx]["close"])
    if direction == "BUY":
        spring_recovery_atr = (recovery_close - extreme_price) / atr
    else:
        spring_recovery_atr = (extreme_price - recovery_close) / atr
    recovery_duration_bars = max(0, recovery_idx - spring_idx)
    duration = max(1, recovery_duration_bars)
    recovery_velocity_atr = spring_recovery_atr / duration
    recovery_close_ratio = compute_recovery_close_ratio(
        work,
        spring_idx,
        recovery_idx,
        direction=direction,
    )

    quality_start = recovery_idx
    quality_end = min(recovery_idx + RECOVERY_QUALITY_BARS, len(work) - 1)
    quality_closes = work.iloc[quality_start : quality_end + 1]["close"].astype(float).values
    deltas = np.diff(quality_closes) if len(quality_closes) >= 2 else np.array([], dtype=float)
    if direction == "SELL":
        positive_mask = deltas < 0
    else:
        positive_mask = deltas > 0
    positive_ratio = float(positive_mask.sum() / max(len(deltas), 1))
    consecutive = 0
    for d in reversed(deltas):
        if direction == "SELL" and d < 0:
            consecutive += 1
        elif direction == "BUY" and d > 0:
            consecutive += 1
        else:
            break

    if len(deltas) >= 2:
        first_vel = abs(deltas[0]) / atr if atr > 0 else 0.0
        last_vel = abs(deltas[-1]) / atr if atr > 0 else 0.0
        recovery_acceleration = last_vel - first_vel
        if direction == "SELL":
            recovery_acceleration = -recovery_acceleration
    else:
        recovery_acceleration = 0.0

    if len(deltas) >= 1 and len(quality_closes) >= 2:
        net = abs(float(quality_closes[-1] - quality_closes[0]))
        path = float(np.abs(deltas).sum())
        directional_efficiency = net / path if path > 0 else 0.0
    else:
        directional_efficiency = 0.0
    noise_ratio = 1.0 - directional_efficiency

    return {
        "spring_recovery_atr": spring_recovery_atr,
        "recovery_duration_bars": recovery_duration_bars,
        "recovery_velocity": recovery_velocity_atr,
        "recovery_velocity_atr": recovery_velocity_atr,
        "recovery_close_ratio": recovery_close_ratio,
        "recovery_acceleration": recovery_acceleration,
        "consecutive_higher_closes": consecutive,
        "positive_close_ratio": positive_ratio,
        "directional_efficiency": directional_efficiency,
        "noise_ratio": noise_ratio,
    }


def wr_mode_from_direction(direction: TradeDirection) -> WrMode:
    """WR エントリーモード — SELL=Upthrust, BUY=Spring。"""
    return "UPTHRUST" if direction == "SELL" else "SPRING"


def build_wyckoff_features(
    *,
    exec_work: pd.DataFrame,
    h1_work: pd.DataFrame,
    accumulation: AccumulationRange,
    trigger_idx: int,
    recovery_idx: int,
    extreme_price: float,
    penetration_or_height: float,
    pattern_duration_bars: int,
    attempt_number: int,
    volume_ratio: float,
    entry_price: float,
    atr: float,
    atr_series: pd.Series,
    pair: str,
    reversal_pattern: ReversalPattern = "SPRING",
    direction: TradeDirection = "BUY",
    smt_intensity: float = 0.0,
    smt_diff: float = 0.0,
    smt_leader: str = "NONE",
    spread_pips: float = WYCKOFF_BT_SPREAD_PIPS,
) -> WyckoffFeatures:
    timestamp = pd.Timestamp(exec_work.iloc[recovery_idx]["datetime"])
    band_width = 2.0 * SUPPORT_BAND_ATR * atr
    if reversal_pattern == "SPRING":
        support_penetration_percent = (penetration_or_height / band_width * 100.0) if band_width > 0 else 0.0
    else:
        support_penetration_percent = 0.0
    pip = pip_size_for_pair(pair)
    spring_velocity = (penetration_or_height / pip / max(pattern_duration_bars, 1)) if pip > 0 else 0.0
    recovery = _recovery_quality_metrics(
        exec_work,
        trigger_idx,
        recovery_idx,
        extreme_price,
        atr,
        direction=direction,
    )
    kalman = _compute_kalman_recovery_features(exec_work, trigger_idx, recovery_idx)
    h1_atr_series = compute_atr(h1_work.iloc[: len(h1_work)], ATR_PERIOD)
    h1_idx = len(h1_work) - 1
    vol_pct = _volatility_percentile(h1_work, h1_idx, h1_atr_series)
    trigger_row = exec_work.iloc[trigger_idx]
    h1_atr = _atr_at(h1_work, h1_idx, h1_atr_series) or atr
    session = resolve_cspa_session_type(timestamp)
    htf = _htf_trend_d1(h1_work, timestamp)
    range_low = accumulation.support_level
    range_high = accumulation.resistance_level
    range_width = max(range_high - range_low, 0.0)
    pos_in_range = _position_in_range(entry_price, range_low, range_high)
    liquidity = _liquidity_distances(exec_work, recovery_idx, entry_price, atr)
    learning = _learning_log_features(
        exec_work,
        trigger_idx,
        recovery_idx,
        direction,
        atr,
        range_width,
    )
    macro_phase = classify_wyckoff_macro_phase(
        accumulation=accumulation,
        entry_price=entry_price,
        htf_trend_d1=htf,
        reversal_pattern=reversal_pattern,
    )
    spring_depth = penetration_or_height / atr if atr > 0 and reversal_pattern == "SPRING" else 0.0
    upthrust_height = penetration_or_height / atr if atr > 0 and reversal_pattern == "UPTHRUST" else 0.0
    wick_ratio = (
        _spring_wick_ratio(trigger_row)
        if reversal_pattern == "SPRING"
        else _upthrust_wick_ratio(trigger_row)
    )
    phase_label: PhaseLabel = "SPRING" if reversal_pattern == "SPRING" else "ACCUMULATION"

    return WyckoffFeatures(
        phase_identified=phase_label,
        support_test_count=accumulation.test_count,
        days_in_accumulation=accumulation.days_in_accumulation,
        phase_b_duration=accumulation.phase_b_duration,
        range_width_atr=accumulation.range_width_atr,
        spring_attempt_number=attempt_number,
        spring_depth_atr=spring_depth,
        spring_velocity=spring_velocity,
        spring_duration_bars=pattern_duration_bars,
        support_penetration_percent=support_penetration_percent,
        spring_volume_ratio=volume_ratio,
        spring_recovery_time=int(recovery["recovery_duration_bars"]),
        resistance_distance=_resistance_distance_atr(entry_price, accumulation.resistance_level, atr),
        spring_wick_ratio=wick_ratio,
        range_compression=_range_compression(h1_work, accumulation, timestamp, h1_atr),
        spring_recovery_atr=float(recovery["spring_recovery_atr"]),
        recovery_duration_bars=int(recovery["recovery_duration_bars"]),
        recovery_velocity=float(recovery["recovery_velocity"]),
        recovery_velocity_atr=float(recovery["recovery_velocity_atr"]),
        recovery_close_ratio=float(recovery["recovery_close_ratio"]),
        recovery_acceleration=float(recovery["recovery_acceleration"]),
        kalman_velocity_at_entry=float(kalman["kalman_velocity_at_entry"]),
        velocity_positive_bars=int(kalman["velocity_positive_bars"]),
        kalman_noise_ratio=float(kalman["kalman_noise_ratio"]),
        consecutive_higher_closes=int(recovery["consecutive_higher_closes"]),
        positive_close_ratio=float(recovery["positive_close_ratio"]),
        directional_efficiency=float(recovery["directional_efficiency"]),
        noise_ratio=float(recovery["noise_ratio"]),
        session_type=session,
        volatility_percentile=vol_pct,
        atr_regime=_atr_regime(vol_pct),
        adr_remaining=_compute_adr_remaining(exec_work, recovery_idx, atr),
        distance_weekly_low=_distance_to_rolling_low(
            exec_work, recovery_idx, WEEKLY_BARS * 4, entry_price, atr
        ),
        distance_monthly_low=_distance_to_rolling_low(
            exec_work, recovery_idx, MONTHLY_BARS * 4, entry_price, atr
        ),
        htf_trend_d1=htf,
        phase_b_ut_occurred=accumulation.phase_b_ut_occurred,
        reversal_pattern=reversal_pattern,
        wr_mode=wr_mode_from_direction(direction),
        wyckoff_phase=macro_phase,
        range_width=range_width,
        range_age_bars=_range_age_bars(accumulation, timestamp),
        upthrust_height=upthrust_height,
        recovery_speed=float(recovery["recovery_velocity"]),
        volume_ratio=volume_ratio,
        volume_percentile=_volume_percentile(exec_work, trigger_idx),
        volume_expansion=max(0.0, volume_ratio - 1.0),
        distance_from_range_low=(entry_price - range_low) / atr if atr > 0 else 0.0,
        distance_from_range_high=(range_high - entry_price) / atr if atr > 0 else 0.0,
        position_in_range=pos_in_range,
        atr_ratio=accumulation.range_width_atr,
        smt_intensity=smt_intensity,
        smt_diff=smt_diff,
        smt_leader=smt_leader,
        minutes_from_session_open=_minutes_from_session_open(timestamp, session),
        adr_used=_adr_used_fraction(exec_work, recovery_idx),
        distance_daily_high=liquidity["distance_daily_high"],
        distance_daily_low=liquidity["distance_daily_low"],
        distance_session_high=liquidity["distance_session_high"],
        distance_session_low=liquidity["distance_session_low"],
        spread=spread_pips,
        spread_percentile=_spread_percentile(spread_pips),
        tail_move_after_entry=float(learning["tail_move_after_entry"]),
        trend_age_bars=int(learning["trend_age_bars"]),
        impulse_quality=float(learning["impulse_quality"]),
        impulse_atr_ratio=float(learning["impulse_atr_ratio"]),
        stagnation_duration=int(learning["stagnation_duration"]),
        stagnation_width=float(learning["stagnation_width"]),
        breakout_velocity=float(learning["breakout_velocity"]),
        breakout_momentum_ratio=float(learning["breakout_momentum_ratio"]),
    )


def detect_spring(
    exec_df: pd.DataFrame,
    accumulation: AccumulationRange,
    pip_size: float,
    *,
    pair: str = WYCKOFF_PAIR_PRIMARY,
    spring_bar_index: int | None = None,
    h1_df: pd.DataFrame | None = None,
) -> ReversalSetup | None:
    """M15 上の Spring 判定（H1 accumulation の support/resistance 価格帯を参照）。"""
    _ = pip_size
    exec_work = _prepare_df(exec_df)
    h1_work = _prepare_df(h1_df) if h1_df is not None else exec_work
    if not accumulation.is_valid:
        return None

    idx = len(exec_work) - 1 if spring_bar_index is None else spring_bar_index
    if idx < 0 or idx >= len(exec_work):
        return None

    spring_ts = pd.Timestamp(exec_work.iloc[idx]["datetime"])
    if spring_ts < accumulation.ar_timestamp:
        return None

    pre_atr_series = compute_atr(exec_work.iloc[: idx + 1], ATR_PERIOD)
    atr = _atr_at(exec_work, idx, pre_atr_series)
    if atr <= 0:
        return None

    row = exec_work.iloc[idx]
    support_level = accumulation.support_level
    spring_low = float(row["low"])
    if spring_low >= support_level:
        return None

    penetration = support_level - spring_low
    if penetration < SPRING_MIN_DEPTH_ATR * atr or penetration > SPRING_MAX_DEPTH_ATR * atr:
        return None

    recovery_idx = idx
    if float(row["close"]) < support_level:
        if idx + 1 >= len(exec_work):
            return None
        if float(exec_work.iloc[idx + 1]["close"]) < support_level:
            return None
        recovery_idx = idx + 1

    h1_clipped = clip_as_of(h1_work, pd.Timestamp(exec_work.iloc[recovery_idx]["datetime"]))
    h1_atr_series = compute_atr(h1_work, ATR_PERIOD)

    spring_duration_bars = max(1, recovery_idx - idx + 1)
    entry_price = float(exec_work.iloc[recovery_idx]["close"])
    stop_loss = accumulation.sc_price - SL_BUFFER_ATR * atr
    take_profit = accumulation.ar_price
    risk = entry_price - stop_loss
    reward = take_profit - entry_price
    if risk <= 0:
        return None
    rr = reward / risk
    if rr < MIN_RR:
        return None

    spring_attempt_number = _count_spring_attempts(
        h1_work,
        accumulation,
        spring_ts,
        support_level,
        h1_atr_series,
    )
    volume_on_spring = _session_normalized_volume_ratio(exec_work, idx)
    if volume_on_spring < SPRING_VOLUME_RATIO_MIN:
        return None

    features = build_wyckoff_features(
        exec_work=exec_work,
        h1_work=h1_clipped,
        accumulation=accumulation,
        trigger_idx=idx,
        recovery_idx=recovery_idx,
        extreme_price=spring_low,
        penetration_or_height=penetration,
        pattern_duration_bars=spring_duration_bars,
        attempt_number=spring_attempt_number,
        volume_ratio=volume_on_spring,
        entry_price=entry_price,
        atr=atr,
        atr_series=pre_atr_series,
        pair=pair.upper(),
        reversal_pattern="SPRING",
        direction="BUY",
    )

    return ReversalSetup(
        timestamp=pd.Timestamp(exec_work.iloc[recovery_idx]["datetime"]),
        pair=pair.upper(),
        accumulation=accumulation,
        spring_depth_atr=features.spring_depth_atr,
        spring_velocity=features.spring_velocity,
        spring_duration_bars=spring_duration_bars,
        support_penetration_percent=features.support_penetration_percent,
        spring_attempt_number=spring_attempt_number,
        volume_on_spring=volume_on_spring,
        entry_price=round(entry_price, 6),
        stop_loss=round(stop_loss, 6),
        take_profit=round(take_profit, 6),
        risk_reward=round(rr, 4),
        spring_bar_index=idx,
        recovery_bar_index=recovery_idx,
        wyckoff_features=features,
        reason_codes=(),
        direction="BUY",
        reversal_pattern="SPRING",
    )


def detect_upthrust(
    exec_df: pd.DataFrame,
    accumulation: AccumulationRange,
    pip_size: float,
    *,
    pair: str = WYCKOFF_PAIR_PRIMARY,
    upthrust_bar_index: int | None = None,
    h1_df: pd.DataFrame | None = None,
) -> ReversalSetup | None:
    """M15 上の Upthrust 判定（レンジ上限ブレイク失敗 → レンジ内回帰）。"""
    _ = pip_size
    exec_work = _prepare_df(exec_df)
    h1_work = _prepare_df(h1_df) if h1_df is not None else exec_work
    if not accumulation.is_valid:
        return None

    idx = len(exec_work) - 1 if upthrust_bar_index is None else upthrust_bar_index
    if idx < 0 or idx >= len(exec_work):
        return None

    upthrust_ts = pd.Timestamp(exec_work.iloc[idx]["datetime"])
    if upthrust_ts < accumulation.ar_timestamp:
        return None

    pre_atr_series = compute_atr(exec_work.iloc[: idx + 1], ATR_PERIOD)
    atr = _atr_at(exec_work, idx, pre_atr_series)
    if atr <= 0:
        return None

    row = exec_work.iloc[idx]
    resistance_level = accumulation.resistance_level
    upthrust_high = float(row["high"])
    close = float(row["close"])

    if upthrust_high <= resistance_level:
        return None
    if close >= resistance_level:
        return None
    if _upthrust_wick_ratio(row) < UPTHRUST_WICK_RATIO_MIN:
        return None

    height = upthrust_high - resistance_level
    if height < UPTHRUST_MIN_HEIGHT_ATR * atr or height > UPTHRUST_MAX_HEIGHT_ATR * atr:
        return None

    pos_at_high = _position_in_range(upthrust_high, accumulation.support_level, resistance_level)
    if pos_at_high < 0.55:
        return None

    recovery_idx = idx
    if close >= resistance_level:
        if idx + 1 >= len(exec_work):
            return None
        if float(exec_work.iloc[idx + 1]["close"]) >= resistance_level:
            return None
        recovery_idx = idx + 1

    volume_on_upthrust = _session_normalized_volume_ratio(exec_work, idx)
    if volume_on_upthrust < UPTHRUST_VOLUME_RATIO_MIN:
        return None

    h1_clipped = clip_as_of(h1_work, pd.Timestamp(exec_work.iloc[recovery_idx]["datetime"]))
    h1_atr_series = compute_atr(h1_work, ATR_PERIOD)

    pattern_duration_bars = max(1, recovery_idx - idx + 1)
    entry_price = float(exec_work.iloc[recovery_idx]["close"])
    stop_loss = accumulation.ar_price + SL_BUFFER_ATR * atr
    take_profit = accumulation.support_level
    risk = stop_loss - entry_price
    reward = entry_price - take_profit
    if risk <= 0 or reward <= 0:
        return None
    rr = reward / risk
    if rr < MIN_RR:
        return None

    attempt_number = _count_upthrust_attempts(
        h1_work,
        accumulation,
        upthrust_ts,
        resistance_level,
        h1_atr_series,
    )
    features = build_wyckoff_features(
        exec_work=exec_work,
        h1_work=h1_clipped,
        accumulation=accumulation,
        trigger_idx=idx,
        recovery_idx=recovery_idx,
        extreme_price=upthrust_high,
        penetration_or_height=height,
        pattern_duration_bars=pattern_duration_bars,
        attempt_number=attempt_number,
        volume_ratio=volume_on_upthrust,
        entry_price=entry_price,
        atr=atr,
        atr_series=pre_atr_series,
        pair=pair.upper(),
        reversal_pattern="UPTHRUST",
        direction="SELL",
    )

    return ReversalSetup(
        timestamp=pd.Timestamp(exec_work.iloc[recovery_idx]["datetime"]),
        pair=pair.upper(),
        accumulation=accumulation,
        spring_depth_atr=0.0,
        spring_velocity=features.spring_velocity,
        spring_duration_bars=pattern_duration_bars,
        support_penetration_percent=0.0,
        spring_attempt_number=attempt_number,
        volume_on_spring=volume_on_upthrust,
        entry_price=round(entry_price, 6),
        stop_loss=round(stop_loss, 6),
        take_profit=round(take_profit, 6),
        risk_reward=round(rr, 4),
        spring_bar_index=idx,
        recovery_bar_index=recovery_idx,
        wyckoff_features=features,
        reason_codes=(),
        direction="SELL",
        reversal_pattern="UPTHRUST",
        upthrust_height_atr=features.upthrust_height,
    )


def _ws_pyramid_enabled() -> bool:
    """WR 固有のピラミッド有効判定（環境変数 WS_PYRAMID_ENABLED 優先）。"""
    if is_wyckoff_pure_bt_mode():
        return False
    if not WS_PYRAMID_ENABLED:
        return False
    from pyramid_manager import is_pyramid_enabled

    return is_pyramid_enabled(SETUP_TYPE)


def _pyramid_time_limit_exec_bars(bar_minutes: int = WYCKOFF_EXEC_BAR_MINUTES) -> int:
    """H1 換算 WS_PYRAMID_TIME_LIMIT_BARS を執行足本数へ変換。"""
    if bar_minutes <= 0:
        return WS_PYRAMID_TIME_LIMIT_BARS
    if bar_minutes >= WYCKOFF_MONITOR_BAR_MINUTES:
        return WS_PYRAMID_TIME_LIMIT_BARS
    scale = max(1, WYCKOFF_MONITOR_BAR_MINUTES // bar_minutes)
    return WS_PYRAMID_TIME_LIMIT_BARS * scale


def simulate_wyckoff_pyramid(
    setup: ReversalSetup,
    h1_df: pd.DataFrame,
    m15_df: pd.DataFrame,
    *,
    bar_minutes: int = WYCKOFF_EXEC_BAR_MINUTES,
    initial_lot: float = 1.0,
    daily_dd_remaining_percent: float = 5.0,
    max_holding_bars: int = MAX_HOLDING_BARS,
) -> WyckoffSpringSimResult:
    """
    Wyckoff Reversal ピラミッド L5 シミュレーション（BE + R トリガー + タイムリミット）。

    BACKTEST_MODE 以外では呼び出し不可（ルックアヘッド / オフライン専用）。
    """
    from pyramid_manager import PyramidManager, is_backtest_mode

    if not is_backtest_mode():
        raise RuntimeError("simulate_wyckoff_pyramid is only available in BACKTEST_MODE")

    from strategies.bt_ohlcv import as_ohlcv
    from strategies.archive.cspa_arrays import atr_at_index, compute_atr_np

    exec_arr = as_ohlcv(m15_df)
    start_index = int(setup.recovery_bar_index)
    entry = float(setup.entry_price)
    stop_loss = float(setup.stop_loss)
    take_profit = float(setup.take_profit)
    direction = setup.direction

    base_risk = abs(entry - stop_loss)
    if base_risk <= 0 or initial_lot <= 0 or start_index < 0 or start_index >= exec_arr.length:
        return WyckoffSpringSimResult(
            result="LOSS",
            profit_r=-1.0,
            profit_pips=0.0,
            holding_minutes=0,
            pyramid_layers=0,
            pyramid_entry_prices=[],
            pyramid_lot_sizes=[],
            final_sl_at_close=round(stop_loss, 5),
            peak_unrealized_r=0.0,
            kalman_velocity_at_entry=0.0,
            kalman_decel_exit_triggered=False,
            time_limit_exit_triggered=False,
            pyramid_rejected_reason="",
        )

    atr_series = compute_atr_np(exec_arr.high, exec_arr.low, exec_arr.close, ATR_PERIOD)
    atr_raw = float(atr_at_index(atr_series, min(start_index, exec_arr.length - 1)))
    if not np.isfinite(atr_raw) or atr_raw <= 0.0:
        atr = base_risk
    else:
        atr = max(atr_raw, base_risk * 0.01)

    mgr = PyramidManager(
        trade_id="",
        direction=direction,
        atr=atr,
        base_risk=base_risk,
        breakeven_price=entry,
        take_profit=take_profit,
        initial_lot=initial_lot,
        initial_stop_loss=stop_loss,
        max_pyramid_layers=WS_PYRAMID_MAX_LAYERS,
        daily_dd_remaining_percent=daily_dd_remaining_percent,
    )

    time_limit_bars = _pyramid_time_limit_exec_bars(bar_minutes)
    end_index = min(start_index + max_holding_bars, exec_arr.length - 1)
    mfe_r = 0.0
    mae_r = 0.0
    realized_r = 0.0
    time_limit_exit_triggered = False
    pyramid_rejected_reason = ""

    for i in range(start_index + 1, end_index + 1):
        high = float(exec_arr.high[i])
        low = float(exec_arr.low[i])
        close = float(exec_arr.close[i])
        elapsed = (i - start_index) * bar_minutes
        bars_since_entry = i - start_index

        if direction == "SELL":
            mfe_r = max(mfe_r, (entry - low) / base_risk)
            mae_r = max(mae_r, (high - entry) / base_risk)
        else:
            mfe_r = max(mfe_r, (high - entry) / base_risk)
            mae_r = max(mae_r, (entry - low) / base_risk)

        if mgr.sl_hit_on_bar(high, low) and mgr.tp_hit_on_bar(high, low):
            exit_price = mgr.unified_stop_loss()
            res, pr, pp = mgr.close_at_price(exit_price)
            pr = min(pr, -0.01)
            total_r = max(-1.0, min(2.4, realized_r + pr))
            return _finalize_ws_sim_result(
                mgr,
                exit_price,
                elapsed,
                res,
                total_r,
                pp,
                0.0,
                False,
                time_limit_exit_triggered,
                pyramid_rejected_reason,
                mfe_r,
                mae_r,
            )

        if mgr.sl_hit_on_bar(high, low):
            exit_price = mgr.unified_stop_loss()
            res, pr, pp = mgr.close_at_price(exit_price)
            total_r = max(-1.0, min(2.4, realized_r + pr))
            return _finalize_ws_sim_result(
                mgr,
                exit_price,
                elapsed,
                res,
                total_r,
                pp,
                0.0,
                False,
                time_limit_exit_triggered,
                pyramid_rejected_reason,
                mfe_r,
                mae_r,
            )

        if mgr.tp_hit_on_bar(high, low):
            res, pr, pp = mgr.close_at_price(take_profit)
            total_r = max(-1.0, min(2.4, realized_r + max(pr, 2.0)))
            return _finalize_ws_sim_result(
                mgr,
                take_profit,
                elapsed,
                "WIN",
                total_r,
                pp,
                0.0,
                False,
                time_limit_exit_triggered,
                pyramid_rejected_reason,
                mfe_r,
                mae_r,
            )

        mgr.update_peak(high, low, close)

        if mgr.portfolio_unrealized_r(close) >= WS_PYRAMID_TRIGGER_R and not mgr._all_sl_at_breakeven():
            mgr.move_all_sl_to_breakeven()

        past_time_limit = bars_since_entry > time_limit_bars
        can_add, reason = mgr.can_add_pyramid(close, daily_dd_remaining_percent)
        ready_for_pyramid = (
            mgr._all_sl_at_breakeven()
            and mgr.portfolio_unrealized_r(close) >= WS_PYRAMID_TRIGGER_R
            and mgr.layer_count < mgr.max_pyramid_layers
        )
        if can_add:
            if mgr.portfolio_unrealized_r(close) < WS_PYRAMID_TRIGGER_R:
                can_add = False
                reason = "UNREALIZED_R_BELOW_TRIGGER"
            elif past_time_limit:
                can_add = False
                reason = "TIME_LIMIT"
                time_limit_exit_triggered = True
        elif past_time_limit and ready_for_pyramid:
            time_limit_exit_triggered = True

        if can_add:
            mgr.add_pyramid_layer(close)
        elif reason and reason != "OK":
            if reason == "SL_NOT_AT_BREAKEVEN":
                pyramid_rejected_reason = reason
            elif pyramid_rejected_reason != "SL_NOT_AT_BREAKEVEN":
                pyramid_rejected_reason = reason

    last_close = float(exec_arr.close[end_index])
    holding = (end_index - start_index) * bar_minutes
    res, pr, pp = mgr.close_at_price(last_close)
    total_r = max(-1.0, min(2.4, realized_r + pr))
    return _finalize_ws_sim_result(
        mgr,
        last_close,
        holding,
        res,
        total_r,
        pp,
        0.0,
        False,
        time_limit_exit_triggered,
        pyramid_rejected_reason,
        mfe_r,
        mae_r,
    )


def _finalize_ws_sim_result(
    mgr: Any,
    exit_price: float,
    holding_minutes: int,
    result: str,
    profit_r: float,
    profit_pips: float,
    kalman_velocity_at_entry: float,
    decel_exit_triggered: bool,
    time_limit_exit_triggered: bool,
    pyramid_rejected_reason: str,
    mfe_r: float,
    mae_r: float,
) -> WyckoffSpringSimResult:
    pyramid = mgr.to_result_fields(exit_price, holding_minutes, result, profit_r, profit_pips)
    return WyckoffSpringSimResult(
        result=pyramid.result,
        profit_r=round(profit_r, 4),
        profit_pips=round(profit_pips, 4),
        holding_minutes=int(holding_minutes),
        pyramid_layers=pyramid.pyramid_layers,
        pyramid_entry_prices=list(pyramid.pyramid_entry_prices),
        pyramid_lot_sizes=list(pyramid.pyramid_lot_sizes),
        final_sl_at_close=pyramid.final_sl_at_close,
        peak_unrealized_r=pyramid.peak_unrealized_r,
        kalman_velocity_at_entry=round(kalman_velocity_at_entry, 8),
        kalman_decel_exit_triggered=decel_exit_triggered,
        time_limit_exit_triggered=time_limit_exit_triggered,
        pyramid_rejected_reason=pyramid_rejected_reason,
        mfe=round(mfe_r, 4),
        mae=round(mae_r, 4),
    )


def compute_wyckoff_trade_excursions(
    pair_df: pd.DataFrame,
    start_index: int,
    entry: float,
    stop_loss: float,
    take_profit: float,
    *,
    max_holding_bars: int = MAX_HOLDING_BARS,
    direction: TradeDirection = "BUY",
) -> dict[str, float | str | int]:
    """L5 固定 SL/TP シャドー追跡（BUY / SELL 両対応）。"""
    from strategies.bt_l5 import compute_trade_excursions_np
    from strategies.bt_ohlcv import as_ohlcv

    ohlcv = as_ohlcv(pair_df)
    result = compute_trade_excursions_np(
        ohlcv,
        start_index,
        entry,
        stop_loss,
        take_profit,
        max_holding_bars=max_holding_bars,
        direction=direction,
    )
    return result


def enrich_setup_with_outcome(
    setup: ReversalSetup,
    pair_df: pd.DataFrame,
    m15_df: pd.DataFrame | None = None,
    *,
    h1_df: pd.DataFrame | None = None,
) -> ReversalSetup:
    ws_sim: WyckoffSpringSimResult | None = None
    if _ws_pyramid_enabled() and m15_df is not None:
        ws_sim = simulate_wyckoff_pyramid(
            setup,
            h1_df=h1_df if h1_df is not None else pair_df,
            m15_df=m15_df,
        )
        excursions = {
            "outcome_label": ws_sim.result,
            "result_r": ws_sim.profit_r,
            "mfe": ws_sim.mfe,
            "mae": ws_sim.mae,
        }
    else:
        excursions = compute_wyckoff_trade_excursions(
            pair_df,
            setup.recovery_bar_index,
            setup.entry_price,
            setup.stop_loss,
            setup.take_profit,
            direction=setup.direction,
        )
    base = setup.wyckoff_features.as_dict()
    base.update(excursions)
    features = WyckoffFeatures(**{k: base[k] for k in WyckoffFeatures.__dataclass_fields__})
    return ReversalSetup(
        timestamp=setup.timestamp,
        pair=setup.pair,
        accumulation=setup.accumulation,
        spring_depth_atr=setup.spring_depth_atr,
        spring_velocity=setup.spring_velocity,
        spring_duration_bars=setup.spring_duration_bars,
        support_penetration_percent=setup.support_penetration_percent,
        spring_attempt_number=setup.spring_attempt_number,
        volume_on_spring=setup.volume_on_spring,
        entry_price=setup.entry_price,
        stop_loss=setup.stop_loss,
        take_profit=setup.take_profit,
        risk_reward=setup.risk_reward,
        spring_bar_index=setup.spring_bar_index,
        recovery_bar_index=setup.recovery_bar_index,
        wyckoff_features=features,
        reason_codes=setup.reason_codes,
        direction=setup.direction,
        reversal_pattern=setup.reversal_pattern,
        upthrust_height_atr=setup.upthrust_height_atr,
        candidate_score=setup.candidate_score,
        ws_sim=ws_sim,
    )


def _detect_reversal_at_bar(
    exec_clipped: pd.DataFrame,
    accumulation: AccumulationRange,
    pip: float,
    *,
    pair: str,
    bar_index: int,
    h1_clipped: pd.DataFrame,
) -> ReversalSetup | None:
    """同一バーで Spring を優先し、無ければ Upthrust を評価。"""
    spring = detect_spring(
        exec_clipped,
        accumulation,
        pip,
        pair=pair,
        spring_bar_index=bar_index,
        h1_df=h1_clipped,
    )
    if spring is not None:
        return spring
    return detect_upthrust(
        exec_clipped,
        accumulation,
        pip,
        pair=pair,
        upthrust_bar_index=bar_index,
        h1_df=h1_clipped,
    )


def detect_wyckoff_reversal_setups_for_pair(
    df: pd.DataFrame,
    pair: str,
    *,
    m15_df: pd.DataFrame | None = None,
    lookback_bars: int = LOOKBACK_BARS,
    max_setups_per_day: int = MAX_SETUPS_PER_DAY,
    include_outcomes: bool = False,
    progress_hook: Callable[[int], None] | None = None,
    resume_from_bar: int | None = None,
    initial_setups: list[ReversalSetup] | None = None,
    on_checkpoint: Callable[[int, list[ReversalSetup], dict[str, Any] | None], None] | None = None,
    checkpoint_every: int = 0,
) -> list[ReversalSetup]:
    """Walk-forward スキャン — H1 で Trading Range、M15 で Spring / Upthrust（numpy hot path）。"""
    if pair.upper() not in ALLOWED_PAIRS:
        return []

    from strategies.bt_ohlcv import as_ohlcv
    from strategies.archive.wyckoff_scan_hot import detect_wyckoff_reversal_setups_np

    h1_arr = as_ohlcv(df)
    exec_arr = as_ohlcv(m15_df) if m15_df is not None else h1_arr
    return detect_wyckoff_reversal_setups_np(
        h1_arr,
        exec_arr,
        pair,
        lookback_bars=lookback_bars,
        max_setups_per_day=max_setups_per_day,
        include_outcomes=include_outcomes,
        progress_hook=progress_hook,
        resume_from_bar=resume_from_bar,
        initial_setups=initial_setups,
        on_checkpoint=on_checkpoint,
        checkpoint_every=checkpoint_every,
    )


detect_wyckoff_spring_setups_for_pair = detect_wyckoff_reversal_setups_for_pair


def build_wyckoff_l6_log_row(
    *,
    trade_id: str,
    setup: ReversalSetup,
    decision_source: str,
    bayes_probability: float,
    reason_codes: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """L6 CSV 行 — wyckoff_features を JSON 文字列で格納。"""
    codes = list(reason_codes or setup.reason_codes)
    return {
        "trade_id": trade_id,
        "timestamp": setup.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "pair": setup.pair,
        "setup_type": SETUP_TYPE,
        "entry_price": setup.entry_price,
        "stop_loss": setup.stop_loss,
        "take_profit": setup.take_profit,
        "decision_source": decision_source,
        "bayes_probability": round(bayes_probability, 4),
        "reason_codes": "|".join(codes),
        "wyckoff_features": setup.wyckoff_features.to_json(),
        "ws_kalman_velocity": round(
            setup.ws_sim.kalman_velocity_at_entry if setup.ws_sim is not None else setup.wyckoff_features.kalman_velocity_at_entry,
            8,
        ),
        "ws_decel_exit": setup.ws_sim.kalman_decel_exit_triggered if setup.ws_sim is not None else False,
        "ws_time_limit_exit": setup.ws_sim.time_limit_exit_triggered if setup.ws_sim is not None else False,
        "ws_pyramid_rejected_reason": setup.ws_sim.pyramid_rejected_reason if setup.ws_sim is not None else "",
    }


class WyckoffReversalStrategy(BaseStrategy):
    """WR — H1 監視 + M15 Spring (BUY) / Upthrust (SELL) トリガー。"""

    @property
    def setup_type(self) -> str:
        return SETUP_TYPE

    def detect_setups(
        self,
        df: pd.DataFrame,
        pair_name: str,
        h1_df: pd.DataFrame | None = None,
    ) -> list[ReversalSetup]:
        if h1_df is not None:
            return detect_wyckoff_reversal_setups_for_pair(h1_df, pair_name, m15_df=df)
        return detect_wyckoff_reversal_setups_for_pair(df, pair_name)

    def analyze_setup(
        self,
        setup: ReversalSetup,
        gbp_setup: ReversalSetup | None,
        eur_setup: ReversalSetup | None,
        h1_gbp: pd.DataFrame,
        h1_eur: pd.DataFrame,
    ) -> StrategyResult:
        from audit.wyckoff_bayes_gate import resolve_wyckoff_bayes_decision

        h1_ref = h1_gbp if setup.pair == WYCKOFF_PAIR_PRIMARY else h1_eur
        htf = analyze_htf_trend(h1_ref, setup.timestamp)

        def _pattern_size(s: ReversalSetup) -> float:
            if s.reversal_pattern == "UPTHRUST":
                return s.upthrust_height_atr or s.wyckoff_features.upthrust_height
            return s.wyckoff_features.spring_depth_atr

        if gbp_setup is not None and eur_setup is not None:
            gbp_size = _pattern_size(gbp_setup) * 10.0
            eur_size = _pattern_size(eur_setup) * 10.0
            smt_diff = gbp_size - eur_size
            smt_intensity = abs(smt_diff)
            if gbp_size > eur_size:
                smt_leader = "GBP"
            elif eur_size > gbp_size:
                smt_leader = "EUR"
            else:
                smt_leader = "NONE"
        else:
            smt_diff = 0.0
            smt_intensity = 0.0
            smt_leader = "NONE"

        score = min(
            100.0,
            max(
                50.0,
                50.0
                + setup.wyckoff_features.spring_recovery_atr * 15.0
                + min(setup.risk_reward, 3.0) * 5.0,
            ),
        )
        decision = resolve_wyckoff_bayes_decision(1.0)
        raw: dict[str, Any] = {
            **setup.wyckoff_features.as_dict(),
            "wr_mode": wr_mode_from_direction(setup.direction),
            "smt_intensity": round(smt_intensity, 4),
            "smt_diff": round(smt_diff, 4),
            "smt_leader": smt_leader,
            "atr_ratio": round(setup.wyckoff_features.range_width_atr, 4),
            "has_bos": False,
            "both_sweep": gbp_setup is not None and eur_setup is not None,
            "htf_trend_direction": htf.direction,
            "htf_would_block": False,
            "reject_reason": "",
            "candidate_score": score,
            "reversal_pattern": setup.reversal_pattern,
        }
        return StrategyResult(
            is_setup=True,
            setup_type=self.setup_type,
            direction=setup.direction,
            strategy_action=decision,
            entry_price=setup.entry_price,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            candidate_score=score,
            raw_features=raw,
        )


WyckoffSpringStrategy = WyckoffReversalStrategy


__all__ = [
    "AccumulationRange",
    "ReversalSetup",
    "SpringSetup",
    "WyckoffFeatures",
    "WyckoffReversalSimResult",
    "WyckoffSpringSimResult",
    "WyckoffReversalStrategy",
    "WyckoffSpringStrategy",
    "WS_PYRAMID_ENABLED",
    "is_wyckoff_pure_bt_mode",
    "simulate_wyckoff_pyramid",
    "WYCKOFF_FEATURE_COLUMNS",
    "STRATEGY_ABBREV",
    "STRATEGY_FULL_NAME",
    "STRATEGY_ID",
    "SETUP_TYPE",
    "SETUP_TYPE_LEGACY",
    "MAX_SETUPS_PER_DAY",
    "WYCKOFF_EXEC_BAR_MINUTES",
    "WYCKOFF_MONITOR_BAR_MINUTES",
    "WYCKOFF_SPRING_BAR_MINUTES",
    "SC_VOLUME_ZSCORE_MIN",
    "detect_accumulation_range",
    "detect_spring",
    "detect_upthrust",
    "detect_wyckoff_reversal_setups_for_pair",
    "detect_wyckoff_spring_setups_for_pair",
    "build_wyckoff_features",
    "build_wyckoff_l6_log_row",
    "compute_recovery_close_ratio",
    "wr_mode_from_direction",
    "compute_wyckoff_trade_excursions",
    "enrich_setup_with_outcome",
    "classify_wyckoff_macro_phase",
]
