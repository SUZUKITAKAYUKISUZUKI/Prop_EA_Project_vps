"""
strategies/archive/tokyo_range_expansion_failure.py — Tokyo Range Expansion Failure (TREF)

ARCHIVED 2026-06: 単独でのプロップ失格率が高いため本番・標準 BT から外した。参照・再検討用。

USDJPY / AUDJPY の東京時間（JST）レンジ・エクスパンション失敗を検出し、
決定論的 candidate_score（最大 100 点）と Gemini 下流用 JSON Payload を生成する。

タイムフレーム:
  - HTF: H1（ATR / 構造）
  - Anchor: M15（東京基準レンジ 09:00–11:30 JST）
  - ETF: M5（11:30–15:00 JST 執行監視）

M5 のみ入力の場合は resample で M15/H1 を内部生成（前方参照バイアス排除）。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import time
from typing import Any, Callable, Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

TrefSliceBand = Literal["range", "exec", "session"]

from strategies.base import StrategyResult
from strategies.base_strategy import BaseStrategy
from strategies.htf_trend_analyzer import analyze_htf_trend
from strategies.market_utils import (
    JPY_PIP_SIZE,
    SMTFeatures,
    calc_smt_features,
    compute_atr,
    pip_size_for_pair,
    uses_primary_dataframe,
)

logger = logging.getLogger(__name__)

SETUP_TYPE = "TOKYO_RANGE_EXPANSION_FAILURE"
TREF_PAIR_PRIMARY = "AUDJPY"
TREF_PAIR_SECONDARY = "USDJPY"
ALLOWED_PAIRS = frozenset({TREF_PAIR_PRIMARY, TREF_PAIR_SECONDARY})

JST = ZoneInfo("Asia/Tokyo")
DEFAULT_INPUT_TZ = ZoneInfo(os.getenv("TREF_INPUT_TZ", "UTC"))

RANGE_START_JST = time(9, 0)
RANGE_END_JST = time(11, 30)
EXEC_START_JST = time(11, 30)
EXEC_END_JST = time(15, 0)

ATR_PERIOD = 20
MAX_BARS_OUTSIDE_M5 = 3  # reject when >= 4
OVER_EXPANSION_ATR_MULT = 1.2
SPREAD_REJECT_MULT = 2.0

Direction = Literal["BUY", "SELL"]
RejectReason = Literal[
    "REJECT_TREF_PAIR",
    "REJECT_TREF_NO_RANGE",
    "REJECT_TREF_TIME_WINDOW",
    "REJECT_TREF_TREND_OUTSIDE",
    "REJECT_TREF_OVER_EXPANSION",
    "REJECT_TREF_SPREAD",
    "REJECT_TREF_NO_TRIGGER",
    "REJECT_TREF_INSUFFICIENT_DATA",
]


# ---------------------------------------------------------------------------
# Pydantic payload models (Gemini downstream)
# ---------------------------------------------------------------------------
class RangeMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokyo_range_width_pips: float
    h1_atr_20_pips: float
    ratio_range_to_htf_atr: float


class ExpansionMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expansion_depth_pips: float
    m15_atr_20_pips: float
    ratio_depth_to_anchor_atr: float
    bars_stayed_outside_m5: int


class ExecutionMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trigger_time_jst: str
    trigger_bar_wick_ratio_pct: float
    re_entry_depth_pct: float


class MarketContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    range_metrics: RangeMetrics
    expansion_metrics: ExpansionMetrics
    execution_metrics: ExecutionMetrics


class TrefPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: str = Field(default=SETUP_TYPE)
    pair: str
    timestamp: str
    direction: Direction
    candidate_score: int
    market_context: MarketContext


@dataclass(frozen=True)
class TrefConfig:
    input_tz: ZoneInfo = DEFAULT_INPUT_TZ
    atr_period: int = ATR_PERIOD
    max_bars_outside_m5: int = MAX_BARS_OUTSIDE_M5
    over_expansion_atr_mult: float = OVER_EXPANSION_ATR_MULT
    spread_reject_mult: float = SPREAD_REJECT_MULT
    sl_buffer_pips: float = 2.0
    min_rr: float = 1.5
    l4_bypass: bool = True
    max_setups_per_day: int = 1


def load_tref_config() -> TrefConfig:
    tz_name = os.getenv("TREF_INPUT_TZ", "UTC").strip()
    try:
        input_tz = ZoneInfo(tz_name)
    except Exception:
        logger.warning("Invalid TREF_INPUT_TZ=%s; falling back to UTC", tz_name)
        input_tz = ZoneInfo("UTC")
    return TrefConfig(
        input_tz=input_tz,
        atr_period=int(os.getenv("TREF_ATR_PERIOD", str(ATR_PERIOD))),
        max_bars_outside_m5=int(os.getenv("TREF_MAX_BARS_OUTSIDE", str(MAX_BARS_OUTSIDE_M5))),
        over_expansion_atr_mult=float(os.getenv("TREF_OVER_EXPANSION_ATR_MULT", "1.2")),
        spread_reject_mult=float(os.getenv("TREF_SPREAD_REJECT_MULT", "2.0")),
        sl_buffer_pips=float(os.getenv("TREF_SL_BUFFER_PIPS", "2.0")),
        min_rr=float(os.getenv("TREF_MIN_RR", "1.5")),
        l4_bypass=os.getenv("TREF_L4_BYPASS", "1").strip().lower()
        not in ("0", "false", "no", "off"),
        max_setups_per_day=int(os.getenv("TREF_MAX_SETUPS_PER_DAY", "1")),
    )


@dataclass
class TrefSetup:
    timestamp: pd.Timestamp
    pair: str
    direction: Direction
    range_high: float
    range_low: float
    range_width_pips: float
    expansion_depth_pips: float
    bars_stayed_outside_m5: int
    entry_price: float
    stop_loss: float
    take_profit: float
    sweep_distance: float
    bar_index: int
    candidate_score: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)
    score_breakdown: dict[str, int] = field(default_factory=dict)
    reject_reason: str | None = None


@dataclass
class TrefEvaluationResult:
    payload: TrefPayload | None
    candidate_score: float
    rejected: bool
    reject_reason: RejectReason | None
    setup: TrefSetup | None
    score_breakdown: dict[str, int] = field(default_factory=dict)


def _normalize_pair(pair_name: str) -> str:
    upper = pair_name.upper().replace(".", "").replace("_", "").replace("-", "")
    if "AUDJPY" in upper:
        return "AUDJPY"
    if "USDJPY" in upper:
        return "USDJPY"
    return upper


def _ensure_bars(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    work = df.sort_values("datetime").copy()
    work["datetime"] = pd.to_datetime(work["datetime"])
    if "volume" not in work.columns:
        work["volume"] = 1.0
    else:
        work["volume"] = work["volume"].fillna(0.0).clip(lower=0.0)
    return work.reset_index(drop=True)


def clip_as_of(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Forward-looking bias exclusion: keep bars with datetime <= as_of."""
    if df.empty:
        return df
    ts = pd.Timestamp(as_of)
    return df.loc[df["datetime"] <= ts].reset_index(drop=True)


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV without lookahead (label/closed=left)."""
    work = _ensure_bars(df)
    if work.empty:
        return work
    indexed = work.set_index("datetime")
    resampled = indexed.resample(rule, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return resampled.dropna(subset=["open"]).reset_index()


def to_jst(ts: pd.Timestamp, input_tz: ZoneInfo = DEFAULT_INPUT_TZ) -> pd.Timestamp:
    """Convert timestamp to JST (aware). Naive timestamps localize to input_tz."""
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(input_tz)
    return stamp.tz_convert(JST)


def _jst_time_of(ts: pd.Timestamp, input_tz: ZoneInfo) -> time:
    return to_jst(ts, input_tz).time()


def _time_in_window(t: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= t <= end
    return t >= start or t <= end


def _atr_pips_at(
    df: pd.DataFrame,
    as_of: pd.Timestamp,
    pip_size: float,
    period: int = ATR_PERIOD,
    atr_series: pd.Series | None = None,
) -> float:
    work = _ensure_bars(df)
    clipped = clip_as_of(work, as_of)
    if len(clipped) < period:
        return 0.0
    if atr_series is not None and len(atr_series) == len(work):
        pos = int(clipped.index[-1])
        if pos >= len(atr_series) or pd.isna(atr_series.iloc[pos]):
            return 0.0
        value = float(atr_series.iloc[pos])
        return value / pip_size if value > 0 else 0.0
    atr = compute_atr(clipped, period=period)
    valid = atr.dropna()
    if valid.empty:
        return 0.0
    value = float(valid.iloc[-1])
    return value / pip_size if value > 0 else 0.0


def _jst_minutes(ts: pd.Timestamp, input_tz: ZoneInfo) -> int:
    jst = to_jst(ts, input_tz)
    return int(jst.hour) * 60 + int(jst.minute)


def filter_tref_bars(
    df: pd.DataFrame,
    band: TrefSliceBand,
    *,
    config: TrefConfig | None = None,
) -> pd.DataFrame:
    """
    JST ベースで TREF 関連時間帯のみ残す（BT precompute / シミュレーション時短用）。

    - range: 09:00–11:15 JST（M15 東京レンジ 10 本）
    - exec:  11:30–15:00 JST（M5 執行監視）
    - session: 09:00–15:00 JST（レンジ + 執行）
    """
    cfg = config or load_tref_config()
    work = _ensure_bars(df)
    if work.empty:
        return work

    dts = pd.to_datetime(work["datetime"])
    if dts.dt.tz is None:
        dts = dts.dt.tz_localize(cfg.input_tz)
    jst = dts.dt.tz_convert(JST)
    minutes = jst.dt.hour.to_numpy(dtype=np.int32) * 60 + jst.dt.minute.to_numpy(dtype=np.int32)

    range_start = 9 * 60
    range_end = 11 * 60 + 15
    exec_start = 11 * 60 + 30
    exec_end = 15 * 60

    if band == "range":
        mask = (minutes >= range_start) & (minutes <= range_end)
    elif band == "exec":
        mask = (minutes >= exec_start) & (minutes <= exec_end)
    else:
        mask = (minutes >= range_start) & (minutes <= exec_end)

    sliced = work.loc[mask].reset_index(drop=True)
    logger.debug(
        "TREF slice band=%s %d -> %d rows (%.1f%% kept)",
        band,
        len(work),
        len(sliced),
        100.0 * len(sliced) / max(len(work), 1),
    )
    return sliced


def compute_tokyo_range(
    m15_df: pd.DataFrame,
    session_date: pd.Timestamp,
    input_tz: ZoneInfo,
    pip_size: float,
) -> tuple[float, float, float] | None:
    """
    09:00–11:30 JST の M15 10 本から Range_High / Range_Low を確定。
    session_date は JST カレンダー日。
    """
    work = filter_tref_bars(m15_df, "range", config=TrefConfig(input_tz=input_tz))
    if work.empty:
        return None

    session_day = session_date.date()
    jst_dates = work["datetime"].apply(lambda ts: to_jst(ts, input_tz).date())
    day = work.loc[jst_dates == session_day]
    if len(day) < 10:
        logger.debug(
            "TREF range incomplete for %s: %d/10 M15 bars in 09:00–11:30 JST",
            session_day,
            len(day),
        )
        return None

    range_high = float(day["high"].max())
    range_low = float(day["low"].min())
    if range_high <= range_low:
        return None
    width_pips = (range_high - range_low) / pip_size
    logger.debug(
        "TREF range %s: high=%.3f low=%.3f width=%.2f pips (bars=%d)",
        session_day,
        range_high,
        range_low,
        width_pips,
        len(day),
    )
    return range_high, range_low, width_pips


def _close_outside(close: float, range_high: float, range_low: float) -> bool:
    return close > range_high or close < range_low


def _expansion_direction(
    max_outside_high: float,
    min_outside_low: float,
    range_high: float,
    range_low: float,
) -> Direction | None:
    above = max_outside_high > range_high
    below = min_outside_low < range_low
    if above and not below:
        return "SELL"
    if below and not above:
        return "BUY"
    if above and below:
        above_depth = max_outside_high - range_high
        below_depth = range_low - min_outside_low
        return "SELL" if above_depth >= below_depth else "BUY"
    return None


def _expansion_depth_pips(
    direction: Direction,
    range_high: float,
    range_low: float,
    max_outside_high: float,
    min_outside_low: float,
    pip_size: float,
) -> float:
    if direction == "SELL":
        return max(0.0, (max_outside_high - range_high) / pip_size)
    return max(0.0, (range_low - min_outside_low) / pip_size)


def _re_entry_depth_pct(
    direction: Direction,
    close: float,
    range_high: float,
    range_low: float,
) -> float:
    width = range_high - range_low
    if width <= 0:
        return 0.0
    if direction == "BUY":
        depth = close - range_low
    else:
        depth = range_high - close
    return max(0.0, min(100.0, depth / width * 100.0))


def _trigger_wick_ratio_pct(bar: pd.Series, direction: Direction) -> float:
    high_p = float(bar["high"])
    low_p = float(bar["low"])
    open_p = float(bar["open"])
    close_p = float(bar["close"])
    total = high_p - low_p
    if total <= 0:
        return 0.0
    if direction == "BUY":
        wick = min(open_p, close_p) - low_p
    else:
        wick = high_p - max(open_p, close_p)
    wick = max(0.0, wick)
    return min(100.0, wick / total * 100.0)


def score_axis1_range_maturity(ratio_range: float) -> int:
    if 0.6 <= ratio_range <= 1.2:
        points = 25
    elif 0.4 <= ratio_range < 0.6:
        points = 15
    elif 1.2 < ratio_range <= 1.5:
        points = 10
    else:
        points = 0
    logger.debug("TREF axis1 range maturity ratio=%.3f -> %d pts", ratio_range, points)
    return points


def score_axis2_expansion_depth(ratio_depth: float) -> int:
    if 0.2 <= ratio_depth <= 0.6:
        points = 30
    elif 0.6 < ratio_depth <= 1.0:
        points = 15
    elif 0.05 <= ratio_depth < 0.2:
        points = 5
    else:
        points = 0
    logger.debug("TREF axis2 expansion depth ratio=%.3f -> %d pts", ratio_depth, points)
    return points


def score_axis3_time_sync(trigger_jst: pd.Timestamp) -> int:
    t = trigger_jst.time()
    if time(12, 30) <= t <= time(14, 0):
        points = 20
    elif time(11, 30) <= t <= time(12, 29):
        points = 10
    elif time(14, 1) <= t <= time(15, 0):
        points = 5
    else:
        points = 0
    logger.debug("TREF axis3 time sync %s -> %d pts", t.strftime("%H:%M"), points)
    return points


def score_axis4_price_action(re_entry_pct: float, wick_ratio_pct: float) -> tuple[int, int, int]:
    if re_entry_pct >= 20.0:
        reentry_pts = 15
    elif re_entry_pct >= 0.0:
        reentry_pts = 5
    else:
        reentry_pts = 0

    if wick_ratio_pct >= 60.0:
        wick_pts = 10
    elif wick_ratio_pct >= 40.0:
        wick_pts = 5
    else:
        wick_pts = 0

    total = reentry_pts + wick_pts
    logger.debug(
        "TREF axis4 re_entry=%.1f%% -> %d pts, wick=%.1f%% -> %d pts (total=%d)",
        re_entry_pct,
        reentry_pts,
        wick_ratio_pct,
        wick_pts,
        total,
    )
    return reentry_pts, wick_pts, total


def _compute_sl_tp(
    direction: Direction,
    entry: float,
    range_high: float,
    range_low: float,
    max_outside_high: float,
    min_outside_low: float,
    pip_size: float,
    config: TrefConfig,
) -> tuple[float, float]:
    buffer = config.sl_buffer_pips * pip_size
    if direction == "BUY":
        stop_loss = min_outside_low - buffer
        risk = entry - stop_loss
        take_profit = entry + config.min_rr * risk
        take_profit = min(take_profit, range_high)
    else:
        stop_loss = max_outside_high + buffer
        risk = stop_loss - entry
        take_profit = entry - config.min_rr * risk
        take_profit = max(take_profit, range_low)
    return stop_loss, take_profit


def _spread_at_trigger(
    spread_series: pd.Series | None,
    trigger_ts: pd.Timestamp,
    m5_df: pd.DataFrame,
) -> float | None:
    if spread_series is None:
        return None
    if trigger_ts in spread_series.index:
        return float(spread_series.loc[trigger_ts])
    idx = m5_df.index[m5_df["datetime"] == trigger_ts]
    if len(idx) == 0:
        return None
    return float(spread_series.iloc[int(idx[0])]) if len(spread_series) > int(idx[0]) else None


def _avg_spread_24h(
    spread_series: pd.Series | None,
    m5_df: pd.DataFrame,
    trigger_ts: pd.Timestamp,
    input_tz: ZoneInfo,
) -> float | None:
    if spread_series is None or spread_series.empty:
        return None
    end = trigger_ts
    start = end - pd.Timedelta(hours=24)
    mask = (m5_df["datetime"] >= start) & (m5_df["datetime"] <= end)
    if not mask.any():
        return None
    indices = m5_df.index[mask]
    try:
        values = spread_series.iloc[indices].astype(float)
    except (IndexError, KeyError, TypeError):
        return None
    valid = values.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return None
    return float(valid.mean())


def evaluate_trigger(
    *,
    pair: str,
    trigger_bar: pd.Series,
    trigger_index: int,
    range_high: float,
    range_low: float,
    range_width_pips: float,
    bars_stayed_outside: int,
    max_outside_high: float,
    min_outside_low: float,
    h1_df: pd.DataFrame,
    m15_df: pd.DataFrame,
    pip_size: float,
    config: TrefConfig | None = None,
    spread_series: pd.Series | None = None,
    m5_df: pd.DataFrame | None = None,
    h1_atr_series: pd.Series | None = None,
    m15_atr_series: pd.Series | None = None,
) -> TrefEvaluationResult:
    """Score a single expansion-failure trigger bar."""
    config = config or load_tref_config()
    pair_norm = _normalize_pair(pair)
    if pair_norm not in ALLOWED_PAIRS:
        return TrefEvaluationResult(None, 0.0, True, "REJECT_TREF_PAIR", None)

    trigger_ts = pd.Timestamp(trigger_bar["datetime"])
    trigger_jst = to_jst(trigger_ts, config.input_tz)
    trigger_time = trigger_jst.time()

    if not _time_in_window(trigger_time, EXEC_START_JST, EXEC_END_JST):
        logger.debug("TREF hard reject: trigger time %s outside exec window", trigger_time)
        return TrefEvaluationResult(None, 0.0, True, "REJECT_TREF_TIME_WINDOW", None)

    if bars_stayed_outside > config.max_bars_outside_m5:
        logger.debug(
            "TREF hard reject: bars outside=%d > max=%d",
            bars_stayed_outside,
            config.max_bars_outside_m5,
        )
        return TrefEvaluationResult(None, 0.0, True, "REJECT_TREF_TREND_OUTSIDE", None)

    direction = _expansion_direction(max_outside_high, min_outside_low, range_high, range_low)
    if direction is None:
        return TrefEvaluationResult(None, 0.0, True, "REJECT_TREF_NO_TRIGGER", None)

    expansion_depth_pips = _expansion_depth_pips(
        direction, range_high, range_low, max_outside_high, min_outside_low, pip_size
    )
    m15_atr_pips = _atr_pips_at(
        m15_df, trigger_ts, pip_size, config.atr_period, atr_series=m15_atr_series
    )
    if m15_atr_pips > 0 and expansion_depth_pips > m15_atr_pips * config.over_expansion_atr_mult:
        logger.debug(
            "TREF hard reject: depth=%.2f > M15_ATR*%.1f (%.2f)",
            expansion_depth_pips,
            config.over_expansion_atr_mult,
            m15_atr_pips * config.over_expansion_atr_mult,
        )
        return TrefEvaluationResult(None, 0.0, True, "REJECT_TREF_OVER_EXPANSION", None)

    if spread_series is not None and m5_df is not None:
        current_spread = _spread_at_trigger(spread_series, trigger_ts, m5_df)
        avg_spread = _avg_spread_24h(spread_series, m5_df, trigger_ts, config.input_tz)
        if current_spread is not None and avg_spread is not None and avg_spread > 0:
            if current_spread >= avg_spread * config.spread_reject_mult:
                logger.debug(
                    "TREF hard reject: spread %.2f >= 2x avg %.2f",
                    current_spread,
                    avg_spread,
                )
                return TrefEvaluationResult(None, 0.0, True, "REJECT_TREF_SPREAD", None)

    h1_atr_pips = _atr_pips_at(
        h1_df, trigger_ts, pip_size, config.atr_period, atr_series=h1_atr_series
    )
    ratio_range = range_width_pips / h1_atr_pips if h1_atr_pips > 0 else 0.0
    ratio_depth = expansion_depth_pips / m15_atr_pips if m15_atr_pips > 0 else 0.0
    re_entry_pct = _re_entry_depth_pct(direction, float(trigger_bar["close"]), range_high, range_low)
    wick_ratio_pct = _trigger_wick_ratio_pct(trigger_bar, direction)

    axis1 = score_axis1_range_maturity(ratio_range)
    axis2 = score_axis2_expansion_depth(ratio_depth)
    axis3 = score_axis3_time_sync(trigger_jst)
    reentry_pts, wick_pts, axis4 = score_axis4_price_action(re_entry_pct, wick_ratio_pct)
    total_score = axis1 + axis2 + axis3 + axis4

    score_breakdown = {
        "axis1_range_maturity": axis1,
        "axis2_expansion_depth": axis2,
        "axis3_time_sync": axis3,
        "axis4_reentry": reentry_pts,
        "axis4_wick": wick_pts,
        "axis4_total": axis4,
        "total": total_score,
    }
    logger.debug(
        "TREF score %s %s @ %s: %d (a1=%d a2=%d a3=%d a4=%d)",
        pair_norm,
        direction,
        trigger_jst.strftime("%Y-%m-%d %H:%M"),
        total_score,
        axis1,
        axis2,
        axis3,
        axis4,
    )

    entry = float(trigger_bar["close"])
    stop_loss, take_profit = _compute_sl_tp(
        direction,
        entry,
        range_high,
        range_low,
        max_outside_high,
        min_outside_low,
        pip_size,
        config,
    )
    sweep_distance = expansion_depth_pips * pip_size

    payload = TrefPayload(
        pair=pair_norm,
        timestamp=trigger_jst.strftime("%Y-%m-%d %H:%M:%S"),
        direction=direction,
        candidate_score=int(total_score),
        market_context=MarketContext(
            range_metrics=RangeMetrics(
                tokyo_range_width_pips=round(range_width_pips, 2),
                h1_atr_20_pips=round(h1_atr_pips, 2),
                ratio_range_to_htf_atr=round(ratio_range, 3),
            ),
            expansion_metrics=ExpansionMetrics(
                expansion_depth_pips=round(expansion_depth_pips, 2),
                m15_atr_20_pips=round(m15_atr_pips, 2),
                ratio_depth_to_anchor_atr=round(ratio_depth, 3),
                bars_stayed_outside_m5=bars_stayed_outside,
            ),
            execution_metrics=ExecutionMetrics(
                trigger_time_jst=trigger_jst.strftime("%H:%M"),
                trigger_bar_wick_ratio_pct=round(wick_ratio_pct, 1),
                re_entry_depth_pct=round(re_entry_pct, 1),
            ),
        ),
    )

    setup = TrefSetup(
        timestamp=trigger_ts,
        pair=pair_norm,
        direction=direction,
        range_high=range_high,
        range_low=range_low,
        range_width_pips=range_width_pips,
        expansion_depth_pips=expansion_depth_pips,
        bars_stayed_outside_m5=bars_stayed_outside,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        sweep_distance=sweep_distance,
        bar_index=trigger_index,
        candidate_score=float(total_score),
        payload=payload.model_dump(),
        score_breakdown=score_breakdown,
    )

    return TrefEvaluationResult(
        payload=payload,
        candidate_score=float(total_score),
        rejected=False,
        reject_reason=None,
        setup=setup,
        score_breakdown=score_breakdown,
    )


def scan_day_triggers(
    m5_df: pd.DataFrame,
    m15_df: pd.DataFrame,
    session_date: pd.Timestamp,
    pair: str,
    input_tz: ZoneInfo,
    pip_size: float,
    *,
    range_high: float | None = None,
    range_low: float | None = None,
) -> list[tuple[pd.Series, int, int, float, float]]:
    """
    Scan one JST session day for expansion-failure triggers on M5.

    Returns list of (trigger_bar, index, bars_outside, max_high, min_low).
    """
    if range_high is None or range_low is None:
        range_vals = compute_tokyo_range(m15_df, session_date, input_tz, pip_size)
        if range_vals is None:
            return []
        range_high, range_low, _ = range_vals

    session_day = session_date.date()
    results: list[tuple[pd.Series, int, int, float, float]] = []
    outside = False
    bars_outside = 0
    max_outside_high = range_high
    min_outside_low = range_low

    for idx, row in m5_df.iterrows():
        bar_time = to_jst(row["datetime"], input_tz).time()
        if not _time_in_window(bar_time, EXEC_START_JST, EXEC_END_JST):
            continue

        close_p = float(row["close"])
        high_p = float(row["high"])
        low_p = float(row["low"])

        if not outside:
            if _close_outside(close_p, range_high, range_low):
                outside = True
                bars_outside = 1
                max_outside_high = max(range_high, high_p)
                min_outside_low = min(range_low, low_p)
            continue

        if _close_outside(close_p, range_high, range_low):
            bars_outside += 1
            max_outside_high = max(max_outside_high, high_p)
            min_outside_low = min(min_outside_low, low_p)
            continue

        results.append((row, idx, bars_outside, max_outside_high, min_outside_low))
        outside = False
        bars_outside = 0
        max_outside_high = range_high
        min_outside_low = range_low

    return results


def detect_tref_setups(
    df: pd.DataFrame,
    pair_name: str,
    h1_df: pd.DataFrame | None = None,
    config: TrefConfig | None = None,
    progress_hook: Callable[[int, int], None] | None = None,
    *,
    m15_df: pd.DataFrame | None = None,
) -> list[TrefSetup]:
    """BT / precompute 用ラッパー（ALS detect_* と同一シグネチャ）。"""
    strategy = TokyoRangeExpansionFailure(config=config or load_tref_config())
    m5, m15, _h1 = strategy._prepare_frames(df, m15_df, h1_df)
    scan_total = max(len(m5), 1)
    setups = strategy.detect_setups(
        df,
        pair_name,
        h1_df=h1_df,
        m15_df=m15,
    )
    if progress_hook is not None:
        progress_hook(scan_total, scan_total)
    return setups


class TokyoRangeExpansionFailure(BaseStrategy):
    """TREF strategy — USDJPY / AUDJPY Tokyo range expansion failure."""

    def __init__(
        self,
        weights_config: dict[str, int] | None = None,
        mode_h1: bool = False,
        config: TrefConfig | None = None,
    ):
        super().__init__(weights_config=weights_config, mode_h1=mode_h1)
        self.tref_config = config or load_tref_config()
        self._htf_gbp: pd.DataFrame = pd.DataFrame()
        self._htf_eur: pd.DataFrame = pd.DataFrame()

    @property
    def setup_type(self) -> str:
        return SETUP_TYPE

    def _prepare_frames(
        self,
        m5_df: pd.DataFrame,
        m15_df: pd.DataFrame | None,
        h1_df: pd.DataFrame | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        m5 = _ensure_bars(m5_df)
        m15 = _ensure_bars(m15_df) if m15_df is not None and not m15_df.empty else resample_ohlcv(m5, "15min")
        h1 = _ensure_bars(h1_df) if h1_df is not None and not h1_df.empty else resample_ohlcv(m5, "1h")
        return m5, m15, h1

    def detect_setups(
        self,
        df: pd.DataFrame,
        pair_name: str,
        h1_df: pd.DataFrame | None = None,
        *,
        m15_df: pd.DataFrame | None = None,
        spread_series: pd.Series | None = None,
    ) -> list[TrefSetup]:
        pair = _normalize_pair(pair_name)
        if pair not in ALLOWED_PAIRS:
            return []

        pip_size = pip_size_for_pair(pair)
        m5, m15, h1 = self._prepare_frames(df, m15_df, h1_df)
        if m5.empty:
            return []

        m5_exec = filter_tref_bars(m5, "exec", config=self.tref_config)
        m15_range = filter_tref_bars(m15, "range", config=self.tref_config)
        if m5_exec.empty or m15_range.empty:
            return []

        h1_atr = compute_atr(h1, period=self.tref_config.atr_period)
        m15_atr = compute_atr(m15, period=self.tref_config.atr_period)

        m15_range = m15_range.copy()
        m15_jst = pd.to_datetime(m15_range["datetime"])
        if m15_jst.dt.tz is None:
            m15_jst = m15_jst.dt.tz_localize(self.tref_config.input_tz)
        m15_range["_jst_date"] = m15_jst.dt.tz_convert(JST).dt.date

        jst = pd.to_datetime(m5_exec["datetime"])
        if jst.dt.tz is None:
            jst = jst.dt.tz_localize(self.tref_config.input_tz)
        m5_exec = m5_exec.copy()
        m5_exec["_jst_date"] = jst.dt.tz_convert(JST).dt.date
        jst_dates = sorted(m5_exec["_jst_date"].unique())

        setups: list[TrefSetup] = []
        for session_day in jst_dates:
            session_ts = pd.Timestamp(session_day)
            day_m5 = m5_exec.loc[m5_exec["_jst_date"] == session_day].drop(columns=["_jst_date"])
            day_m15 = m15_range.loc[m15_range["_jst_date"] == session_day]
            if len(day_m15) < 10:
                continue
            range_high = float(day_m15["high"].max())
            range_low = float(day_m15["low"].min())
            if range_high <= range_low:
                continue
            range_width_pips = (range_high - range_low) / pip_size
            triggers = scan_day_triggers(
                day_m5,
                m15_range,
                session_ts,
                pair,
                self.tref_config.input_tz,
                pip_size,
                range_high=range_high,
                range_low=range_low,
            )
            if not triggers:
                continue
            bar, idx, bars_outside, max_hi, min_lo = triggers[0]
            if bars_outside <= self.tref_config.max_bars_outside_m5:
                result = evaluate_trigger(
                    pair=pair,
                    trigger_bar=bar,
                    trigger_index=idx,
                    range_high=range_high,
                    range_low=range_low,
                    range_width_pips=range_width_pips,
                    bars_stayed_outside=bars_outside,
                    max_outside_high=max_hi,
                    min_outside_low=min_lo,
                    h1_df=h1,
                    m15_df=m15,
                    pip_size=pip_size,
                    config=self.tref_config,
                    spread_series=spread_series,
                    m5_df=m5,
                    h1_atr_series=h1_atr,
                    m15_atr_series=m15_atr,
                )
                if result.setup is not None and not result.rejected:
                    setups.append(result.setup)

        setups.sort(key=lambda s: s.timestamp)
        return setups

    def evaluate_market_state(
        self,
        pair: str,
        m5_df: pd.DataFrame,
        m15_df: pd.DataFrame | None = None,
        h1_df: pd.DataFrame | None = None,
        *,
        trigger_bar_index: int | None = None,
        spread_series: pd.Series | None = None,
    ) -> TrefEvaluationResult:
        """
        Evaluate the latest (or specified) expansion-failure trigger.

        Primary API for downstream Gemini audit payload generation.
        """
        pair_norm = _normalize_pair(pair)
        if pair_norm not in ALLOWED_PAIRS:
            return TrefEvaluationResult(None, 0.0, True, "REJECT_TREF_PAIR", None)

        pip_size = pip_size_for_pair(pair_norm)
        m5, m15, h1 = self._prepare_frames(m5_df, m15_df, h1_df)
        if m5.empty:
            return TrefEvaluationResult(None, 0.0, True, "REJECT_TREF_INSUFFICIENT_DATA", None)

        if trigger_bar_index is not None:
            if trigger_bar_index < 0 or trigger_bar_index >= len(m5):
                return TrefEvaluationResult(None, 0.0, True, "REJECT_TREF_NO_TRIGGER", None)
            trigger_ts = pd.Timestamp(m5.iloc[trigger_bar_index]["datetime"])
            session_ts = pd.Timestamp(to_jst(trigger_ts, self.tref_config.input_tz).date())
            triggers = scan_day_triggers(m5, m15, session_ts, pair_norm, self.tref_config.input_tz, pip_size)
            match = next((t for t in triggers if t[1] == trigger_bar_index), None)
            if match is None:
                return TrefEvaluationResult(None, 0.0, True, "REJECT_TREF_NO_TRIGGER", None)
            bar, idx, bars_outside, max_hi, min_lo = match
        else:
            all_setups = self.detect_setups(
                m5, pair_norm, h1_df=h1, m15_df=m15, spread_series=spread_series
            )
            if not all_setups:
                return TrefEvaluationResult(None, 0.0, True, "REJECT_TREF_NO_TRIGGER", None)
            latest = all_setups[-1]
            return TrefEvaluationResult(
                payload=TrefPayload.model_validate(latest.payload),
                candidate_score=latest.candidate_score,
                rejected=False,
                reject_reason=None,
                setup=latest,
                score_breakdown=latest.score_breakdown,
            )

        range_vals = compute_tokyo_range(m15, session_ts, self.tref_config.input_tz, pip_size)
        if range_vals is None:
            return TrefEvaluationResult(None, 0.0, True, "REJECT_TREF_NO_RANGE", None)
        range_high, range_low, range_width_pips = range_vals

        return evaluate_trigger(
            pair=pair_norm,
            trigger_bar=bar,
            trigger_index=idx,
            range_high=range_high,
            range_low=range_low,
            range_width_pips=range_width_pips,
            bars_stayed_outside=bars_outside,
            max_outside_high=max_hi,
            min_outside_low=min_lo,
            h1_df=h1,
            m15_df=m15,
            pip_size=pip_size,
            config=self.tref_config,
            spread_series=spread_series,
            m5_df=m5,
        )

    def analyze_setup(
        self,
        setup: TrefSetup,
        gbp_setup: TrefSetup | None,
        eur_setup: TrefSetup | None,
        h1_gbp: pd.DataFrame,
        h1_eur: pd.DataFrame,
    ) -> StrategyResult:
        pip_size = pip_size_for_pair(setup.pair)
        smt: SMTFeatures = calc_smt_features(gbp_setup, eur_setup, pip_size)
        h1_ref = h1_gbp if uses_primary_dataframe(setup.pair) else h1_eur
        htf_df = self._htf_gbp if uses_primary_dataframe(setup.pair) else self._htf_eur
        htf_result = analyze_htf_trend(
            h1_ref,
            setup.timestamp,
            htf_df=htf_df if isinstance(htf_df, pd.DataFrame) and not htf_df.empty else None,
        )
        htf_trend_direction = htf_result.direction
        payload = setup.payload or {}
        mc = payload.get("market_context", {}) if isinstance(payload, dict) else {}
        range_metrics = mc.get("range_metrics", {}) if isinstance(mc, dict) else {}
        expansion_metrics = mc.get("expansion_metrics", {}) if isinstance(mc, dict) else {}
        execution_metrics = mc.get("execution_metrics", {}) if isinstance(mc, dict) else {}
        ratio_range = float(range_metrics.get("ratio_range_to_htf_atr", 0.0) or 0.0)
        ratio_depth = float(expansion_metrics.get("ratio_depth_to_anchor_atr", 0.0) or 0.0)
        wick_ratio_pct = float(execution_metrics.get("trigger_bar_wick_ratio_pct", 0.0) or 0.0)
        l4_bypass = self.tref_config.l4_bypass

        raw_features: dict[str, Any] = {
            "smt_intensity": smt.intensity,
            "smt_diff": smt.diff,
            "smt_leader": smt.leader,
            "wick_ratio_pct": wick_ratio_pct,
            "atr_ratio": round(ratio_depth, 4),
            "has_bos": False,
            "both_sweep": gbp_setup is not None and eur_setup is not None,
            "htf_trend_direction": htf_trend_direction,
            "range_high": setup.range_high,
            "range_low": setup.range_low,
            "range_width_pips": setup.range_width_pips,
            "expansion_depth_pips": setup.expansion_depth_pips,
            "bars_stayed_outside_m5": setup.bars_stayed_outside_m5,
            "ratio_range_to_htf_atr": ratio_range,
            "ratio_depth_to_anchor_atr": ratio_depth,
            "reversal_pattern": "MEAN_REVERSION",
            "payload": setup.payload,
            "score_breakdown": setup.score_breakdown,
            "htf_bypass": l4_bypass,
            "htf_would_block": False,
            "l4_bypass": l4_bypass,
            "pyramid_layers": 0,
        }

        if setup.reject_reason or setup.candidate_score <= 0:
            raw_features["reject_reason"] = setup.reject_reason
            return StrategyResult(
                is_setup=True,
                setup_type=self.setup_type,
                direction=setup.direction,
                entry_price=setup.entry_price,
                stop_loss=setup.stop_loss,
                take_profit=setup.take_profit,
                candidate_score=0.0,
                strategy_action="REJECT",
                raw_features=raw_features,
            )

        return StrategyResult(
            is_setup=True,
            setup_type=self.setup_type,
            direction=setup.direction,
            entry_price=setup.entry_price,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            candidate_score=setup.candidate_score,
            strategy_action="ALLOW",
            raw_features=raw_features,
        )

    def evaluate(self, market_data: dict[str, Any], account_state: dict[str, Any]) -> StrategyResult:
        pair = _normalize_pair(str(market_data.get("pair", "")))
        m5 = market_data.get("m5_df") or market_data.get("ohlcv")
        if m5 is None:
            return StrategyResult(
                is_setup=False,
                setup_type=self.setup_type,
                direction="FLAT",
                strategy_action="REJECT",
                raw_features={"reject_reason": "missing_m5_data"},
            )

        spread = market_data.get("spread_series")
        result = self.evaluate_market_state(
            pair,
            m5,
            m15_df=market_data.get("m15_df"),
            h1_df=market_data.get("h1_df"),
            trigger_bar_index=market_data.get("trigger_bar_index"),
            spread_series=spread,
        )
        if result.rejected or result.setup is None:
            return StrategyResult(
                is_setup=False,
                setup_type=self.setup_type,
                direction="FLAT",
                strategy_action="REJECT",
                raw_features={"reject_reason": result.reject_reason},
            )
        analyzed = self.analyze_setup(
            result.setup,
            market_data.get("primary_setup"),
            market_data.get("secondary_setup"),
            market_data.get("h1_primary") or pd.DataFrame(),
            market_data.get("h1_secondary") or pd.DataFrame(),
        )
        return self._apply_account_guards(analyzed, market_data, account_state)
