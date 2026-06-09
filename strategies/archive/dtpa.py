"""
strategies/archive/dtpa.py — DTPA (Dow Theory structure shift + PA trigger).

ARCHIVED 2026-06: 取引回数が少ないため本番・標準 BT から外した。参照・再検討用。

Timeframes:
  - Environment (環境認識足): H4 — BOS / Dow structure (Step 1+)
  - Monitor (監視足): H1 — pullback wait + PA trigger (Step 2+)

No horizontal levels, fibonacci, or S/R flip concepts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from strategies.base import StrategyResult
from strategies.base_strategy import BaseStrategy
from strategies.htf_trend_analyzer import analyze_htf_trend, is_counter_trend, resample_to_htf
from strategies.market_utils import calc_smt_features, compute_atr, pip_size_for_pair

SETUP_TYPE = "DTPA"

DTPA_PAIR_PRIMARY = "GBPUSD"
DTPA_PAIR_SECONDARY = "EURUSD"
ALLOWED_PAIRS = frozenset({DTPA_PAIR_PRIMARY, DTPA_PAIR_SECONDARY})

# Environment = H4, Monitor = H1
ENV_BAR_MINUTES = 240
MONITOR_BAR_MINUTES = 60
ENV_RESAMPLE_RULE = "4h"

SWING_LOOKBACK_DEFAULT = 5
MONITOR_SWING_LOOKBACK = 3
MIN_CONSECUTIVE_STRUCTURE = 2
PULLBACK_MAX_H1_BARS = 48

PA_ATR_PERIOD = 14
PA_MIN_BODY_ATR_RATIO = 0.15
DEFAULT_RISK_REWARD = 2.0
SL_ATR_BUFFER_RATIO = 0.3
DTPA_L2_CANDIDATE_SCORE = int(os.getenv("DTPA_L2_CANDIDATE_SCORE", "70"))
DTPA_LLM_REJECT_BELOW = int(os.getenv("DTPA_LLM_REJECT_BELOW", "65"))
DTPA_LLM_ALLOW_MIN = int(os.getenv("DTPA_LLM_ALLOW_MIN", "85"))
LLM_MIN_CONFIDENCE = DTPA_LLM_ALLOW_MIN

REASON_STRUCTURE_INVALIDATED = "STRUCTURE_INVALIDATED"
REASON_MISS_NO_PULLBACK = "MISS_NO_PULLBACK"
REASON_REJECT_HTF_CONFLICT = "REJECT_HTF_CONFLICT"
REASON_ENGULFING = "DOW_3RD_WAVE_ENGULFING"
REASON_PINBAR = "DOW_3RD_WAVE_PINBAR"
REASON_INSIDE_BAR = "DOW_3RD_WAVE_INSIDE_BAR"

BOSDirection = Literal["LONG", "SHORT", "NONE"]
SwingLabel = Literal["HH", "HL", "LH", "LL", "UNKNOWN"]
PullbackStatus = Literal["READY", "WAITING", "INVALIDATED", "MISSED"]
PATriggerType = Literal["ENGULFING", "PIN_BAR", "INSIDE_BAR_BREAK", "NONE"]
TradeDirection = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class SwingPoint:
    bar_index: int
    timestamp: pd.Timestamp
    price: float
    kind: Literal["HIGH", "LOW"]
    label: SwingLabel


@dataclass(frozen=True)
class BOSResult:
    """Break-of-structure evaluation at a single H4 bar (環境認識足)."""

    detected: bool
    direction: BOSDirection
    bar_index: int
    timestamp: pd.Timestamp
    close_price: float
    broken_level: float | None
    structure_invalidation_level: float | None
    consecutive_structure_count: int
    bos_bar_high: float
    bos_bar_low: float
    recent_swing_high: SwingPoint | None
    recent_swing_low: SwingPoint | None
    swing_highs: tuple[SwingPoint, ...]
    swing_lows: tuple[SwingPoint, ...]

    @property
    def bos_long(self) -> bool:
        return self.detected and self.direction == "LONG"

    @property
    def bos_short(self) -> bool:
        return self.detected and self.direction == "SHORT"


@dataclass(frozen=True)
class PullbackResult:
    """Wave-2 pullback state on H1 monitor timeframe."""

    status: PullbackStatus
    reason_code: str | None
    bar_index: int
    timestamp: pd.Timestamp
    bars_since_bos: int
    zone_touched: bool
    pullback_depth_atr_ratio: float
    anchor: SwingPoint | None


@dataclass(frozen=True)
class PATriggerResult:
    trigger_type: PATriggerType
    bar_index: int
    timestamp: pd.Timestamp
    entry_price: float
    trigger_high: float
    trigger_low: float
    body_size: float
    atr: float
    reason_code: str | None

    @property
    def detected(self) -> bool:
        return self.trigger_type != "NONE"


@dataclass(frozen=True)
class DtpaSetup:
    timestamp: pd.Timestamp
    pair: str
    direction: TradeDirection
    bos: BOSResult
    pullback: PullbackResult
    pa_trigger: PATriggerResult
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    candidate_score: float
    h1_trend: str
    htf_aligned: bool
    reason_codes: tuple[str, ...]
    bar_index: int
    sweep_distance: float


def prepare_env_df(df: pd.DataFrame, *, input_is_h1: bool = False) -> pd.DataFrame:
    """
    Normalize environment OHLCV for BOS (H4).

    Pass native H4 bars, or set ``input_is_h1=True`` to resample H1 → H4.
    """
    work = _prepare_df(df)
    if input_is_h1:
        return resample_to_htf(work, rule=ENV_RESAMPLE_RULE)
    return work


def prepare_monitor_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize monitor OHLCV (H1) for pullback / PA steps."""
    return _prepare_df(df)


def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close"])
    required = {"datetime", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV missing columns: {sorted(missing)}")
    work = df.sort_values("datetime").reset_index(drop=True)
    work["datetime"] = pd.to_datetime(work["datetime"])
    return work


def _label_swings(swings: list[SwingPoint]) -> list[SwingPoint]:
    if not swings:
        return []
    labeled: list[SwingPoint] = []
    for i, swing in enumerate(swings):
        if i == 0:
            label: SwingLabel = "UNKNOWN"
        elif swing.kind == "HIGH":
            label = "HH" if swing.price > swings[i - 1].price else "LH"
        else:
            label = "HL" if swing.price > swings[i - 1].price else "LL"
        labeled.append(
            SwingPoint(
                bar_index=swing.bar_index,
                timestamp=swing.timestamp,
                price=swing.price,
                kind=swing.kind,
                label=label,
            )
        )
    return labeled


def _trailing_label_count(labels: list[SwingLabel], target: SwingLabel) -> int:
    count = 0
    for label in reversed(labels):
        if label == target:
            count += 1
        else:
            break
    return count


def _find_swings_on_work(
    work: pd.DataFrame,
    *,
    lookback: int,
    up_to_bar_index: int | None = None,
) -> tuple[list[SwingPoint], list[SwingPoint]]:
    if lookback < 1 or len(work) < lookback * 2 + 1:
        return [], []

    last_bar = len(work) - 1 if up_to_bar_index is None else min(up_to_bar_index, len(work) - 1)
    last_pivot = last_bar - lookback
    if last_pivot < lookback:
        return [], []

    highs = work["high"].astype(float).values
    lows = work["low"].astype(float).values

    swing_highs: list[SwingPoint] = []
    swing_lows: list[SwingPoint] = []

    for pivot in range(lookback, last_pivot + 1):
        left_h = highs[pivot - lookback : pivot]
        right_h = highs[pivot + 1 : pivot + lookback + 1]
        if highs[pivot] >= left_h.max() and highs[pivot] >= right_h.max():
            swing_highs.append(
                SwingPoint(
                    bar_index=pivot,
                    timestamp=pd.Timestamp(work.iloc[pivot]["datetime"]),
                    price=float(highs[pivot]),
                    kind="HIGH",
                    label="UNKNOWN",
                )
            )

        left_l = lows[pivot - lookback : pivot]
        right_l = lows[pivot + 1 : pivot + lookback + 1]
        if lows[pivot] <= left_l.min() and lows[pivot] <= right_l.min():
            swing_lows.append(
                SwingPoint(
                    bar_index=pivot,
                    timestamp=pd.Timestamp(work.iloc[pivot]["datetime"]),
                    price=float(lows[pivot]),
                    kind="LOW",
                    label="UNKNOWN",
                )
            )

    return _label_swings(swing_highs), _label_swings(swing_lows)


def find_confirmed_swings(
    df: pd.DataFrame,
    *,
    lookback: int = SWING_LOOKBACK_DEFAULT,
    up_to_bar_index: int | None = None,
) -> tuple[list[SwingPoint], list[SwingPoint]]:
    """
    H4 swing highs/lows confirmed by ``lookback`` bars on each side (環境認識足).

    A pivot at index ``p`` is confirmed at bar ``p + lookback``.
    """
    work = prepare_env_df(df)
    return _find_swings_on_work(work, lookback=lookback, up_to_bar_index=up_to_bar_index)


def find_monitor_swings(
    monitor_df: pd.DataFrame,
    *,
    lookback: int = MONITOR_SWING_LOOKBACK,
    up_to_bar_index: int | None = None,
) -> tuple[list[SwingPoint], list[SwingPoint]]:
    """H1 monitor swings for wave-2 HL/LH detection."""
    work = prepare_monitor_df(monitor_df)
    return _find_swings_on_work(work, lookback=lookback, up_to_bar_index=up_to_bar_index)


def detect_bos_at(
    env_df: pd.DataFrame,
    bar_index: int,
    lookback: int = SWING_LOOKBACK_DEFAULT,
    *,
    input_is_h1: bool = False,
) -> BOSResult:
    """Evaluate BOS on a specific H4 bar (close break of latest confirmed LH/HL)."""
    work = prepare_env_df(env_df, input_is_h1=input_is_h1)
    empty = BOSResult(
        detected=False,
        direction="NONE",
        bar_index=bar_index,
        timestamp=pd.Timestamp.utcnow(),
        close_price=0.0,
        broken_level=None,
        structure_invalidation_level=None,
        consecutive_structure_count=0,
        bos_bar_high=0.0,
        bos_bar_low=0.0,
        recent_swing_high=None,
        recent_swing_low=None,
        swing_highs=(),
        swing_lows=(),
    )
    if work.empty or bar_index < 0 or bar_index >= len(work):
        return empty

    row = work.iloc[bar_index]
    close_price = float(row["close"])
    timestamp = pd.Timestamp(row["datetime"])
    bar_high = float(row["high"])
    bar_low = float(row["low"])

    swing_highs, swing_lows = find_confirmed_swings(work, lookback=lookback, up_to_bar_index=bar_index)
    if not swing_highs or not swing_lows:
        return BOSResult(
            detected=False,
            direction="NONE",
            bar_index=bar_index,
            timestamp=timestamp,
            close_price=close_price,
            broken_level=None,
            structure_invalidation_level=None,
            consecutive_structure_count=0,
            bos_bar_high=bar_high,
            bos_bar_low=bar_low,
            recent_swing_high=swing_highs[-1] if swing_highs else None,
            recent_swing_low=swing_lows[-1] if swing_lows else None,
            swing_highs=tuple(swing_highs),
            swing_lows=tuple(swing_lows),
        )

    high_labels = [sw.label for sw in swing_highs]
    low_labels = [sw.label for sw in swing_lows]
    recent_high = swing_highs[-1]
    recent_low = swing_lows[-1]

    lh_count = _trailing_label_count(high_labels, "LH")
    ll_count = _trailing_label_count(low_labels, "LL")
    if (
        lh_count >= MIN_CONSECUTIVE_STRUCTURE
        and ll_count >= MIN_CONSECUTIVE_STRUCTURE
        and close_price > recent_high.price
    ):
        return BOSResult(
            detected=True,
            direction="LONG",
            bar_index=bar_index,
            timestamp=timestamp,
            close_price=close_price,
            broken_level=recent_high.price,
            structure_invalidation_level=recent_low.price,
            consecutive_structure_count=min(lh_count, ll_count),
            bos_bar_high=bar_high,
            bos_bar_low=bar_low,
            recent_swing_high=recent_high,
            recent_swing_low=recent_low,
            swing_highs=tuple(swing_highs),
            swing_lows=tuple(swing_lows),
        )

    hh_count = _trailing_label_count(high_labels, "HH")
    hl_count = _trailing_label_count(low_labels, "HL")
    if (
        hh_count >= MIN_CONSECUTIVE_STRUCTURE
        and hl_count >= MIN_CONSECUTIVE_STRUCTURE
        and close_price < recent_low.price
    ):
        return BOSResult(
            detected=True,
            direction="SHORT",
            bar_index=bar_index,
            timestamp=timestamp,
            close_price=close_price,
            broken_level=recent_low.price,
            structure_invalidation_level=recent_high.price,
            consecutive_structure_count=min(hh_count, hl_count),
            bos_bar_high=bar_high,
            bos_bar_low=bar_low,
            recent_swing_high=recent_high,
            recent_swing_low=recent_low,
            swing_highs=tuple(swing_highs),
            swing_lows=tuple(swing_lows),
        )

    return BOSResult(
        detected=False,
        direction="NONE",
        bar_index=bar_index,
        timestamp=timestamp,
        close_price=close_price,
        broken_level=None,
        structure_invalidation_level=None,
        consecutive_structure_count=0,
        bos_bar_high=bar_high,
        bos_bar_low=bar_low,
        recent_swing_high=recent_high,
        recent_swing_low=recent_low,
        swing_highs=tuple(swing_highs),
        swing_lows=tuple(swing_lows),
    )


def detect_bos(
    env_df: pd.DataFrame,
    lookback: int = SWING_LOOKBACK_DEFAULT,
    *,
    input_is_h1: bool = False,
) -> BOSResult:
    """
    Evaluate BOS on the latest H4 bar in ``env_df`` (環境認識足).

    Long BOS (downtrend → reversal):
      - Last two swing highs are LH, last two swing lows are LL
      - H4 close breaks above the most recent confirmed swing high (LH)

    Short BOS (uptrend → reversal):
      - Last two swing highs are HH, last two swing lows are HL
      - H4 close breaks below the most recent confirmed swing low (HL)

    Set ``input_is_h1=True`` when ``env_df`` is H1; it will be resampled to H4.
    """
    work = prepare_env_df(env_df, input_is_h1=input_is_h1)
    if work.empty:
        return detect_bos_at(work, 0, lookback=lookback)
    return detect_bos_at(work, len(work) - 1, lookback=lookback)


def scan_bos(
    env_df: pd.DataFrame,
    lookback: int = SWING_LOOKBACK_DEFAULT,
    *,
    input_is_h1: bool = False,
) -> list[BOSResult]:
    """Return every H4 bar where ``detect_bos_at`` fires (historical scan)."""
    work = prepare_env_df(env_df, input_is_h1=input_is_h1)
    start = lookback * 2
    events: list[BOSResult] = []
    for bar_index in range(start, len(work)):
        result = detect_bos_at(work, bar_index, lookback=lookback)
        if result.detected:
            events.append(result)
    return events


def _monitor_start_index(work: pd.DataFrame, bos: BOSResult) -> int | None:
    if work.empty or not bos.detected:
        return None
    after = work.index[work["datetime"] >= bos.timestamp]
    if len(after) == 0:
        return None
    return int(after[0])


def _atr_at_bar(monitor_df: pd.DataFrame, bar_index: int, period: int = PA_ATR_PERIOD) -> float:
    work = prepare_monitor_df(monitor_df)
    if bar_index < 0 or bar_index >= len(work):
        return 0.0
    window = work.iloc[: bar_index + 1]
    if len(window) < period + 1:
        return 0.0
    atr = compute_atr(window, period=period)
    value = float(atr.iloc[-1])
    return value if not pd.isna(value) else 0.0


def _pullback_zone(bos: BOSResult) -> tuple[float, float]:
    assert bos.broken_level is not None
    if bos.bos_long:
        return float(bos.bos_bar_low), float(bos.broken_level)
    return float(bos.broken_level), float(bos.bos_bar_high)


def _price_in_pullback_zone(bos: BOSResult, low: float, high: float) -> bool:
    zone_low, zone_high = _pullback_zone(bos)
    if bos.bos_long:
        return low <= zone_high
    return high >= zone_low


def wait_for_pullback(
    bos: BOSResult,
    monitor_df: pd.DataFrame,
    bar_index: int,
    *,
    swing_lookback: int = MONITOR_SWING_LOOKBACK,
    max_h1_bars: int = PULLBACK_MAX_H1_BARS,
) -> PullbackResult:
    """
    Wave-2 pullback wait on H1 (監視足).

    Long: retrace between BOS bar low and broken LH; confirm HL swing.
    Short: retrace between broken HL and BOS bar high; confirm LH swing.
    """
    work = prepare_monitor_df(monitor_df)
    waiting = PullbackResult(
        status="WAITING",
        reason_code=None,
        bar_index=bar_index,
        timestamp=pd.Timestamp(work.iloc[bar_index]["datetime"]) if 0 <= bar_index < len(work) else pd.Timestamp.utcnow(),
        bars_since_bos=0,
        zone_touched=False,
        pullback_depth_atr_ratio=0.0,
        anchor=None,
    )
    if not bos.detected or work.empty or bar_index < 0 or bar_index >= len(work):
        return waiting

    start_idx = _monitor_start_index(work, bos)
    if start_idx is None or bar_index < start_idx:
        return waiting

    zone_low, zone_high = _pullback_zone(bos)
    zone_touched = False

    for i in range(start_idx, bar_index + 1):
        row = work.iloc[i]
        close = float(row["close"])
        low = float(row["low"])
        high = float(row["high"])
        ts = pd.Timestamp(row["datetime"])
        bars_since = i - start_idx + 1
        window = work.iloc[start_idx : i + 1]

        if bos.bos_long:
            if bos.structure_invalidation_level is not None and close < bos.structure_invalidation_level:
                return PullbackResult(
                    status="INVALIDATED",
                    reason_code=REASON_STRUCTURE_INVALIDATED,
                    bar_index=i,
                    timestamp=ts,
                    bars_since_bos=bars_since,
                    zone_touched=zone_touched,
                    pullback_depth_atr_ratio=0.0,
                    anchor=None,
                )
            if _price_in_pullback_zone(bos, low, high):
                zone_touched = True
            elif (
                bars_since >= 4
                and not zone_touched
                and float(window["low"].min()) > zone_high
                and float(window["close"].max()) > bos.close_price
            ):
                return PullbackResult(
                    status="MISSED",
                    reason_code=REASON_MISS_NO_PULLBACK,
                    bar_index=i,
                    timestamp=ts,
                    bars_since_bos=bars_since,
                    zone_touched=False,
                    pullback_depth_atr_ratio=0.0,
                    anchor=None,
                )
        else:
            if bos.structure_invalidation_level is not None and close > bos.structure_invalidation_level:
                return PullbackResult(
                    status="INVALIDATED",
                    reason_code=REASON_STRUCTURE_INVALIDATED,
                    bar_index=i,
                    timestamp=ts,
                    bars_since_bos=bars_since,
                    zone_touched=zone_touched,
                    pullback_depth_atr_ratio=0.0,
                    anchor=None,
                )
            if _price_in_pullback_zone(bos, low, high):
                zone_touched = True
            elif (
                bars_since >= 4
                and not zone_touched
                and float(window["high"].max()) < zone_low
                and float(window["close"].min()) < bos.close_price
            ):
                return PullbackResult(
                    status="MISSED",
                    reason_code=REASON_MISS_NO_PULLBACK,
                    bar_index=i,
                    timestamp=ts,
                    bars_since_bos=bars_since,
                    zone_touched=False,
                    pullback_depth_atr_ratio=0.0,
                    anchor=None,
                )

        if bars_since > max_h1_bars and not zone_touched:
            return PullbackResult(
                status="MISSED",
                reason_code=REASON_MISS_NO_PULLBACK,
                bar_index=i,
                timestamp=ts,
                bars_since_bos=bars_since,
                zone_touched=False,
                pullback_depth_atr_ratio=0.0,
                anchor=None,
            )

        if zone_touched:
            _, swing_lows = find_monitor_swings(work, lookback=swing_lookback, up_to_bar_index=i)
            swing_highs, _ = find_monitor_swings(work, lookback=swing_lookback, up_to_bar_index=i)
            atr = _atr_at_bar(work, i)
            depth = (zone_high - low) / atr if bos.bos_long and atr > 0 else (high - zone_low) / atr if atr > 0 else 0.0

            if bos.bos_long and len(swing_lows) >= 2 and swing_lows[-1].label == "HL":
                return PullbackResult(
                    status="READY",
                    reason_code=None,
                    bar_index=i,
                    timestamp=ts,
                    bars_since_bos=bars_since,
                    zone_touched=True,
                    pullback_depth_atr_ratio=round(depth, 4),
                    anchor=swing_lows[-1],
                )
            if bos.bos_short and len(swing_highs) >= 2 and swing_highs[-1].label == "LH":
                return PullbackResult(
                    status="READY",
                    reason_code=None,
                    bar_index=i,
                    timestamp=ts,
                    bars_since_bos=bars_since,
                    zone_touched=True,
                    pullback_depth_atr_ratio=round(depth, 4),
                    anchor=swing_highs[-1],
                )

    row = work.iloc[bar_index]
    return PullbackResult(
        status="WAITING",
        reason_code=None,
        bar_index=bar_index,
        timestamp=pd.Timestamp(row["datetime"]),
        bars_since_bos=bar_index - start_idx + 1,
        zone_touched=zone_touched,
        pullback_depth_atr_ratio=0.0,
        anchor=None,
    )


def detect_pa_trigger(
    monitor_df: pd.DataFrame,
    bar_index: int,
    direction: BOSDirection,
    *,
    atr: float | None = None,
) -> PATriggerResult:
    """PA trigger on H1 at ``bar_index`` (wave-3 origin). Entry at trigger close."""
    work = prepare_monitor_df(monitor_df)
    empty = PATriggerResult(
        trigger_type="NONE",
        bar_index=bar_index,
        timestamp=pd.Timestamp.utcnow(),
        entry_price=0.0,
        trigger_high=0.0,
        trigger_low=0.0,
        body_size=0.0,
        atr=0.0,
        reason_code=None,
    )
    if direction not in ("LONG", "SHORT") or bar_index < 1 or bar_index >= len(work):
        return empty

    row = work.iloc[bar_index]
    prev = work.iloc[bar_index - 1]
    open_ = float(row["open"])
    close = float(row["close"])
    high = float(row["high"])
    low = float(row["low"])
    prev_open = float(prev["open"])
    prev_close = float(prev["close"])
    prev_high = float(prev["high"])
    prev_low = float(prev["low"])
    ts = pd.Timestamp(row["datetime"])
    atr_val = float(atr) if atr is not None else _atr_at_bar(work, bar_index)
    body = abs(close - open_)

    if atr_val <= 0 or body < PA_MIN_BODY_ATR_RATIO * atr_val:
        return empty

    if direction == "LONG":
        if prev_close < prev_open and close > open_ and open_ <= prev_close and close >= prev_open:
            return PATriggerResult(
                trigger_type="ENGULFING",
                bar_index=bar_index,
                timestamp=ts,
                entry_price=close,
                trigger_high=high,
                trigger_low=low,
                body_size=body,
                atr=atr_val,
                reason_code=REASON_ENGULFING,
            )
        lower_wick = min(open_, close) - low
        if lower_wick >= 2.0 * body and close > open_:
            return PATriggerResult(
                trigger_type="PIN_BAR",
                bar_index=bar_index,
                timestamp=ts,
                entry_price=close,
                trigger_high=high,
                trigger_low=low,
                body_size=body,
                atr=atr_val,
                reason_code=REASON_PINBAR,
            )
        if bar_index >= 2:
            prev2 = work.iloc[bar_index - 2]
            inside = prev_high <= float(prev2["high"]) and prev_low >= float(prev2["low"])
            if inside and close > open_ and close > prev_high:
                return PATriggerResult(
                    trigger_type="INSIDE_BAR_BREAK",
                    bar_index=bar_index,
                    timestamp=ts,
                    entry_price=close,
                    trigger_high=high,
                    trigger_low=low,
                    body_size=body,
                    atr=atr_val,
                    reason_code=REASON_INSIDE_BAR,
                )
    else:
        if prev_close > prev_open and close < open_ and open_ >= prev_close and close <= prev_open:
            return PATriggerResult(
                trigger_type="ENGULFING",
                bar_index=bar_index,
                timestamp=ts,
                entry_price=close,
                trigger_high=high,
                trigger_low=low,
                body_size=body,
                atr=atr_val,
                reason_code=REASON_ENGULFING,
            )
        upper_wick = high - max(open_, close)
        if upper_wick >= 2.0 * body and close < open_:
            return PATriggerResult(
                trigger_type="PIN_BAR",
                bar_index=bar_index,
                timestamp=ts,
                entry_price=close,
                trigger_high=high,
                trigger_low=low,
                body_size=body,
                atr=atr_val,
                reason_code=REASON_PINBAR,
            )
        if bar_index >= 2:
            prev2 = work.iloc[bar_index - 2]
            inside = prev_high <= float(prev2["high"]) and prev_low >= float(prev2["low"])
            if inside and close < open_ and close < prev_low:
                return PATriggerResult(
                    trigger_type="INSIDE_BAR_BREAK",
                    bar_index=bar_index,
                    timestamp=ts,
                    entry_price=close,
                    trigger_high=high,
                    trigger_low=low,
                    body_size=body,
                    atr=atr_val,
                    reason_code=REASON_INSIDE_BAR,
                )

    return empty


def build_risk_levels(
    direction: TradeDirection,
    entry: float,
    trigger_high: float,
    trigger_low: float,
    atr: float,
    *,
    sl_buffer_ratio: float = SL_ATR_BUFFER_RATIO,
    risk_reward: float = DEFAULT_RISK_REWARD,
) -> tuple[float, float] | None:
    """SL beyond PA trigger wick + ATR buffer; TP at fixed RR."""
    buffer = sl_buffer_ratio * atr
    if direction == "BUY":
        stop_loss = float(trigger_low) - buffer
        risk = entry - stop_loss
        if risk <= 0:
            return None
        return stop_loss, entry + risk_reward * risk
    stop_loss = float(trigger_high) + buffer
    risk = stop_loss - entry
    if risk <= 0:
        return None
    return stop_loss, entry - risk_reward * risk


def build_llm_context(
    bos: BOSResult,
    pullback: PullbackResult,
    pa: PATriggerResult,
    *,
    htf_aligned: bool,
) -> dict[str, Any]:
    """Gemini L4 audit payload (confidence_score requested downstream)."""
    return {
        "setup_type": SETUP_TYPE,
        "bos_direction": bos.direction,
        "bars_since_bos": pullback.bars_since_bos,
        "pullback_depth_atr_ratio": pullback.pullback_depth_atr_ratio,
        "pa_trigger_type": pa.trigger_type,
        "htf_aligned": htf_aligned,
        "consecutive_structure_count": bos.consecutive_structure_count,
        "broken_level": bos.broken_level,
        "structure_invalidation_level": bos.structure_invalidation_level,
        "anchor_price": pullback.anchor.price if pullback.anchor else None,
    }


def _env_bar_index(monitor: pd.DataFrame, env: pd.DataFrame, bar_index: int) -> int:
    ts = pd.Timestamp(monitor.iloc[bar_index]["datetime"])
    idx = int(env["datetime"].searchsorted(ts, side="right")) - 1
    return max(0, min(idx, len(env) - 1))


def _bos_to_trade_direction(direction: BOSDirection) -> TradeDirection:
    return "BUY" if direction == "LONG" else "SELL"


def detect_dtpa_setups(
    df: pd.DataFrame,
    pair_name: str,
    h1_df: pd.DataFrame | None = None,
    *,
    progress_hook: Any | None = None,
) -> list[DtpaSetup]:
    if pair_name.upper() not in ALLOWED_PAIRS:
        return []
    return DtpaStrategy().detect_setups(
        df,
        pair_name,
        h1_df=h1_df,
        progress_hook=progress_hook,
    )


def detect_dtpa_setups_for_pair(
    df: pd.DataFrame,
    pair_name: str,
    h1_df: pd.DataFrame | None = None,
    progress_hook: Any | None = None,
) -> list[DtpaSetup]:
    return detect_dtpa_setups(df, pair_name, h1_df, progress_hook=progress_hook)


class DtpaStrategy(BaseStrategy):
    """Dow Theory BOS → H1 pullback → PA trigger (3rd wave)."""

    @property
    def setup_type(self) -> str:
        return SETUP_TYPE

    def detect_setups(
        self,
        df: pd.DataFrame,
        pair_name: str,
        h1_df: pd.DataFrame | None = None,
        *,
        progress_hook: Any | None = None,
    ) -> list[DtpaSetup]:
        if pair_name.upper() not in ALLOWED_PAIRS:
            return []
        monitor = prepare_monitor_df(h1_df if h1_df is not None else df)
        env = prepare_env_df(h1_df if h1_df is not None else df, input_is_h1=True)
        setups: list[DtpaSetup] = []
        active_bos: BOSResult | None = None
        start = MONITOR_SWING_LOOKBACK * 2
        total = max(len(monitor) - start, 1)

        for bar_index in range(start, len(monitor)):
            if progress_hook is not None:
                progress_hook(bar_index - start + 1, total)
            env_idx = _env_bar_index(monitor, env, bar_index)
            bos = detect_bos_at(env, env_idx, lookback=SWING_LOOKBACK_DEFAULT)
            if bos.detected:
                active_bos = bos

            if active_bos is None or not active_bos.detected:
                continue

            pullback = wait_for_pullback(active_bos, monitor, bar_index)
            if pullback.status in ("INVALIDATED", "MISSED"):
                active_bos = None
                continue
            if pullback.status != "READY":
                continue

            pa = detect_pa_trigger(monitor, bar_index, active_bos.direction)
            if not pa.detected:
                continue

            htf = analyze_htf_trend(monitor, monitor.iloc[bar_index]["datetime"])
            trade_dir = _bos_to_trade_direction(active_bos.direction)
            aligned = not is_counter_trend(trade_dir, htf.direction)
            if not aligned:
                active_bos = None
                continue

            levels = build_risk_levels(
                trade_dir,
                pa.entry_price,
                pa.trigger_high,
                pa.trigger_low,
                pa.atr,
            )
            if levels is None:
                continue
            stop_loss, take_profit = levels
            reason_codes = tuple(
                code
                for code in (pa.reason_code,)
                if code is not None
            )
            setups.append(
                DtpaSetup(
                    timestamp=pa.timestamp,
                    pair=pair_name.upper(),
                    direction=trade_dir,
                    bos=active_bos,
                    pullback=pullback,
                    pa_trigger=pa,
                    entry_price=pa.entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward=DEFAULT_RISK_REWARD,
                    candidate_score=float(DTPA_L2_CANDIDATE_SCORE),
                    h1_trend=htf.direction,
                    htf_aligned=aligned,
                    reason_codes=reason_codes,
                    bar_index=bar_index,
                    sweep_distance=abs(float(pa.trigger_high) - float(pa.trigger_low)),
                )
            )
            active_bos = None

        return setups

    def analyze_setup(
        self,
        setup: DtpaSetup,
        gbp_setup: DtpaSetup | None,
        eur_setup: DtpaSetup | None,
        h1_gbp: pd.DataFrame,
        h1_eur: pd.DataFrame,
    ) -> StrategyResult:
        h1_ref = h1_gbp if setup.pair == DTPA_PAIR_PRIMARY else h1_eur
        htf = analyze_htf_trend(h1_ref, setup.timestamp)
        aligned = not is_counter_trend(setup.direction, htf.direction)
        smt_feats = calc_smt_features(
            gbp_setup,
            eur_setup,
            pip_size=pip_size_for_pair(setup.pair),
        )
        atr_ratio = (
            setup.pullback.pullback_depth_atr_ratio
            if setup.pullback.pullback_depth_atr_ratio > 0
            else setup.pa_trigger.atr / max(setup.entry_price, 1e-9)
        )
        llm_ctx = build_llm_context(setup.bos, setup.pullback, setup.pa_trigger, htf_aligned=aligned)
        raw: dict[str, Any] = {
            **llm_ctx,
            "smt_intensity": smt_feats.intensity,
            "smt_diff": smt_feats.diff,
            "smt_leader": smt_feats.leader,
            "wick_ratio_pct": 0.0,
            "atr_ratio": round(float(atr_ratio), 4),
            "has_bos": True,
            "both_sweep": gbp_setup is not None and eur_setup is not None,
            "htf_trend_direction": htf.direction,
            "reject_reason": "" if aligned else "REJECT_BY_HTF_TREND",
            "reason_codes": list(setup.reason_codes),
            "candidate_score": setup.candidate_score,
        }
        if not aligned:
            return StrategyResult(
                is_setup=False,
                setup_type=self.setup_type,
                direction=setup.direction,
                raw_features=raw,
            )
        return StrategyResult(
            is_setup=True,
            setup_type=self.setup_type,
            direction=setup.direction,
            entry_price=setup.entry_price,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            candidate_score=setup.candidate_score,
            raw_features=raw,
        )


__all__ = [
    "ALLOWED_PAIRS",
    "BOSResult",
    "BOSDirection",
    "DEFAULT_RISK_REWARD",
    "DTPA_PAIR_PRIMARY",
    "DTPA_PAIR_SECONDARY",
    "DtpaSetup",
    "DtpaStrategy",
    "ENV_BAR_MINUTES",
    "ENV_RESAMPLE_RULE",
    "DTPA_L2_CANDIDATE_SCORE",
    "DTPA_LLM_ALLOW_MIN",
    "DTPA_LLM_REJECT_BELOW",
    "LLM_MIN_CONFIDENCE",
    "MIN_CONSECUTIVE_STRUCTURE",
    "MONITOR_BAR_MINUTES",
    "MONITOR_SWING_LOOKBACK",
    "PATriggerResult",
    "PATriggerType",
    "PA_ATR_PERIOD",
    "PA_MIN_BODY_ATR_RATIO",
    "PULLBACK_MAX_H1_BARS",
    "PullbackResult",
    "PullbackStatus",
    "REASON_ENGULFING",
    "REASON_INSIDE_BAR",
    "REASON_MISS_NO_PULLBACK",
    "REASON_PINBAR",
    "REASON_REJECT_HTF_CONFLICT",
    "REASON_STRUCTURE_INVALIDATED",
    "SETUP_TYPE",
    "SL_ATR_BUFFER_RATIO",
    "SWING_LOOKBACK_DEFAULT",
    "SwingPoint",
    "TradeDirection",
    "build_llm_context",
    "build_risk_levels",
    "detect_bos",
    "detect_bos_at",
    "detect_dtpa_setups",
    "detect_dtpa_setups_for_pair",
    "detect_pa_trigger",
    "find_confirmed_swings",
    "find_monitor_swings",
    "prepare_env_df",
    "prepare_monitor_df",
    "scan_bos",
    "wait_for_pullback",
]
