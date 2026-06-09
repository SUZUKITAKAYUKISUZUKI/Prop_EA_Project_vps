"""
strategies/archive/vexp.py — VEXP (Volatility Expansion Strategy).

ARCHIVED 2026-06: 取引回数が少ないため本番・標準 BT から外した。参照・再検討用。

**Portfolio slot:** Strategy **E** (low correlation vs LSFC / ALS / FVG / TREF)

Strategy thesis
---------------
Volatility mean-reverts, but after extreme compression directional energy release
is common. VEXP trades **Coiled spring → First impulse** only — not trend follow,
not range fade. Third regime.

Deprecated research patterns (forbidden)
----------------------------------------
- Feature Store / Bayes-first research tooling before strategy
- 10-stage gates + threshold optimization
- Composite score weight tuning
- Post-hoc entry condition additions

VEXP: **≤5 detection stages**, **fixed thresholds**, minimal backtestable MVP.

Differentiation vs TREF / FVG
-----------------------------
|        | TREF              | FVG           | VEXP                    |
|--------|-------------------|---------------|-------------------------|
| Premise| Tokyo range fail  | Gap fill      | ATR compress → expand   |
| Bias   | Counter-trend     | LONG          | Breakout direction      |
| Time   | JST session       | London        | Compression cycle       |
| Pairs  | AUDJPY / USDJPY   | GBPUSD        | GBPUSD / EURUSD (M15)   |

Core definitions
----------------
**Compression (M15, ATR period 20):**

    atr_now      = ATR(20) at bar t
    atr_baseline = mean daily ATR over last 30 calendar days (same TF)
    ratio        = atr_now / atr_baseline
    COMPRESSION  ⇔  ratio <= 0.40  (fixed, no optimization)

Min compression bars: **8** (2h M15). Coil box: high_max / low_min over compression.

**Expansion trigger (Armed)** — either:

    A) atr_ratio >= 0.55  (hysteresis)
    B) bar_range >= 1.5 × atr_now

**Direction:**

    LONG  : close > coil_high AND close > open
    SHORT : close < coil_low  AND close < open

Close break required (no wick-only break). No mid-coil entries.

**First move only:**

- 1 trade per compression cycle
- Max holding: 32 M15 bars (~8h)
- TP: 1.2R fixed (2.0R tested unfavorably vs 1.2R with 32-bar hold; see BT step2)
- No partials (MVP); re-arm only after new compression cycle

Filters (MVP)
-------------
1. Pairs: GBPUSD, EURUSD
2. Session: London **15:00–20:59** server (``session_dst``) — **breakout bar only**;
   ARMED may occur off-session (overnight carry, timeout 80 bars)
3. Spread: ``spread <= 1.5 × median_spread_20d`` (Phase 2)
4. HTF: H1 counter-trend blocked (``htf_trend_analyzer``)

candidate_score (L2)
--------------------
::

    score = clamp(
        40
      + compression_depth_score   # ratio 0.40→0.25 → +30
      + breakout_strength_score   # (close - coil_edge) / ATR
      + expansion_bar_bonus
      + htf_alignment_bonus       # +10 if aligned
    , 0, 100)

L2 minimum: **80** (exclude 55–79 low-quality band).

Compression FSM (``CompressionFSM``)
------------------------------------
Phases: IDLE → COMPRESSING → ARMED → IDLE (RESET / EXPIRED / CONSUMED).

- **IDLE → COMPRESSING:** ``atr_ratio <= 0.40``; new ``cycle_id``, init coil
- **COMPRESSING:** while ``ratio <= 0.40``: increment bars, widen coil, track
  ``atr_ratio_min``; do not increment when ratio above threshold
- **COMPRESSING → ARMED:** ``bars >= 8`` AND (ratio >= 0.55 OR expansion bar);
  freeze ``VexpCompressionSnapshot``
- **COMPRESSING → IDLE:** ``ratio > 0.40`` AND no arm trigger → RESET
- **ARMED → IDLE:** ``mark_consumed()`` after setup, or timeout **80 bars** → EXPIRED

Coil: ``coil_high = max(high)``, ``coil_low = min(low)`` over compression bars;
frozen at arm time.

ATR baseline (``build_atr_context``)
------------------------------------
::

    daily_atr[date]   = mean(ATR(20) of all M15 bars on that date)
    baseline_at_bar   = mean(daily_atr over last 30 calendar days ending bar.date)
    atr_ratio         = atr_now / baseline_at_bar

Bars with NaN baseline are skipped (FSM stays IDLE).

Module layout
-------------
::

    Constants          — fixed thresholds (no optimization)
    Types              — CompressionBarInput, VexpSetup, snapshots
    Config             — VexpConfig + env overrides
    ATR engine         — build_atr_context, build_atr_series
    Candidate score    — L2 scoring for execution priority
    Compression FSM    — CompressionFSM, scan_compression_events
    Breakout           — detect_breakout_direction, build_risk_levels
    Setup detector     — detect_vexp_setups, session filter
    VexpStrategy       — BaseStrategy adapter

Platform integration
--------------------
- ``strategies/__init__.py`` — ``VexpStrategy`` registry, mode ``vexp``, letter ``E``
- ``backtest_runner.py`` — ``--strategy vexp``; default CSV ``backtest_results/vexp_m15_h1_3y.csv``
- ``backtest_tracking.py`` — L5 = M15, max holding = ``MAX_HOLDING_BARS``
- ``main_platform.py`` — ``VEXP_SETUP_TYPE`` in ``RULE_BASE_ONLY_SETUP_TYPES``
- Portfolio ``abcde`` (future); LSFC conflict → ``strategy_priority_index``

Backtest
--------
::

    python backtest_runner.py --strategy vexp

Success criteria (3y): 30–120 trades, E[R] > 0.05, DD contribution < LSFC solo,
correlation with LSFC < 0.3. Tests: ``python -m unittest test_vexp``.

Fixed parameters (no optimization)
----------------------------------
| Parameter              | Value                          |
|------------------------|--------------------------------|
| ATR period             | 20                             |
| Baseline window        | 30 calendar days               |
| Compression threshold  | ratio ≤ 0.40                   |
| Arm threshold          | ratio ≥ 0.55                   |
| Min compression bars   | 8                              |
| Breakout range mult    | 1.5 × ATR                      |
| TP                     | 1.2R (2.0R rejected in 3y BT)  |
| SL buffer              | 0.25 × ATR                     |
| Max holding            | 32 M15 bars                    |
| L2 min score           | 80                             |
| Allowed pairs          | GBPUSD, EURUSD                 |
| Armed timeout          | 80 M15 bars (~20h)             |
| Session (breakout only)| Server 15–20 (London block)    |
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Callable, Literal

import pandas as pd

from strategies.base import StrategyResult
from strategies.base_strategy import BaseStrategy
from strategies.htf_trend_analyzer import analyze_htf_trend, build_htf_trend_lookup, is_counter_trend
from strategies.market_utils import (
    LONDON_SESSION_HOUR_END,
    LONDON_SESSION_HOUR_START,
    SMTFeatures,
    calc_smt_features,
    compute_atr,
    pip_size_for_pair,
)
from strategies.session_dst import DATA_DST_TYPE, shift_hour_range


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


# ---------------------------------------------------------------------------
# Constants (fixed thresholds — no optimization; env overrides for BT sweeps)
# ---------------------------------------------------------------------------

SETUP_TYPE = "VEXP_VOLATILITY_EXPANSION"
VEXP_VERSION = "0.1.0"

VEXP_PAIR_PRIMARY = "GBPUSD"
VEXP_PAIR_SECONDARY = "EURUSD"
ALLOWED_PAIRS = frozenset({VEXP_PAIR_PRIMARY, VEXP_PAIR_SECONDARY})

ATR_PERIOD = 20
BASELINE_LOOKBACK_DAYS = 30
COMPRESSION_RATIO_MAX = 0.40
ARM_RATIO_MIN = 0.55
MIN_COMPRESSION_BARS = 8
EXPANSION_BAR_RANGE_MULT = 1.5

VEXP_SCORE_BASE = 40.0
VEXP_COMPRESSION_RATIO_START = 0.40
VEXP_COMPRESSION_RATIO_DEEP = 0.25
VEXP_COMPRESSION_DEPTH_MAX = 30.0
VEXP_BREAKOUT_STRENGTH_MAX = 20.0
VEXP_BREAKOUT_EDGE_FULL_ATR = 0.50
VEXP_EXPANSION_BAR_BONUS_MAX = 5.0
VEXP_HTF_ALIGNMENT_BONUS = 10.0
VEXP_L2_MIN_CANDIDATE_SCORE = _env_float("VEXP_L2_MIN_SCORE", 80.0)

DEFAULT_RISK_REWARD = _env_float("VEXP_RISK_REWARD", 1.2)
SL_ATR_BUFFER_RATIO = 0.25
MAX_HOLDING_BARS = _env_int("VEXP_MAX_HOLDING_BARS", 32)
ARMED_TIMEOUT_BARS = 80

VEXP_SESSION_HOUR_START = LONDON_SESSION_HOUR_START
VEXP_SESSION_HOUR_END = LONDON_SESSION_HOUR_END

HTF_TREND_MISMATCH_TAG = "HTF_TREND_MISMATCH"

ProgressHook = Callable[[int, int], None]

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

TradeDirection = Literal["BUY", "SELL"]
ArmReason = Literal["ATR_RATIO", "EXPANSION_BAR", "NONE"]


class CompressionPhase(str, Enum):
    IDLE = "IDLE"
    COMPRESSING = "COMPRESSING"
    ARMED = "ARMED"


class CompressionEvent(str, Enum):
    ENTERED = "ENTERED"
    ARMED = "ARMED"
    EXPIRED = "EXPIRED"
    RESET = "RESET"
    CONSUMED = "CONSUMED"


@dataclass(frozen=True)
class CompressionBarInput:
    bar_index: int
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    atr: float
    baseline_atr: float
    atr_ratio: float

    @property
    def bar_range(self) -> float:
        return max(0.0, float(self.high) - float(self.low))


@dataclass(frozen=True)
class VexpScoreInput:
    direction: TradeDirection
    close: float
    open: float
    high: float
    low: float
    atr: float
    coil_high: float
    coil_low: float
    atr_ratio_at_arm: float
    atr_ratio_min: float
    bars_in_compression: int
    htf_aligned: bool
    expansion_bar: bool = False


@dataclass(frozen=True)
class VexpScoreBreakdown:
    base: float
    compression_depth: float
    breakout_strength: float
    expansion_bar_bonus: float
    htf_alignment: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return {
            "score_base": self.base,
            "score_compression_depth": self.compression_depth,
            "score_breakout_strength": self.breakout_strength,
            "score_expansion_bar_bonus": self.expansion_bar_bonus,
            "score_htf_alignment": self.htf_alignment,
            "candidate_score": self.total,
        }


@dataclass(frozen=True)
class VexpCompressionSnapshot:
    cycle_id: str
    start_timestamp: pd.Timestamp
    arm_timestamp: pd.Timestamp
    coil_high: float
    coil_low: float
    bars_in_compression: int
    atr_ratio_min: float
    atr_ratio_at_arm: float
    baseline_atr: float
    arm_reason: ArmReason


@dataclass(frozen=True)
class CompressionUpdateResult:
    phase: CompressionPhase
    event: CompressionEvent | None
    snapshot: VexpCompressionSnapshot | None
    bars_in_compression: int
    meets_min_bars: bool
    is_armed: bool
    arm_reason: ArmReason


@dataclass(frozen=True)
class CompressionStateView:
    phase: CompressionPhase
    cycle_id: str | None
    bars_in_compression: int
    coil_high: float | None
    coil_low: float | None
    atr_ratio_min: float | None
    armed_bars: int


@dataclass(frozen=True)
class VexpSetup:
    timestamp: pd.Timestamp
    pair: str
    direction: TradeDirection
    bar_index: int
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    atr: float
    atr_ratio: float
    coil_high: float
    coil_low: float
    coil_width_pips: float
    breakout_strength: float
    cycle_id: str
    bars_in_compression: int
    atr_ratio_min: float
    arm_reason: ArmReason
    candidate_score: float
    h1_trend: str
    session: str
    htf_aligned: bool
    expansion_bar: bool
    sweep_distance: float


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VexpConfig:
    atr_period: int = ATR_PERIOD
    baseline_lookback_days: int = BASELINE_LOOKBACK_DAYS
    compression_ratio_max: float = COMPRESSION_RATIO_MAX
    arm_ratio_min: float = ARM_RATIO_MIN
    min_compression_bars: int = MIN_COMPRESSION_BARS
    expansion_bar_range_mult: float = EXPANSION_BAR_RANGE_MULT
    armed_timeout_bars: int = ARMED_TIMEOUT_BARS
    session_hour_start: int = VEXP_SESSION_HOUR_START
    session_hour_end: int = VEXP_SESSION_HOUR_END
    risk_reward: float = DEFAULT_RISK_REWARD


def load_vexp_config() -> VexpConfig:
    return VexpConfig(
        atr_period=_env_int("VEXP_ATR_PERIOD", ATR_PERIOD),
        baseline_lookback_days=_env_int("VEXP_BASELINE_DAYS", BASELINE_LOOKBACK_DAYS),
        compression_ratio_max=_env_float("VEXP_COMPRESSION_RATIO_MAX", COMPRESSION_RATIO_MAX),
        arm_ratio_min=_env_float("VEXP_ARM_RATIO_MIN", ARM_RATIO_MIN),
        min_compression_bars=_env_int("VEXP_MIN_COMPRESSION_BARS", MIN_COMPRESSION_BARS),
        expansion_bar_range_mult=_env_float("VEXP_EXPANSION_BAR_MULT", EXPANSION_BAR_RANGE_MULT),
        armed_timeout_bars=_env_int("VEXP_ARMED_TIMEOUT_BARS", ARMED_TIMEOUT_BARS),
        session_hour_start=_env_int("VEXP_SESSION_HOUR_START", VEXP_SESSION_HOUR_START),
        session_hour_end=_env_int("VEXP_SESSION_HOUR_END", VEXP_SESSION_HOUR_END),
        risk_reward=_env_float("VEXP_RISK_REWARD", DEFAULT_RISK_REWARD),
    )


# ---------------------------------------------------------------------------
# ATR engine
#
# daily_atr[date]   = mean(ATR(20) of all M15 bars on that date)
# baseline_at_bar   = mean(daily_atr over last 30 calendar days)
# atr_ratio         = atr_now / baseline_at_bar
# ---------------------------------------------------------------------------


def build_atr_series(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    return compute_atr(df, period=period)


def build_atr_baseline_series_fast(
    df: pd.DataFrame,
    atr_series: pd.Series,
    lookback_days: int = BASELINE_LOOKBACK_DAYS,
) -> pd.Series:
    if df is None or df.empty or atr_series.empty:
        return pd.Series(dtype=float)

    work = pd.DataFrame(
        {"datetime": pd.to_datetime(df["datetime"]), "atr": atr_series.values},
        index=df.index,
    )
    work["date"] = work["datetime"].dt.date
    daily = work.groupby("date")["atr"].mean().sort_index()

    baseline_by_date: dict[object, float] = {}
    for i, session_date in enumerate(daily.index):
        start = max(0, i - lookback_days + 1)
        window = daily.iloc[start : i + 1]
        baseline_by_date[session_date] = float(window.mean()) if not window.empty else float("nan")

    return work["date"].map(baseline_by_date)


def build_atr_context(
    df: pd.DataFrame,
    period: int = ATR_PERIOD,
    baseline_days: int = BASELINE_LOOKBACK_DAYS,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    atr = build_atr_series(df, period=period)
    baseline = build_atr_baseline_series_fast(df, atr, lookback_days=baseline_days)
    ratio = atr / baseline.replace(0, pd.NA)
    return atr, baseline, ratio


def atr_ratio_at(
    atr_series: pd.Series,
    baseline_series: pd.Series,
    bar_index: int,
) -> float | None:
    if bar_index < 0 or bar_index >= len(atr_series):
        return None
    atr = float(atr_series.iloc[bar_index])
    baseline = float(baseline_series.iloc[bar_index])
    if pd.isna(atr) or pd.isna(baseline) or baseline <= 0:
        return None
    return atr / baseline


# ---------------------------------------------------------------------------
# Candidate score (L2)
#
# score = clamp(40 + depth + strength + expansion + htf, 0, 100); min 55
# ---------------------------------------------------------------------------


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def score_compression_depth(atr_ratio_min: float) -> float:
    ratio = float(atr_ratio_min)
    if ratio >= VEXP_COMPRESSION_RATIO_START:
        return 0.0
    if ratio <= VEXP_COMPRESSION_RATIO_DEEP:
        return VEXP_COMPRESSION_DEPTH_MAX
    span = VEXP_COMPRESSION_RATIO_START - VEXP_COMPRESSION_RATIO_DEEP
    if span <= 0:
        return 0.0
    progress = (VEXP_COMPRESSION_RATIO_START - ratio) / span
    return round(VEXP_COMPRESSION_DEPTH_MAX * progress, 2)


def breakout_edge_distance(inp: VexpScoreInput) -> float:
    if inp.direction == "BUY":
        return max(0.0, float(inp.close) - float(inp.coil_high))
    return max(0.0, float(inp.coil_low) - float(inp.close))


def score_breakout_strength(inp: VexpScoreInput) -> float:
    atr = float(inp.atr)
    if atr <= 0:
        return 0.0
    edge_ratio = breakout_edge_distance(inp) / atr
    if edge_ratio <= 0:
        return 0.0
    linear = (edge_ratio / VEXP_BREAKOUT_EDGE_FULL_ATR) * VEXP_BREAKOUT_STRENGTH_MAX
    return round(min(VEXP_BREAKOUT_STRENGTH_MAX, linear), 2)


def score_expansion_bar_bonus(inp: VexpScoreInput) -> float:
    atr = float(inp.atr)
    if atr <= 0:
        return 0.0
    bar_range = float(inp.high) - float(inp.low)
    if bar_range < EXPANSION_BAR_RANGE_MULT * atr:
        return 0.0
    excess = (bar_range / atr) - EXPANSION_BAR_RANGE_MULT
    return round(min(VEXP_EXPANSION_BAR_BONUS_MAX, excess * VEXP_EXPANSION_BAR_BONUS_MAX), 2)


def score_htf_alignment(htf_aligned: bool) -> float:
    return VEXP_HTF_ALIGNMENT_BONUS if htf_aligned else 0.0


def calc_vexp_candidate_score(inp: VexpScoreInput) -> VexpScoreBreakdown:
    compression = score_compression_depth(inp.atr_ratio_min)
    breakout = score_breakout_strength(inp)
    expansion_bonus = score_expansion_bar_bonus(inp)
    htf = score_htf_alignment(inp.htf_aligned)
    raw_total = VEXP_SCORE_BASE + compression + breakout + expansion_bonus + htf
    total = round(_clamp(raw_total), 2)
    return VexpScoreBreakdown(
        base=VEXP_SCORE_BASE,
        compression_depth=compression,
        breakout_strength=breakout,
        expansion_bar_bonus=expansion_bonus,
        htf_alignment=htf,
        total=total,
    )


def calc_vexp_candidate_score_total(inp: VexpScoreInput) -> float:
    return calc_vexp_candidate_score(inp).total


def passes_vexp_l2_gate(score: float) -> bool:
    return float(score) >= VEXP_L2_MIN_CANDIDATE_SCORE


def is_valid_breakout_candle(inp: VexpScoreInput) -> bool:
    if inp.direction == "BUY":
        return float(inp.close) > float(inp.coil_high) and float(inp.close) > float(inp.open)
    return float(inp.close) < float(inp.coil_low) and float(inp.close) < float(inp.open)


# ---------------------------------------------------------------------------
# Compression FSM
#
# IDLE → COMPRESSING (ratio<=0.40) → ARMED (bars>=8 + ratio>=0.55|expansion)
#   → IDLE via RESET / EXPIRED (80 bars) / mark_consumed()
# Breakout evaluation is downstream (Breakout + Setup detector sections).
# ---------------------------------------------------------------------------


def _new_cycle_id() -> str:
    return uuid.uuid4().hex[:12]


def is_compression_ratio(ratio: float, compression_max: float) -> bool:
    return ratio <= compression_max


def is_armed_by_ratio(ratio: float, arm_min: float) -> bool:
    return ratio >= arm_min


def is_expansion_bar(bar: CompressionBarInput, mult: float = EXPANSION_BAR_RANGE_MULT) -> bool:
    if bar.atr <= 0:
        return False
    return bar.bar_range >= mult * bar.atr


def evaluate_arm_trigger(
    bar: CompressionBarInput,
    *,
    arm_ratio_min: float,
    expansion_mult: float = EXPANSION_BAR_RANGE_MULT,
) -> ArmReason:
    if is_armed_by_ratio(bar.atr_ratio, arm_ratio_min):
        return "ATR_RATIO"
    if is_expansion_bar(bar, expansion_mult):
        return "EXPANSION_BAR"
    return "NONE"


@dataclass
class CompressionFSM:
    """ATR compression cycle state machine.

    Input per bar: ``CompressionBarInput`` (OHLC + atr + baseline_atr + atr_ratio).
    Output per bar: ``CompressionUpdateResult`` (phase, event, snapshot, is_armed).

    See module docstring for full transition table and coil box rules.
    """

    config: VexpConfig = field(default_factory=load_vexp_config)
    _phase: CompressionPhase = CompressionPhase.IDLE
    _cycle_id: str | None = None
    _start_timestamp: pd.Timestamp | None = None
    _coil_high: float | None = None
    _coil_low: float | None = None
    _bars_in_compression: int = 0
    _atr_ratio_min: float | None = None
    _baseline_atr: float | None = None
    _snapshot: VexpCompressionSnapshot | None = None
    _armed_bars: int = 0

    def reset(self) -> None:
        self._phase = CompressionPhase.IDLE
        self._cycle_id = None
        self._start_timestamp = None
        self._coil_high = None
        self._coil_low = None
        self._bars_in_compression = 0
        self._atr_ratio_min = None
        self._baseline_atr = None
        self._snapshot = None
        self._armed_bars = 0

    @property
    def phase(self) -> CompressionPhase:
        return self._phase

    @property
    def snapshot(self) -> VexpCompressionSnapshot | None:
        return self._snapshot

    @property
    def meets_min_bars(self) -> bool:
        return self._bars_in_compression >= self.config.min_compression_bars

    def view(self) -> CompressionStateView:
        return CompressionStateView(
            phase=self._phase,
            cycle_id=self._cycle_id,
            bars_in_compression=self._bars_in_compression,
            coil_high=self._coil_high,
            coil_low=self._coil_low,
            atr_ratio_min=self._atr_ratio_min,
            armed_bars=self._armed_bars,
        )

    def _start_cycle(self, bar: CompressionBarInput) -> None:
        self._cycle_id = _new_cycle_id()
        self._start_timestamp = pd.Timestamp(bar.timestamp)
        self._phase = CompressionPhase.COMPRESSING
        self._coil_high = float(bar.high)
        self._coil_low = float(bar.low)
        self._bars_in_compression = 1
        self._atr_ratio_min = float(bar.atr_ratio)
        self._baseline_atr = float(bar.baseline_atr)
        self._snapshot = None
        self._armed_bars = 0

    def _update_coil(self, bar: CompressionBarInput) -> None:
        self._coil_high = max(float(self._coil_high or bar.high), float(bar.high))
        self._coil_low = min(float(self._coil_low or bar.low), float(bar.low))
        self._bars_in_compression += 1
        if self._atr_ratio_min is None:
            self._atr_ratio_min = float(bar.atr_ratio)
        else:
            self._atr_ratio_min = min(self._atr_ratio_min, float(bar.atr_ratio))

    def _arm(self, bar: CompressionBarInput, reason: ArmReason) -> VexpCompressionSnapshot:
        assert self._cycle_id is not None
        assert self._start_timestamp is not None
        assert self._coil_high is not None and self._coil_low is not None
        assert self._atr_ratio_min is not None
        assert self._baseline_atr is not None
        self._phase = CompressionPhase.ARMED
        self._armed_bars = 0
        self._snapshot = VexpCompressionSnapshot(
            cycle_id=self._cycle_id,
            start_timestamp=self._start_timestamp,
            arm_timestamp=pd.Timestamp(bar.timestamp),
            coil_high=float(self._coil_high),
            coil_low=float(self._coil_low),
            bars_in_compression=int(self._bars_in_compression),
            atr_ratio_min=float(self._atr_ratio_min),
            atr_ratio_at_arm=float(bar.atr_ratio),
            baseline_atr=float(self._baseline_atr),
            arm_reason=reason,
        )
        return self._snapshot

    def _clear_cycle(self) -> None:
        self._phase = CompressionPhase.IDLE
        self._cycle_id = None
        self._start_timestamp = None
        self._coil_high = None
        self._coil_low = None
        self._bars_in_compression = 0
        self._atr_ratio_min = None
        self._baseline_atr = None
        self._snapshot = None
        self._armed_bars = 0

    def mark_consumed(self) -> None:
        self._clear_cycle()

    def update(self, bar: CompressionBarInput) -> CompressionUpdateResult:
        if bar.atr <= 0 or bar.baseline_atr <= 0 or pd.isna(bar.atr_ratio):
            return CompressionUpdateResult(
                phase=self._phase,
                event=None,
                snapshot=self._snapshot,
                bars_in_compression=self._bars_in_compression,
                meets_min_bars=self.meets_min_bars,
                is_armed=self._phase == CompressionPhase.ARMED,
                arm_reason=self._snapshot.arm_reason if self._snapshot else "NONE",
            )

        event: CompressionEvent | None = None
        arm_reason: ArmReason = "NONE"

        if self._phase == CompressionPhase.IDLE:
            if is_compression_ratio(bar.atr_ratio, self.config.compression_ratio_max):
                self._start_cycle(bar)
                event = CompressionEvent.ENTERED

        elif self._phase == CompressionPhase.COMPRESSING:
            compressed = is_compression_ratio(bar.atr_ratio, self.config.compression_ratio_max)
            if compressed:
                self._update_coil(bar)
            trigger = evaluate_arm_trigger(
                bar,
                arm_ratio_min=self.config.arm_ratio_min,
                expansion_mult=self.config.expansion_bar_range_mult,
            )
            if self.meets_min_bars and trigger != "NONE":
                self._arm(bar, trigger)
                event = CompressionEvent.ARMED
                arm_reason = trigger
            elif not compressed and trigger == "NONE":
                self._clear_cycle()
                event = CompressionEvent.RESET

        elif self._phase == CompressionPhase.ARMED:
            self._armed_bars += 1
            if self._armed_bars >= self.config.armed_timeout_bars:
                self._clear_cycle()
                event = CompressionEvent.EXPIRED

        return CompressionUpdateResult(
            phase=self._phase,
            event=event,
            snapshot=self._snapshot,
            bars_in_compression=self._bars_in_compression,
            meets_min_bars=self.meets_min_bars,
            is_armed=self._phase == CompressionPhase.ARMED,
            arm_reason=self._snapshot.arm_reason if self._snapshot else arm_reason,
        )


def bar_input_from_row(
    df: pd.DataFrame,
    bar_index: int,
    atr: float,
    baseline_atr: float,
    atr_ratio: float,
) -> CompressionBarInput:
    row = df.iloc[bar_index]
    return CompressionBarInput(
        bar_index=bar_index,
        timestamp=pd.Timestamp(row["datetime"]),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        atr=float(atr),
        baseline_atr=float(baseline_atr),
        atr_ratio=float(atr_ratio),
    )


def scan_compression_events(
    df: pd.DataFrame,
    atr_series: pd.Series,
    baseline_series: pd.Series,
    ratio_series: pd.Series,
    config: VexpConfig | None = None,
) -> list[tuple[int, CompressionUpdateResult]]:
    cfg = config or load_vexp_config()
    fsm = CompressionFSM(config=cfg)
    armed: list[tuple[int, CompressionUpdateResult]] = []
    for bar_index in range(len(df)):
        atr = float(atr_series.iloc[bar_index]) if bar_index < len(atr_series) else float("nan")
        baseline = float(baseline_series.iloc[bar_index]) if bar_index < len(baseline_series) else float("nan")
        ratio = float(ratio_series.iloc[bar_index]) if bar_index < len(ratio_series) else float("nan")
        if pd.isna(atr) or pd.isna(baseline) or pd.isna(ratio):
            continue
        bar = bar_input_from_row(df, bar_index, atr, baseline, ratio)
        result = fsm.update(bar)
        if result.event == CompressionEvent.ARMED and result.snapshot is not None:
            armed.append((bar_index, result))
    return armed


# ---------------------------------------------------------------------------
# Breakout
#
# LONG: close > coil_high AND close > open; SHORT: close < coil_low AND close < open.
# ---------------------------------------------------------------------------


def score_input_from_bar(
    bar: CompressionBarInput,
    snapshot: VexpCompressionSnapshot,
    direction: TradeDirection,
    *,
    htf_aligned: bool = False,
) -> VexpScoreInput:
    return VexpScoreInput(
        direction=direction,
        close=float(bar.close),
        open=float(bar.open),
        high=float(bar.high),
        low=float(bar.low),
        atr=float(bar.atr),
        coil_high=float(snapshot.coil_high),
        coil_low=float(snapshot.coil_low),
        atr_ratio_at_arm=float(snapshot.atr_ratio_at_arm),
        atr_ratio_min=float(snapshot.atr_ratio_min),
        bars_in_compression=int(snapshot.bars_in_compression),
        htf_aligned=htf_aligned,
        expansion_bar=is_expansion_bar(bar),
    )


def detect_breakout_direction(
    bar: CompressionBarInput,
    snapshot: VexpCompressionSnapshot,
) -> TradeDirection | None:
    buy_inp = score_input_from_bar(bar, snapshot, "BUY")
    sell_inp = score_input_from_bar(bar, snapshot, "SELL")
    buy_ok = is_valid_breakout_candle(buy_inp)
    sell_ok = is_valid_breakout_candle(sell_inp)
    if buy_ok and sell_ok:
        buy_pen = breakout_edge_distance(buy_inp)
        sell_pen = breakout_edge_distance(sell_inp)
        return "BUY" if buy_pen >= sell_pen else "SELL"
    if buy_ok:
        return "BUY"
    if sell_ok:
        return "SELL"
    return None


def breakout_strength_ratio(
    bar: CompressionBarInput,
    snapshot: VexpCompressionSnapshot,
    direction: TradeDirection,
) -> float:
    inp = score_input_from_bar(bar, snapshot, direction)
    atr = float(bar.atr)
    if atr <= 0:
        return 0.0
    return breakout_edge_distance(inp) / atr


# ---------------------------------------------------------------------------
# Setup detector
#
# Pipeline: build_atr_context → CompressionFSM per bar → session on breakout bar
# → detect_breakout_direction → HTF filter → candidate_score → build_vexp_setup
# ---------------------------------------------------------------------------


def _ensure_bars(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    work = df.sort_values("datetime").copy()
    work["datetime"] = pd.to_datetime(work["datetime"])
    work["hour"] = work["datetime"].dt.hour
    work["date"] = work["datetime"].dt.date
    return work.reset_index(drop=True)


def _session_hours(session_date: date, config: VexpConfig) -> range:
    return shift_hour_range(
        session_date,
        config.session_hour_start,
        config.session_hour_end,
        dst_type=DATA_DST_TYPE,
    )


def is_vexp_session_bar(bar: CompressionBarInput, config: VexpConfig) -> bool:
    """Breakout bar must fall in London session. ARMED may occur off-session."""
    ts = pd.Timestamp(bar.timestamp)
    hours = _session_hours(ts.date(), config)
    return int(ts.hour) in hours


def resolve_session_label(hour: int) -> str:
    if 15 <= hour <= 20:
        return "LONDON"
    return "OFF_SESSION"


def build_risk_levels(
    direction: str,
    entry: float,
    coil_high: float,
    coil_low: float,
    atr: float,
    *,
    sl_buffer_ratio: float = SL_ATR_BUFFER_RATIO,
    risk_reward: float = DEFAULT_RISK_REWARD,
) -> tuple[float, float] | None:
    buffer = sl_buffer_ratio * atr
    if direction == "BUY":
        stop_loss = float(coil_low) - buffer
        risk = entry - stop_loss
        if risk <= 0:
            return None
        return stop_loss, entry + risk_reward * risk
    stop_loss = float(coil_high) + buffer
    risk = stop_loss - entry
    if risk <= 0:
        return None
    return stop_loss, entry - risk_reward * risk


def build_vexp_setup(
    bar: CompressionBarInput,
    snapshot: VexpCompressionSnapshot,
    direction: TradeDirection,
    *,
    pair: str,
    candidate_score: float,
    h1_trend: str,
    htf_aligned: bool,
    pip_size: float,
    config: VexpConfig,
) -> VexpSetup | None:
    entry = float(bar.close)
    levels = build_risk_levels(
        direction,
        entry,
        snapshot.coil_high,
        snapshot.coil_low,
        float(bar.atr),
        sl_buffer_ratio=SL_ATR_BUFFER_RATIO,
        risk_reward=config.risk_reward,
    )
    if levels is None:
        return None
    stop_loss, take_profit = levels
    coil_width_pips = abs(float(snapshot.coil_high) - float(snapshot.coil_low)) / pip_size
    strength = breakout_strength_ratio(bar, snapshot, direction)
    score_inp = score_input_from_bar(bar, snapshot, direction, htf_aligned=htf_aligned)
    return VexpSetup(
        timestamp=pd.Timestamp(bar.timestamp),
        pair=pair,
        direction=direction,
        bar_index=int(bar.bar_index),
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_reward=float(config.risk_reward),
        atr=float(bar.atr),
        atr_ratio=float(bar.atr_ratio),
        coil_high=float(snapshot.coil_high),
        coil_low=float(snapshot.coil_low),
        coil_width_pips=round(coil_width_pips, 2),
        breakout_strength=round(strength, 4),
        cycle_id=str(snapshot.cycle_id),
        bars_in_compression=int(snapshot.bars_in_compression),
        atr_ratio_min=float(snapshot.atr_ratio_min),
        arm_reason=snapshot.arm_reason,
        candidate_score=round(candidate_score, 2),
        h1_trend=h1_trend,
        session=resolve_session_label(int(pd.Timestamp(bar.timestamp).hour)),
        htf_aligned=htf_aligned,
        expansion_bar=score_inp.expansion_bar,
        sweep_distance=abs(float(snapshot.coil_high) - float(snapshot.coil_low)),
    )


def detect_vexp_setups(
    df: pd.DataFrame,
    pair_name: str,
    h1_df: pd.DataFrame | None = None,
    *,
    config: VexpConfig | None = None,
    progress_hook: ProgressHook | None = None,
) -> list[VexpSetup]:
    if pair_name.upper() not in ALLOWED_PAIRS:
        return []

    cfg = config or load_vexp_config()
    work = _ensure_bars(df)
    if len(work) < cfg.min_compression_bars + cfg.atr_period:
        return []

    atr_series, baseline_series, ratio_series = build_atr_context(
        work,
        period=cfg.atr_period,
        baseline_days=cfg.baseline_lookback_days,
    )
    fsm = CompressionFSM(config=cfg)
    pip_size = pip_size_for_pair(pair_name)
    structure = h1_df if h1_df is not None else work
    htf_lookup = build_htf_trend_lookup(structure)
    setups: list[VexpSetup] = []

    scan_total = len(work)
    for bar_index in range(scan_total):
        if progress_hook is not None:
            progress_hook(bar_index, scan_total)

        atr = float(atr_series.iloc[bar_index]) if bar_index < len(atr_series) else float("nan")
        baseline = float(baseline_series.iloc[bar_index]) if bar_index < len(baseline_series) else float("nan")
        ratio = float(ratio_series.iloc[bar_index]) if bar_index < len(ratio_series) else float("nan")
        if pd.isna(atr) or pd.isna(baseline) or pd.isna(ratio) or atr <= 0:
            continue

        bar = bar_input_from_row(work, bar_index, atr, baseline, ratio)
        fsm.update(bar)

        if fsm.phase != CompressionPhase.ARMED or fsm.snapshot is None:
            continue
        if not is_vexp_session_bar(bar, cfg):
            continue

        direction = detect_breakout_direction(bar, fsm.snapshot)
        if direction is None:
            continue

        htf = htf_lookup.at(bar.timestamp)
        if is_counter_trend(direction, htf.direction):
            continue

        htf_aligned = htf.direction != "NEUTRAL" and not is_counter_trend(direction, htf.direction)
        score_inp = score_input_from_bar(bar, fsm.snapshot, direction, htf_aligned=htf_aligned)
        breakdown = calc_vexp_candidate_score(score_inp)
        if not passes_vexp_l2_gate(breakdown.total):
            continue

        setup = build_vexp_setup(
            bar,
            fsm.snapshot,
            direction,
            pair=pair_name.upper(),
            candidate_score=breakdown.total,
            h1_trend=htf.direction,
            htf_aligned=htf_aligned,
            pip_size=pip_size,
            config=cfg,
        )
        if setup is None:
            continue

        setups.append(setup)
        fsm.mark_consumed()

    return setups


def detect_vexp_setups_for_pair(
    df: pd.DataFrame,
    pair_name: str,
    h1_df: pd.DataFrame | None = None,
    progress_hook: ProgressHook | None = None,
) -> list[VexpSetup]:
    return detect_vexp_setups(df, pair_name, h1_df, progress_hook=progress_hook)


# ---------------------------------------------------------------------------
# BaseStrategy adapter
# ---------------------------------------------------------------------------


def _bar_from_setup(setup: VexpSetup) -> CompressionBarInput:
    return CompressionBarInput(
        bar_index=setup.bar_index,
        timestamp=setup.timestamp,
        open=setup.entry_price,
        high=max(setup.entry_price, setup.coil_high),
        low=min(setup.entry_price, setup.coil_low),
        close=setup.entry_price,
        atr=setup.atr,
        baseline_atr=setup.atr / setup.atr_ratio if setup.atr_ratio > 0 else setup.atr,
        atr_ratio=setup.atr_ratio,
    )


def _snapshot_from_setup(setup: VexpSetup) -> VexpCompressionSnapshot:
    return VexpCompressionSnapshot(
        cycle_id=setup.cycle_id,
        start_timestamp=setup.timestamp,
        arm_timestamp=setup.timestamp,
        coil_high=setup.coil_high,
        coil_low=setup.coil_low,
        bars_in_compression=setup.bars_in_compression,
        atr_ratio_min=setup.atr_ratio_min,
        atr_ratio_at_arm=setup.atr_ratio,
        baseline_atr=setup.atr / setup.atr_ratio if setup.atr_ratio > 0 else setup.atr,
        arm_reason=setup.arm_reason,
    )


class VexpStrategy(BaseStrategy):
    """Volatility Expansion — ATR coil → first impulse breakout."""

    def __init__(self, weights_config: dict[str, int] | None = None, mode_h1: bool = False):
        super().__init__(weights_config=weights_config, mode_h1=mode_h1)
        self.config = load_vexp_config()
        self._htf_gbp: pd.DataFrame | None = None
        self._htf_eur: pd.DataFrame | None = None

    @property
    def setup_type(self) -> str:
        return SETUP_TYPE

    def detect_setups(
        self,
        df: pd.DataFrame,
        pair_name: str,
        h1_df: pd.DataFrame | None = None,
    ) -> list[VexpSetup]:
        return detect_vexp_setups(df, pair_name, h1_df, config=self.config)

    def analyze_setup(
        self,
        setup: VexpSetup,
        gbp_setup: VexpSetup | None,
        eur_setup: VexpSetup | None,
        h1_gbp: pd.DataFrame,
        h1_eur: pd.DataFrame,
    ) -> StrategyResult:
        h1_ref = h1_gbp if setup.pair in ("GBPUSD", "AUDUSD", "AUDJPY") else h1_eur
        htf_df = self._htf_gbp if setup.pair in ("GBPUSD", "AUDUSD", "AUDJPY") else self._htf_eur
        htf_result = analyze_htf_trend(h1_ref, setup.timestamp, htf_df=htf_df)
        htf_trend_direction = htf_result.direction
        htf_mismatch = is_counter_trend(setup.direction, htf_trend_direction)

        smt_feats = calc_smt_features(gbp_setup, eur_setup, pip_size=pip_size_for_pair(setup.pair))
        bar_inp = score_input_from_bar(
            _bar_from_setup(setup),
            _snapshot_from_setup(setup),
            setup.direction,
            htf_aligned=setup.htf_aligned,
        )
        breakdown = calc_vexp_candidate_score(bar_inp)
        candidate_score = breakdown.total

        raw_features: dict[str, Any] = {
            "smt_intensity": smt_feats.intensity,
            "smt_diff": smt_feats.diff,
            "smt_leader": smt_feats.leader,
            "wick_ratio_pct": 0.0,
            "atr_ratio": round(setup.atr_ratio, 4),
            "has_bos": True,
            "both_sweep": gbp_setup is not None and eur_setup is not None,
            "htf_trend_direction": htf_trend_direction,
            "vexp_cycle_id": setup.cycle_id,
            "vexp_coil_width_pips": setup.coil_width_pips,
            "vexp_breakout_strength": setup.breakout_strength,
            "vexp_bars_in_compression": setup.bars_in_compression,
            "vexp_arm_reason": setup.arm_reason,
            "vexp_session": setup.session,
            "l4_bypass": True,
            "htf_bypass": False,
            "htf_would_block": htf_mismatch,
            "htf_trend_hard_filter": True,
            "reject_reason": "REJECT_BY_HTF_TREND" if htf_mismatch else "",
            "reason_codes": [HTF_TREND_MISMATCH_TAG] if htf_mismatch else [],
            "candidate_score": candidate_score,
            "l2_min_score": VEXP_L2_MIN_CANDIDATE_SCORE,
        }

        return StrategyResult(
            is_setup=True,
            setup_type=self.setup_type,
            direction=setup.direction,
            entry_price=setup.entry_price,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            candidate_score=candidate_score,
            raw_features=raw_features,
        )

    def evaluate(self, payload: dict, state: dict) -> StrategyResult:
        active: VexpSetup | None = state.get("active_setup")
        if active is None:
            return StrategyResult(is_setup=False, setup_type=self.setup_type, direction="")
        return self.analyze_setup(
            active,
            state.get("gbp_setup"),
            state.get("eur_setup"),
            state["h1_gbp"],
            state["h1_eur"],
        )


__all__ = [
    "ALLOWED_PAIRS",
    "ARMED_TIMEOUT_BARS",
    "ArmReason",
    "CompressionBarInput",
    "CompressionEvent",
    "CompressionFSM",
    "CompressionPhase",
    "CompressionStateView",
    "CompressionUpdateResult",
    "DEFAULT_RISK_REWARD",
    "MAX_HOLDING_BARS",
    "SETUP_TYPE",
    "VEXP_L2_MIN_CANDIDATE_SCORE",
    "VEXP_PAIR_PRIMARY",
    "VEXP_PAIR_SECONDARY",
    "VEXP_VERSION",
    "VexpCompressionSnapshot",
    "VexpConfig",
    "VexpScoreBreakdown",
    "VexpScoreInput",
    "VexpSetup",
    "VexpStrategy",
    "bar_input_from_row",
    "build_atr_context",
    "build_atr_series",
    "build_risk_levels",
    "build_vexp_setup",
    "calc_vexp_candidate_score",
    "calc_vexp_candidate_score_total",
    "detect_breakout_direction",
    "detect_vexp_setups",
    "detect_vexp_setups_for_pair",
    "evaluate_arm_trigger",
    "is_compression_ratio",
    "is_expansion_bar",
    "is_valid_breakout_candle",
    "is_vexp_session_bar",
    "load_vexp_config",
    "passes_vexp_l2_gate",
    "scan_compression_events",
    "score_breakout_strength",
    "score_compression_depth",
    "score_input_from_bar",
]
