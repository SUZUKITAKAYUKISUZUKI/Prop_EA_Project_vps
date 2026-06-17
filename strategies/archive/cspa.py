"""
CSPA — Candlestick Price Action (旧 Strategy B / ローソク足 × 半値戻し × 波の角)

アーカイブ: 2026-06-01 — 検証の結果、プロップ用ポートフォリオには向いていない。
（WFT 採用時: 2026-06-09 — IS12/OOS3/step3 平均 OOS +18.88R, DD 超過 0%）

Methodology source: docs/candlestick.pdf

MTF (fixed):
  - bias_tf      = H1  — Dow trend / 方向感
  - structure_tf = M15 — impulse, 半値戻し, 修正波リズム, レンジ除外
  - trigger_tf   = M1  — 停滞クラスタ → モメンタム確定足

Pairs: EURUSD / GBPUSD

Fibonacci / 押し戻し深さの設計思想
------------------------------------
CSPA はフィボナッチ比率を「価格に意味がある水準」として信仰するのではなく、
**押し戻りの深さを数値化する仮の物差し**としてのみ用いる。

  - 連続値 ``retrace_ratio``（= インパルスに対して何 % 戻したか）を算出し、
    Bayes 学習用特徴量 ``pullback_depth`` として記録する。
  - 38.2–61.8% 帯（``FIB_RETRACE_MIN`` / ``FIB_RETRACE_MAX``）は L2 の**粗いゲート**
    および L2 スコア（``score_cspa_retrace_beauty``）の便宜上の参照帯であり、
    最終的な優劣判断はベイズモデルが行う（CSPA は L3.5 Bayes バイパス中だが、
    学習データ収集後に有効化する想定）。
  - フィボナッチを出発点としつつ、Bayes がデータから最適な物差しを自律的に
    発見する — 「固定のフィボ水準に依存したエントリー」ではない。

要約: フィボナッチ = 連続特徴量のスケール定義 + L2 粗フィルター。
      フィボナッチ信仰 = 排除。

L3.5 ベイズ意思決定（有効化後）
---------------------------------
CSPA 専用 ``CSPABayesEngine``（``audit.cspa_bayes_gate.evaluate_cspa_bayes_gate``）による
3-Tier 多次元動的ゲート::

    Tier 1: reaccel_follow_through × reacceleration_score decile → REJECT
    Tier 2: session × ATR レジーム → ベース lot/tp
    Tier 3: rhythm / market_breath / breakout_velocity → 動的補正

現状: Pure BT（``CSPA_PURE_BT=1``）では L3.5 バイパス（特徴量 + ラベル収集）。

CSPA v2.0 — 市場の「呼吸」
----------------------------
資料 (candlestick.md) の本質は構造だけでなく **方向感 → 勢い → 修正波リズム → 停滞 → 再加速**
の連鎖。v2 では L2 を次の 2 層に分離する::

    entry_score = structure_score × 0.4 + market_breath_score × 0.6

  - **structure_score** — 半値戻しの美しさ、鮮度、H1 方向整合（骨格）
  - **market_breath_score** — 観測可能な 3 軸（Bayes 分析の主対象）:
      ``stagnation_quality``, ``rhythm_score``, ``reacceleration_score``

設計原則: 裁量トレーダーの曖昧な感覚をそのままコード化しない。
各軸は compression / range_decay / wick_balance 等の**測定可能な分解指標**の加重合成のみ。

利確は固定 2R ではなく、M15 過去停滞帯（次の consolidation zone）を優先ターゲットとする。

建値 / トレーリング（L5 追跡）
--------------------------------
原著「角から伸びて戻してきたら建値決済」を L5 シャドー追跡に反映する。

  - **建値ストップ**: MFE が ``CSPA_BE_TRIGGER_MFE_R`` に達するか、一度伸びた後
    終値が ``CSPA_BE_PULLBACK_CLOSE_R`` まで戻ったら SL をエントリーへ移動
  - **トレーリング**: 建値移動後、ピーク favorable ± ``CSPA_TRAIL_ATR_MULT`` × ATR で ratchet

``CSPA_BE_ENABLED=0`` / ``CSPA_TRAIL_ENABLED=0`` で従来の固定 SL に戻せる。
"""

from __future__ import annotations

import bisect
import logging
import os
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from strategies.base import StrategyResult
from strategies.base_strategy import BaseStrategy
from strategies.htf_trend_analyzer import analyze_htf_trend, build_htf_trend_lookup, is_counter_trend
from strategies.market_utils import calc_smt_features, compute_atr, pip_size_for_pair
from strategies.session_dst import DATA_DST_TYPE, shift_hour, shift_hour_range
from volume_profile_analyzer import (
    DEFAULT_VP_BUFFER_PIPS,
    DEFAULT_VP_LOCATION_SCORE_TIERS,
    SessionVolumeProfile,
    VolumeProfileLevels,
    VpLocationScoreTiers,
)

logger = logging.getLogger("cspa")
CSPA_SCAN_HEARTBEAT_EVERY = int(os.getenv("CSPA_SCAN_HEARTBEAT_EVERY", "5000"))

SETUP_TYPE = "CSPA"
STRATEGY_ABBREV = "CSPA"
STRATEGY_FULL_NAME = "Candlestick Price Action"

CSPA_PAIR_PRIMARY = "GBPUSD"
CSPA_PAIR_SECONDARY = "EURUSD"
ALLOWED_PAIRS = frozenset({CSPA_PAIR_PRIMARY, CSPA_PAIR_SECONDARY})

CSPA_BIAS_TF = "H1"
CSPA_STRUCTURE_TF = "M15"
CSPA_TRIGGER_TF = "M1"

SWING_LOOKBACK_BIAS = 3
SWING_LOOKBACK_STRUCTURE = 3
TRIGGER_WARMUP_BARS = 240

# 押し戻し深さの L2 粗フィルター帯（Fib 名は物差しの慣習的ラベルのみ。
# 38.2/61.8% に価格的な意味は置かない。連続値は pullback_depth として Bayes へ）。
FIB_RETRACE_MIN = float(os.getenv("CSPA_FIB_RETRACE_MIN", "0.382"))
FIB_RETRACE_MAX = float(os.getenv("CSPA_FIB_RETRACE_MAX", "0.618"))
FIB_RETRACE_IDEAL = 0.5  # L2 retrace_beauty の中心。Bayes 特徴量のターゲットではない。

STAGNATION_MAX_BODY_ATR = float(os.getenv("CSPA_STAGNATION_MAX_BODY_ATR", "0.35"))
STAGNATION_MIN_BARS = int(os.getenv("CSPA_STAGNATION_MIN_BARS", "1"))
STAGNATION_MAX_BARS = int(os.getenv("CSPA_STAGNATION_MAX_BARS", "4"))

MOMENTUM_MIN_BODY_ATR = float(os.getenv("CSPA_MOMENTUM_MIN_BODY_ATR", "0.45"))
CORRECTION_RHYTHM_MAX_RATIO = float(os.getenv("CSPA_CORRECTION_RHYTHM_MAX_RATIO", "1.35"))

ATR_PERIOD = 14
SL_ATR_BUFFER_RATIO = 0.3
DEFAULT_RISK_REWARD = 2.0
MAX_BARS_SINCE_IMPULSE = 32

CSPA_V2_SCORING = os.getenv("CSPA_V2_SCORING", "1").strip().lower() in ("1", "true", "yes", "on")
CSPA_V2_STRUCTURE_WEIGHT = float(os.getenv("CSPA_V2_STRUCTURE_WEIGHT", "0.4"))
CSPA_V2_BREATH_WEIGHT = float(os.getenv("CSPA_V2_BREATH_WEIGHT", "0.6"))
CSPA_CONSOLIDATION_TP = os.getenv("CSPA_CONSOLIDATION_TP", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
CSPA_CONSOLIDATION_LOOKBACK = int(os.getenv("CSPA_CONSOLIDATION_LOOKBACK", "75"))
CSPA_MIN_RR = float(os.getenv("CSPA_MIN_RR", "1.2"))
CSPA_MAX_RR = float(os.getenv("CSPA_MAX_RR", "4.5"))
CSPA_M15_STAGNATION_MAX_BODY_ATR = float(os.getenv("CSPA_M15_STAGNATION_MAX_BODY_ATR", "0.45"))
CSPA_TREND_MATURE_BARS = int(os.getenv("CSPA_TREND_MATURE_BARS", "24"))
CSPA_COMPRESSION_IDEAL = float(os.getenv("CSPA_COMPRESSION_IDEAL", "0.3"))
CSPA_COMPRESSION_BAD = float(os.getenv("CSPA_COMPRESSION_BAD", "0.8"))
CSPA_CONSOLIDATION_WINDOW = int(os.getenv("CSPA_CONSOLIDATION_WINDOW", "10"))
CSPA_CONSOLIDATION_WIDTH_ATR = float(os.getenv("CSPA_CONSOLIDATION_WIDTH_ATR", "1.5"))
CSPA_REACCEL_IMBALANCE_LOOKBACK = int(os.getenv("CSPA_REACCEL_IMBALANCE_LOOKBACK", "5"))

CSPA_L2_MIN_SCORE = float(os.getenv("CSPA_L2_MIN_SCORE", "65"))

# --- L2 candidate_score weights (v1 legacy; v2 uses structure/breath blend) ---
CSPA_SCORE_BIAS_MAX = 20.0
CSPA_SCORE_RETRACE_MAX = 25.0
CSPA_SCORE_STAGNATION_MAX = 20.0
CSPA_SCORE_MOMENTUM_MAX = 20.0
CSPA_SCORE_FRESHNESS_MAX = 10.0
CSPA_SCORE_RHYTHM_MAX = 20.0  # v2: 5 → 20（修正波リズム格上げ）
CSPA_BREATH_COMPONENT_MAX = 25.0  # breath 内 4 成分各 25 点

CSPA_RETRACE_BEAUTY_HALF_WIDTH = FIB_RETRACE_IDEAL - FIB_RETRACE_MIN
CSPA_M15_FRESHNESS_PEAK_BARS = (2, 10)
CSPA_MOMENTUM_BODY_ATR_FULL = 0.80
CSPA_BT_SPREAD_PIPS = float(os.getenv("CSPA_BT_SPREAD_PIPS", "1.2"))
CSPA_SPREAD_REF_PIPS = float(os.getenv("CSPA_SPREAD_REF_PIPS", "2.0"))

# FX_logic.md — SWEEP_ENGULFING 専用（他 momentum / CSPA 構造は変更しない）
CSPA_FX_SWEEP_SL = os.getenv("CSPA_FX_SWEEP_SL", "1").strip().lower() in ("1", "true", "yes", "on")
CSPA_FX_SWEEP_SL_BUFFER_PIPS = float(os.getenv("CSPA_FX_SWEEP_SL_BUFFER_PIPS", "0.8"))
CSPA_FX_SWEEP_MIN_RISK_PIPS = float(os.getenv("CSPA_FX_SWEEP_MIN_RISK_PIPS", "0"))
CSPA_FX_SWEEP_MIN_RANGE_ATR = float(os.getenv("CSPA_FX_SWEEP_MIN_RANGE_ATR", "0.5"))
CSPA_FX_SWEEP_MIN_OUTSIDE_RATIO = float(os.getenv("CSPA_FX_SWEEP_MIN_OUTSIDE_RATIO", "0.85"))
CSPA_FX_SWEEP_RR = os.getenv("CSPA_FX_SWEEP_RR", "").strip()

# SWEEP_ENGULFING BUY — セッション Volume Profile (VAL) ゲート
CSPA_VP_SWEEP_FILTER = os.getenv("CSPA_VP_SWEEP_FILTER", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
CSPA_VP_BUFFER_PIPS = float(os.getenv("CSPA_VP_BUFFER_PIPS", str(DEFAULT_VP_BUFFER_PIPS)))
CSPA_VP_BUFFER_ATR_MULT = float(os.getenv("CSPA_VP_BUFFER_ATR_MULT", "0.1"))
CSPA_VP_SCORE_TIERS = VpLocationScoreTiers(
    sweep_zone=int(os.getenv("CSPA_VP_SCORE_SWEEP", "30")),
    favorable=int(os.getenv("CSPA_VP_SCORE_FAVORABLE", "10")),
    neutral=int(os.getenv("CSPA_VP_SCORE_NEUTRAL", "0")),
    adverse=int(os.getenv("CSPA_VP_SCORE_ADVERSE", "-20")),
)

# --- 建値 / トレーリング（原著: 伸びて戻したら建値決済）---
CSPA_BE_ENABLED = os.getenv("CSPA_BE_ENABLED", "1").strip().lower() in ("1", "true", "yes", "on")
CSPA_TRAIL_ENABLED = os.getenv("CSPA_TRAIL_ENABLED", "1").strip().lower() in ("1", "true", "yes", "on")
CSPA_BE_ARM_MFE_R = float(os.getenv("CSPA_BE_ARM_MFE_R", "0.35"))
CSPA_BE_TRIGGER_MFE_R = float(os.getenv("CSPA_BE_TRIGGER_MFE_R", "0.5"))
CSPA_BE_PULLBACK_CLOSE_R = float(os.getenv("CSPA_BE_PULLBACK_CLOSE_R", "0.08"))
CSPA_BE_RHYTHM_MAX_BARS = int(os.getenv("CSPA_BE_RHYTHM_MAX_BARS", "12"))
CSPA_TRAIL_ATR_MULT = float(os.getenv("CSPA_TRAIL_ATR_MULT", "1.0"))
CSPA_BE_BUFFER_ATR = float(os.getenv("CSPA_BE_BUFFER_ATR", "0.0"))


def is_cspa_pure_bt_mode() -> bool:
    """Bayes 特徴量収集用: エントリーゲート（Fib/L2/HTF 等）を無効化するピュア BT。"""
    return os.getenv("CSPA_PURE_BT", "0").strip().lower() in ("1", "true", "yes", "on")


def resolve_cspa_pure_mode(pure_bt: bool | None = None) -> bool:
    """スキャンと evaluate の pure 判定を env / 引数で統一。"""
    if is_cspa_pure_bt_mode():
        return True
    if pure_bt is True:
        return True
    if pure_bt is False:
        return False
    return False


CSPA_DOW_EARLY_BARS = int(os.getenv("CSPA_DOW_EARLY_BARS", "8"))
CSPA_ADR_LOOKBACK_DAYS = int(os.getenv("CSPA_ADR_LOOKBACK_DAYS", "14"))
CSPA_VOLATILITY_LOOKBACK_BARS = int(os.getenv("CSPA_VOLATILITY_LOOKBACK_BARS", "120"))
CSPA_VOLUME_LOOKBACK_BARS = int(os.getenv("CSPA_VOLUME_LOOKBACK_BARS", "20"))
CSPA_ASIA_HOUR_START = 0
CSPA_ASIA_HOUR_END = 14
CSPA_LONDON_HOUR_START = 15
CSPA_LONDON_HOUR_END = 20
CSPA_NY_HOUR_START = 21
CSPA_NY_HOUR_END = 23
# 後方互換（非 DST パス）。セッション判定は session_dst 経由を使用。
CSPA_LONDON_HOURS = range(CSPA_LONDON_HOUR_START, CSPA_LONDON_HOUR_END + 1)
CSPA_NY_HOURS = range(CSPA_NY_HOUR_START, CSPA_NY_HOUR_END + 1)

DowPhaseLabel = Literal["EARLY_TREND", "MATURE_TREND", "RANGE"]
SessionType = Literal["ASIA", "LONDON", "NY", "OFF_HOURS"]
OutcomeLabel = Literal["WIN", "LOSS"]

CSPA_SESSION_OPEN_HOUR: dict[SessionType, int] = {
    "ASIA": CSPA_ASIA_HOUR_START,
    "LONDON": CSPA_LONDON_HOUR_START,
    "NY": CSPA_NY_HOUR_START,
    "OFF_HOURS": 0,
}

# Bayes 学習ログ専用。L2/L3.5 判断には未使用。ブローカー依存のため推論時は低ウェイト推奨。
CSPA_BAYES_LOW_WEIGHT_FEATURES: tuple[str, ...] = ("tick_volume_ratio",)

CSPA_BAYES_FEATURE_COLUMNS: tuple[str, ...] = (
    "trade_id",
    "timestamp",
    "pair",
    "direction",
    "decision_source",
    "executed",
    # 構造
    "dow_phase",
    "trend_age_bars",
    "pullback_depth",
    "pullback_duration",
    "impulse_quality",
    "impulse_atr_ratio",
    # 停滞
    "stagnation_duration",
    "stagnation_pips_width",
    "stagnation_compression_ratio",
    "stagnation_wick_balance",
    "stagnation_range_decay_rate",
    "stagnation_quality_score",
    # 修正波リズム (v2)
    "pullback_overlap_ratio",
    "pullback_efficiency",
    "correction_smoothness",
    "rhythm_score",
    "correction_velocity",
    # ブレイク / 再加速
    "breakout_momentum_ratio",
    "breakout_velocity",
    "wick_ratio",
    "reaccel_follow_through",
    "reaccel_candle_imbalance",
    "reacceleration_score",
    "h1_momentum_score",
    # TP
    "tp_mode",
    "tp_rr_actual",
    # v2 合成
    "structure_score",
    "market_breath_score",
    # ボラティリティ
    "current_atr_h1",
    "volatility_percentile",
    # セッション
    "session_type",
    "minutes_from_session_open",
    # ADR
    "adr_used",
    "adr_remaining",
    # 流動性
    "distance_daily_high",
    "distance_daily_low",
    "distance_session_high",
    "distance_session_low",
    # コスト
    "spread",
    "spread_percentile",
    # 出来高
    "tick_volume_ratio",
    "vp_location_score",
    # ラベル
    "outcome_label",
    "result_r",
    "mfe",
    "mae",
    "candidate_score",
    "bayes_probability",
)
TradeDirection = Literal["BUY", "SELL"]
MomentumType = Literal["BODY_BREAK", "ENGULFING", "PIN_BAR", "SWEEP_ENGULFING", "NONE"]
VolatilityRegime = Literal["LOW", "NORMAL", "HIGH"]
TpMode = Literal["CONSOLIDATION", "FIXED_RR"]


@dataclass(frozen=True)
class SwingPoint:
    bar_index: int
    timestamp: pd.Timestamp
    price: float
    kind: Literal["HIGH", "LOW"]


@dataclass(frozen=True)
class ImpulseLeg:
    direction: Literal["UP", "DOWN"]
    start_index: int
    end_index: int
    start_price: float
    end_price: float
    impulse_size: float


@dataclass(frozen=True)
class StagnationCluster:
    start_index: int
    end_index: int
    bar_count: int
    avg_body_atr: float
    zone_high: float
    zone_low: float
    solid_ground: bool


@dataclass(frozen=True)
class MomentumSignal:
    detected: bool
    trigger_type: MomentumType
    bar_index: int
    timestamp: pd.Timestamp
    entry_price: float
    trigger_high: float
    trigger_low: float
    body_atr: float
    atr: float


@dataclass(frozen=True)
class TrendContext:
    """H1 方向感 — Dow 構造 + 勢い + トレンド寿命 + ボラレジーム。"""

    direction: TrendPhase
    momentum_score: float
    trend_age_bars: int
    volatility_regime: VolatilityRegime


@dataclass(frozen=True)
class PullbackRhythm:
    """修正波リズム — overlap / efficiency / smoothness の観測値 + 合成。"""

    duration_bars: int
    retracement_depth: float
    overlap_ratio: float
    pullback_efficiency: float
    correction_smoothness: float
    rhythm_score: float


@dataclass(frozen=True)
class StagnationQuality:
    """M1 停滞品質 — compression / range_decay / wick_balance の観測値 + 合成。"""

    compression_ratio: float
    wick_balance: float
    range_decay_rate: float
    composite_score: float


@dataclass(frozen=True)
class Reacceleration:
    """再加速 — breakout_velocity / follow_through / candle_imbalance + 合成。"""

    breakout_velocity: float
    follow_through: float
    candle_imbalance: float
    composite_score: float


@dataclass(frozen=True)
class ConsolidationZone:
    """M15 停滞帯ノード。"""

    bar_start: int
    bar_end: int
    zone_high: float
    zone_low: float
    zone_mid: float


@dataclass(frozen=True)
class CspaScoreInput:
    """Deterministic L2 inputs assembled at setup detection time."""

    trade_direction: TradeDirection
    h1_trend: str
    retrace_ratio: float
    prior_retrace_ratio: float | None
    stagnation_bars: int
    stagnation_avg_body_atr: float
    stagnation_solid_ground: bool
    momentum_type: MomentumType
    momentum_body_atr: float
    m15_bars_since_impulse: int
    impulse_size_atr: float
    trend_context: TrendContext | None = None
    pullback_rhythm: PullbackRhythm | None = None
    stagnation_quality: StagnationQuality | None = None
    reacceleration: Reacceleration | None = None
    vp_location_score: int = 0


@dataclass(frozen=True)
class CspaScoreBreakdown:
    bias_alignment: float
    retrace_beauty: float
    stagnation: float
    momentum: float
    structure_freshness: float
    correction_rhythm: float
    structure_score: float
    market_breath_score: float
    trend_context_score: float
    stagnation_quality_score: float
    reacceleration_score: float
    vp_location_score: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return {
            "bias_alignment": self.bias_alignment,
            "retrace_beauty": self.retrace_beauty,
            "stagnation": self.stagnation,
            "momentum": self.momentum,
            "structure_freshness": self.structure_freshness,
            "correction_rhythm": self.correction_rhythm,
            "structure_score": self.structure_score,
            "market_breath_score": self.market_breath_score,
            "trend_context_score": self.trend_context_score,
            "stagnation_quality_score": self.stagnation_quality_score,
            "reacceleration_score": self.reacceleration_score,
            "vp_location_score": self.vp_location_score,
            "total": self.total,
        }


@dataclass(frozen=True)
class CspaBayesFeatures:
    """CSPA 専用ベイズ推定モデル構築用の事前特徴量（エントリー時点）。

    pullback_depth: インパルスに対する押し戻し率（0–1 連続値）。
        Fib 比率名は物差しの出発点のみ。Bayes が最適な閾値・形状を学習する。
    distance_daily_high/low: H1 ATR 正規化。
    tick_volume_ratio: ブローカー依存のため Bayes 学習ログのみ（L2/L3.5 未使用）。
    """

    dow_phase: DowPhaseLabel
    trend_age_bars: int
    pullback_depth: float
    pullback_duration: int
    impulse_quality: float
    impulse_atr_ratio: float
    stagnation_duration: int
    stagnation_pips_width: float
    stagnation_compression_ratio: float
    stagnation_wick_balance: float
    stagnation_range_decay_rate: float
    stagnation_quality_score: float
    pullback_overlap_ratio: float
    pullback_efficiency: float
    correction_smoothness: float
    rhythm_score: float
    correction_velocity: float
    breakout_momentum_ratio: float
    breakout_velocity: float
    wick_ratio: float
    reaccel_follow_through: float
    reaccel_candle_imbalance: float
    reacceleration_score: float
    h1_momentum_score: float
    tp_mode: TpMode
    tp_rr_actual: float
    structure_score: float
    market_breath_score: float
    current_atr_h1: float
    volatility_percentile: float
    session_type: SessionType
    minutes_from_session_open: int
    adr_used: float
    adr_remaining: float
    distance_daily_high: float
    distance_daily_low: float
    distance_session_high: float
    distance_session_low: float
    spread: float
    spread_percentile: float
    tick_volume_ratio: float
    vp_location_score: int

    def as_dict(self) -> dict[str, float | str | int]:
        return {
            "dow_phase": self.dow_phase,
            "trend_age_bars": self.trend_age_bars,
            "pullback_depth": self.pullback_depth,
            "pullback_duration": self.pullback_duration,
            "impulse_quality": self.impulse_quality,
            "impulse_atr_ratio": self.impulse_atr_ratio,
            "stagnation_duration": self.stagnation_duration,
            "stagnation_pips_width": self.stagnation_pips_width,
            "stagnation_compression_ratio": self.stagnation_compression_ratio,
            "stagnation_wick_balance": self.stagnation_wick_balance,
            "stagnation_range_decay_rate": self.stagnation_range_decay_rate,
            "stagnation_quality_score": self.stagnation_quality_score,
            "pullback_overlap_ratio": self.pullback_overlap_ratio,
            "pullback_efficiency": self.pullback_efficiency,
            "correction_smoothness": self.correction_smoothness,
            "rhythm_score": self.rhythm_score,
            "correction_velocity": self.correction_velocity,
            "breakout_momentum_ratio": self.breakout_momentum_ratio,
            "breakout_velocity": self.breakout_velocity,
            "wick_ratio": self.wick_ratio,
            "reaccel_follow_through": self.reaccel_follow_through,
            "reaccel_candle_imbalance": self.reaccel_candle_imbalance,
            "reacceleration_score": self.reacceleration_score,
            "h1_momentum_score": self.h1_momentum_score,
            "tp_mode": self.tp_mode,
            "tp_rr_actual": self.tp_rr_actual,
            "structure_score": self.structure_score,
            "market_breath_score": self.market_breath_score,
            "current_atr_h1": self.current_atr_h1,
            "volatility_percentile": self.volatility_percentile,
            "session_type": self.session_type,
            "minutes_from_session_open": self.minutes_from_session_open,
            "adr_used": self.adr_used,
            "adr_remaining": self.adr_remaining,
            "distance_daily_high": self.distance_daily_high,
            "distance_daily_low": self.distance_daily_low,
            "distance_session_high": self.distance_session_high,
            "distance_session_low": self.distance_session_low,
            "spread": self.spread,
            "spread_percentile": self.spread_percentile,
            "tick_volume_ratio": self.tick_volume_ratio,
            "vp_location_score": self.vp_location_score,
        }


TrendPhase = Literal["UPTREND", "DOWNTREND", "RANGE", "NONE"]


@dataclass(frozen=True)
class CspaSetup:
    timestamp: pd.Timestamp
    pair: str
    direction: TradeDirection
    bias_phase: TrendPhase
    impulse: ImpulseLeg
    retrace_ratio: float
    stagnation: StagnationCluster
    momentum: MomentumSignal
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    tp_mode: TpMode
    candidate_score: float
    score_breakdown: CspaScoreBreakdown
    h1_trend: str
    htf_aligned: bool
    reason_codes: tuple[str, ...]
    bar_index: int
    structure_bar_index: int
    sweep_distance: float
    bayes_features: CspaBayesFeatures


def scale_cspa_take_profit(
    entry_price: float,
    take_profit: float,
    direction: TradeDirection,
    tp_multiplier: float,
) -> float:
    """CSPA ベイズゲート Tier 3 の TP 倍率を利確価格に反映する。"""
    if tp_multiplier <= 0.0:
        return take_profit
    reward = abs(take_profit - entry_price) * tp_multiplier
    if direction == "BUY":
        return entry_price + reward
    return entry_price - reward


def classify_dow_phase_maturity(bias_phase: TrendPhase, correction_bars: int) -> DowPhaseLabel:
    """H1 Dow 局面の成熟度: 転換初動 / 成熟トレンド / レンジ。"""
    if bias_phase in ("RANGE", "NONE"):
        return "RANGE"
    if correction_bars <= CSPA_DOW_EARLY_BARS:
        return "EARLY_TREND"
    return "MATURE_TREND"


def _cspa_session_hour_ranges(
    session_date: date,
    dst_type: str = DATA_DST_TYPE,
) -> dict[SessionType, range]:
    """CSPA セッション帯（GMT 基準。GMT_FIXED + 米国 DST 期間は -1h シフト）。"""
    return {
        "ASIA": shift_hour_range(
            session_date, CSPA_ASIA_HOUR_START, CSPA_ASIA_HOUR_END, dst_type
        ),
        "LONDON": shift_hour_range(
            session_date, CSPA_LONDON_HOUR_START, CSPA_LONDON_HOUR_END, dst_type
        ),
        "NY": shift_hour_range(session_date, CSPA_NY_HOUR_START, CSPA_NY_HOUR_END, dst_type),
    }


def resolve_cspa_session_type(
    timestamp: pd.Timestamp,
    *,
    dst_type: str = DATA_DST_TYPE,
) -> SessionType:
    ts = pd.Timestamp(timestamp)
    hour = int(ts.hour)
    ranges = _cspa_session_hour_ranges(ts.date(), dst_type)
    if hour in ranges["LONDON"]:
        return "LONDON"
    if hour in ranges["NY"]:
        return "NY"
    if hour in ranges["ASIA"]:
        return "ASIA"
    return "OFF_HOURS"


def _compute_adr_used(structure_df: pd.DataFrame, bar_index: int) -> float:
    adr_used, _ = _compute_adr_pair(structure_df, bar_index)
    return adr_used


@dataclass(frozen=True)
class _StructureAdrCache:
    day_norm: np.ndarray
    day_start: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    daily_range: dict[np.datetime64, float]


def _build_structure_adr_cache(structure_df: pd.DataFrame) -> _StructureAdrCache:
    work = _as_prepared_ohlcv(structure_df)
    day_norm = pd.to_datetime(work["datetime"]).dt.normalize().values.astype("datetime64[D]")
    highs = work["high"].astype(float).to_numpy()
    lows = work["low"].astype(float).to_numpy()
    day_start = np.zeros(len(work), dtype=np.int64)
    for i in range(1, len(work)):
        day_start[i] = day_start[i - 1] if day_norm[i] == day_norm[i - 1] else i
    daily_range: dict[np.datetime64, float] = {}
    for day in np.unique(day_norm):
        mask = day_norm == day
        if int(mask.sum()) >= 4:
            daily_range[day] = float(highs[mask].max() - lows[mask].min())
    return _StructureAdrCache(
        day_norm=day_norm,
        day_start=day_start,
        highs=highs,
        lows=lows,
        daily_range=daily_range,
    )


def _compute_adr_pair(
    structure_df: pd.DataFrame,
    bar_index: int,
    *,
    adr_cache: _StructureAdrCache | None = None,
) -> tuple[float, float]:
    if adr_cache is not None:
        if bar_index < 0 or bar_index >= len(adr_cache.highs):
            return 0.0, 1.0
        start = int(adr_cache.day_start[bar_index])
        day_range = float(
            adr_cache.highs[start : bar_index + 1].max() - adr_cache.lows[start : bar_index + 1].min()
        )
        day = adr_cache.day_norm[bar_index]
        prior_ranges: list[float] = []
        for offset in range(1, CSPA_ADR_LOOKBACK_DAYS + 1):
            prior_day = day - np.timedelta64(offset, "D")
            val = adr_cache.daily_range.get(prior_day)
            if val is not None:
                prior_ranges.append(val)
        if not prior_ranges:
            return 0.0, 1.0
        adr = sum(prior_ranges) / len(prior_ranges)
        if adr <= 0:
            return 0.0, 1.0
        adr_used = round(day_range / adr, 4)
        return adr_used, round(max(0.0, 1.0 - adr_used), 4)

    work = _as_prepared_ohlcv(structure_df)
    if work.empty or bar_index < 0 or bar_index >= len(work):
        return 0.0, 1.0
    ts = pd.Timestamp(work.iloc[bar_index]["datetime"])
    day = ts.normalize()
    day_bars = work.loc[work["datetime"].dt.normalize() == day]
    if day_bars.empty:
        return 0.0, 1.0
    day_range = float(day_bars["high"].max()) - float(day_bars["low"].min())
    daily_ranges: list[float] = []
    for offset in range(1, CSPA_ADR_LOOKBACK_DAYS + 1):
        prior_day = day - pd.Timedelta(days=offset)
        prior_bars = work.loc[work["datetime"].dt.normalize() == prior_day]
        if len(prior_bars) >= 4:
            daily_ranges.append(float(prior_bars["high"].max()) - float(prior_bars["low"].min()))
    if not daily_ranges:
        return 0.0, 1.0
    adr = sum(daily_ranges) / len(daily_ranges)
    if adr <= 0:
        return 0.0, 1.0
    adr_used = round(day_range / adr, 4)
    adr_remaining = round(max(0.0, 1.0 - adr_used), 4)
    return adr_used, adr_remaining


def _breakout_momentum_ratio(trigger_df: pd.DataFrame, momentum: MomentumSignal) -> float:
    work = _as_prepared_ohlcv(trigger_df)
    if momentum.bar_index < 0 or momentum.bar_index >= len(work):
        return 0.0
    row = work.iloc[momentum.bar_index]
    body = _body_size(row)
    bar_range = float(row["high"]) - float(row["low"])
    body_share = body / bar_range if bar_range > 0 else 0.0
    return round(momentum.body_atr * body_share, 4)


def _breakout_velocity(stagnation: StagnationCluster, momentum: MomentumSignal) -> float:
    if stagnation.avg_body_atr <= 0:
        return round(momentum.body_atr, 4)
    return round(momentum.body_atr / stagnation.avg_body_atr, 4)


def _wick_ratio(trigger_df: pd.DataFrame, momentum: MomentumSignal) -> float:
    work = _as_prepared_ohlcv(trigger_df)
    if momentum.bar_index < 0 or momentum.bar_index >= len(work):
        return 0.0
    row = work.iloc[momentum.bar_index]
    bar_range = float(row["high"]) - float(row["low"])
    if bar_range <= 0:
        return 0.0
    body = _body_size(row)
    return round(max(0.0, (bar_range - body) / bar_range), 4)


def _stagnation_compression_ratio(stagnation: StagnationCluster, momentum: MomentumSignal) -> float:
    zone_width = stagnation.zone_high - stagnation.zone_low
    atr = momentum.atr if momentum.atr > 0 else 1e-9
    denom = atr * max(stagnation.bar_count, 1)
    return round(zone_width / denom, 4)


def _volatility_percentile_from_atr(
    atr_series: pd.Series | np.ndarray,
    bar_index: int,
    *,
    lookback: int = CSPA_VOLATILITY_LOOKBACK_BARS,
) -> float:
    """Precomputed H1 ATR 系列から分位（O(lookback)・ATR 再計算なし）。"""
    if isinstance(atr_series, np.ndarray):
        from strategies.archive.cspa_scan_hot import volatility_percentile_np

        return volatility_percentile_np(
            atr_series,
            bar_index,
            lookback=lookback,
            atr_period=ATR_PERIOD,
        )
    if bar_index < ATR_PERIOD or bar_index >= len(atr_series):
        return 0.5
    start = max(ATR_PERIOD, bar_index - lookback + 1)
    window = atr_series.iloc[start : bar_index + 1].astype(float)
    if window.empty:
        return 0.5
    current = float(window.iloc[-1])
    if current <= 0:
        return 0.5
    below = int((window <= current).sum())
    return round(below / len(window), 4)


def _volatility_percentile(
    bias_df: pd.DataFrame,
    timestamp: pd.Timestamp,
    *,
    bias_atr: pd.Series | None = None,
    bias_idx: int | None = None,
) -> float:
    if bias_atr is not None and bias_idx is not None:
        return _volatility_percentile_from_atr(bias_atr, bias_idx)
    work = _as_prepared_ohlcv(bias_df)
    if work.empty:
        return 0.5
    idx = bias_idx if bias_idx is not None else _bar_index_as_of(work, timestamp)
    if idx < ATR_PERIOD:
        return 0.5
    if bias_atr is None:
        bias_atr = compute_atr(work, period=ATR_PERIOD)
    return _volatility_percentile_from_atr(bias_atr, idx)


def _minutes_from_session_open(
    timestamp: pd.Timestamp,
    session_type: SessionType,
    *,
    dst_type: str = DATA_DST_TYPE,
) -> int:
    ts = pd.Timestamp(timestamp)
    open_hour = shift_hour(ts.date(), CSPA_SESSION_OPEN_HOUR[session_type], dst_type)
    session_open = ts.normalize() + pd.Timedelta(hours=open_hour)
    if ts < session_open and session_type != "ASIA":
        session_open -= pd.Timedelta(days=1)
    return max(0, int((ts - session_open).total_seconds() // 60))


def _session_window_start(
    timestamp: pd.Timestamp,
    session_type: SessionType,
    *,
    dst_type: str = DATA_DST_TYPE,
) -> pd.Timestamp:
    ts = pd.Timestamp(timestamp)
    open_hour = shift_hour(ts.date(), CSPA_SESSION_OPEN_HOUR[session_type], dst_type)
    start = ts.normalize() + pd.Timedelta(hours=open_hour)
    if session_type == "NY" and ts.hour < 21:
        start -= pd.Timedelta(days=1)
    if ts < start and session_type not in ("ASIA", "OFF_HOURS"):
        start -= pd.Timedelta(days=1)
    return start


def _distance_to_daily_extremes(
    structure_df: pd.DataFrame,
    struct_idx: int,
    entry_price: float,
    atr_h1: float,
) -> tuple[float, float]:
    """当日高値/安値までの距離を H1 ATR で正規化（ペア・ボラ横断比較用）。"""
    work = _as_prepared_ohlcv(structure_df)
    if work.empty or struct_idx < 0 or struct_idx >= len(work) or atr_h1 <= 0:
        return 0.0, 0.0
    ts = pd.Timestamp(work.iloc[struct_idx]["datetime"])
    day_bars = work.loc[work["datetime"].dt.normalize() == ts.normalize()]
    if day_bars.empty:
        return 0.0, 0.0
    day_high = float(day_bars["high"].max())
    day_low = float(day_bars["low"].min())
    return (
        round((day_high - entry_price) / atr_h1, 4),
        round((entry_price - day_low) / atr_h1, 4),
    )


def _distance_to_session_extremes(
    trigger_df: pd.DataFrame,
    timestamp: pd.Timestamp,
    session_type: SessionType,
    entry_price: float,
    pair: str,
    *,
    bar_index: int | None = None,
) -> tuple[float, float]:
    work = _as_prepared_ohlcv(trigger_df)
    if work.empty:
        return 0.0, 0.0
    end_i = min(bar_index if bar_index is not None else len(work) - 1, len(work) - 1)
    if end_i < 0:
        return 0.0, 0.0
    start = _session_window_start(timestamp, session_type)
    start_i = int(work["datetime"].searchsorted(pd.Timestamp(start), side="left"))
    if start_i > end_i:
        return 0.0, 0.0
    seg = work.iloc[start_i : end_i + 1]
    if seg.empty:
        return 0.0, 0.0
    sess_high = float(seg["high"].max())
    sess_low = float(seg["low"].min())
    pip = pip_size_for_pair(pair)
    if pip <= 0:
        return 0.0, 0.0
    return round((sess_high - entry_price) / pip, 2), round((entry_price - sess_low) / pip, 2)


def _session_trigger_slice(
    trigger_df: pd.DataFrame,
    timestamp: pd.Timestamp,
    session_type: SessionType,
    bar_index: int,
) -> pd.DataFrame:
    """resolve_cspa_session_type のセッション開始〜 bar_index までの M1 足（非破壊）。"""
    work = _as_prepared_ohlcv(trigger_df)
    if work.empty or session_type == "OFF_HOURS":
        return work.iloc[0:0].copy()
    start = _session_window_start(timestamp, session_type)
    start_i = int(work["datetime"].searchsorted(pd.Timestamp(start), side="left"))
    end_i = min(bar_index, len(work) - 1)
    if start_i > end_i:
        return work.iloc[0:0].copy()
    return work.iloc[start_i : end_i + 1].copy()


def compute_cspa_session_volume_profile(
    trigger_df: pd.DataFrame,
    timestamp: pd.Timestamp,
    pair: str,
    bar_index: int,
    *,
    session_type: SessionType | None = None,
) -> VolumeProfileLevels:
    """セッション開始から現在 bar までの M1 データで VAH / VAL / POC を算出。"""
    session = session_type or resolve_cspa_session_type(timestamp)
    session_df = _session_trigger_slice(trigger_df, timestamp, session, bar_index)
    return SessionVolumeProfile.for_pair(pair).calculate_profile(session_df)


def evaluate_cspa_vp_location(
    trigger_df: pd.DataFrame,
    momentum: MomentumSignal,
    pair: str,
    direction: TradeDirection,
    *,
    bar_index: int | None = None,
    buffer_pips: float | None = None,
    buffer_atr: float | None = None,
    score_tiers: VpLocationScoreTiers | None = None,
) -> tuple[bool, int, VolumeProfileLevels]:
    """セッション VP を算出し、(SWEEP 許可, vp_location_score, profile) を返す。"""
    idx = bar_index if bar_index is not None else momentum.bar_index
    session_type = resolve_cspa_session_type(momentum.timestamp)
    if session_type == "OFF_HOURS":
        empty: VolumeProfileLevels = {"vah": np.nan, "val": np.nan, "poc": np.nan}
        return False, 0, empty

    levels = compute_cspa_session_volume_profile(
        trigger_df,
        momentum.timestamp,
        pair,
        idx,
        session_type=session_type,
    )
    pip = pip_size_for_pair(pair)
    filter_price = momentum.trigger_low if direction == "BUY" else momentum.trigger_high
    atr_buffer = buffer_atr
    if atr_buffer is None and CSPA_VP_BUFFER_ATR_MULT > 0 and momentum.atr > 0:
        atr_buffer = momentum.atr * CSPA_VP_BUFFER_ATR_MULT

    profiler = SessionVolumeProfile.for_pair(pair)
    is_allowed, location_score = profiler.evaluate_vp_location(
        direction,
        levels,
        pip_size=pip,
        filter_price=filter_price,
        score_price=momentum.entry_price,
        buffer_pips=CSPA_VP_BUFFER_PIPS if buffer_pips is None else buffer_pips,
        buffer_atr=atr_buffer,
        score_tiers=CSPA_VP_SCORE_TIERS if score_tiers is None else score_tiers,
    )
    return is_allowed, location_score, levels


def _spread_percentile(spread_pips: float) -> float:
    ref = max(CSPA_SPREAD_REF_PIPS, 1e-9)
    return round(min(max(spread_pips / ref, 0.0), 1.0), 4)


def _tick_volume_ratio(
    trigger_df: pd.DataFrame,
    stagnation: StagnationCluster,
    momentum: MomentumSignal,
) -> float:
    """Bayes 学習ログ専用。ティックボリュームはブローカー依存（L2/L3.5 未使用）。"""
    work = _as_prepared_ohlcv(trigger_df)
    if work.empty or "volume" not in work.columns:
        return 1.0
    if momentum.bar_index < 0 or momentum.bar_index >= len(work):
        return 1.0
    breakout_vol = float(work.iloc[momentum.bar_index]["volume"])
    start = max(0, momentum.bar_index - CSPA_VOLUME_LOOKBACK_BARS)
    baseline = work.iloc[start:momentum.bar_index]["volume"]
    if baseline.empty:
        stagnation_vols = work.iloc[stagnation.start_index : stagnation.end_index + 1]["volume"]
        baseline = stagnation_vols if not stagnation_vols.empty else baseline
    if baseline.empty:
        return 1.0
    avg_vol = float(baseline.mean())
    if avg_vol <= 0:
        return 1.0
    return round(breakout_vol / avg_vol, 4)


def build_cspa_bayes_features(
    *,
    pair: str,
    bias_phase: TrendPhase,
    impulse: ImpulseLeg,
    struct_idx: int,
    retrace_ratio: float,
    stagnation: StagnationCluster,
    momentum: MomentumSignal,
    impulse_size_atr: float,
    structure_df: pd.DataFrame,
    bias_df: pd.DataFrame,
    trigger_df: pd.DataFrame,
    timestamp: pd.Timestamp,
    trend_context: TrendContext | None = None,
    pullback_rhythm: PullbackRhythm | None = None,
    stagnation_quality: StagnationQuality | None = None,
    reacceleration: Reacceleration | None = None,
    tp_mode: TpMode = "FIXED_RR",
    tp_rr_actual: float = DEFAULT_RISK_REWARD,
    structure_score: float = 0.0,
    market_breath_score: float = 0.0,
    bias_atr: pd.Series | None = None,
    bias_idx: int | None = None,
    trigger_bar_index: int | None = None,
    volatility_percentile: float | None = None,
    adr_cache: _StructureAdrCache | None = None,
    vp_location_score: int = 0,
) -> CspaBayesFeatures:
    pullback_duration = max(0, struct_idx - impulse.end_index)
    trend_age_bars = max(0, struct_idx - impulse.start_index)
    pip = pip_size_for_pair(pair)
    stagnation_pips = (stagnation.zone_high - stagnation.zone_low) / pip if pip > 0 else 0.0
    resolved_bias_idx = bias_idx if bias_idx is not None else _bar_index_as_of(bias_df, timestamp)
    h1_atr = _atr_at_bar(_as_prepared_ohlcv(bias_df), resolved_bias_idx, atr_series=bias_atr)
    impulse_quality = round(min(max(impulse_size_atr / 1.5, 0.0), 1.0), 4)
    session_type = resolve_cspa_session_type(timestamp)
    adr_used, adr_remaining = _compute_adr_pair(structure_df, struct_idx, adr_cache=adr_cache)
    dist_daily_high, dist_daily_low = _distance_to_daily_extremes(
        structure_df, struct_idx, momentum.entry_price, h1_atr
    )
    dist_sess_high, dist_sess_low = _distance_to_session_extremes(
        trigger_df, timestamp, session_type, momentum.entry_price, pair,
        bar_index=trigger_bar_index if trigger_bar_index is not None else momentum.bar_index,
    )
    spread = CSPA_BT_SPREAD_PIPS

    if trend_context is None:
        trend_context = build_trend_context(
            bias_df, bias_idx, bias_phase, impulse, struct_idx, timestamp
        )
    if pullback_rhythm is None:
        pullback_rhythm = build_pullback_rhythm(structure_df, impulse, struct_idx, retrace_ratio)
    if stagnation_quality is None:
        trade_dir: TradeDirection = "BUY" if bias_phase == "UPTREND" else "SELL"
        stagnation_quality = build_stagnation_quality(
            trigger_df, stagnation, momentum, trade_dir
        )
    if reacceleration is None:
        reacceleration = build_reacceleration(trigger_df, stagnation, momentum, trade_dir)

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
        breakout_momentum_ratio=_breakout_momentum_ratio(trigger_df, momentum),
        breakout_velocity=reacceleration.breakout_velocity,
        wick_ratio=_wick_ratio(trigger_df, momentum),
        reaccel_follow_through=reacceleration.follow_through,
        reaccel_candle_imbalance=reacceleration.candle_imbalance,
        reacceleration_score=reacceleration.composite_score,
        h1_momentum_score=trend_context.momentum_score,
        tp_mode=tp_mode,
        tp_rr_actual=round(tp_rr_actual, 4),
        structure_score=round(structure_score, 2),
        market_breath_score=round(market_breath_score, 2),
        current_atr_h1=round(h1_atr, 6),
        volatility_percentile=(
            volatility_percentile
            if volatility_percentile is not None
            else _volatility_percentile(
                bias_df, timestamp, bias_atr=bias_atr, bias_idx=resolved_bias_idx
            )
        ),
        session_type=session_type,
        minutes_from_session_open=_minutes_from_session_open(timestamp, session_type),
        adr_used=adr_used,
        adr_remaining=adr_remaining,
        distance_daily_high=dist_daily_high,
        distance_daily_low=dist_daily_low,
        distance_session_high=dist_sess_high,
        distance_session_low=dist_sess_low,
        spread=spread,
        spread_percentile=_spread_percentile(spread),
        tick_volume_ratio=_tick_volume_ratio(trigger_df, stagnation, momentum),
        vp_location_score=vp_location_score,
    )


def is_cspa_be_trail_enabled() -> bool:
    return CSPA_BE_ENABLED or CSPA_TRAIL_ENABLED


def _cspa_profit_r(direction: TradeDirection, entry: float, exit_price: float, initial_risk: float) -> float:
    if initial_risk <= 0:
        return 0.0
    if direction == "BUY":
        return (exit_price - entry) / initial_risk
    return (entry - exit_price) / initial_risk


def _cspa_close_unrealized_r(direction: TradeDirection, entry: float, close: float, initial_risk: float) -> float:
    return _cspa_profit_r(direction, entry, close, initial_risk)


def _cspa_ratchet_sl(direction: TradeDirection, current_sl: float, new_sl: float) -> float:
    if direction == "BUY":
        return max(current_sl, new_sl)
    return min(current_sl, new_sl)


def _cspa_breakeven_sl(direction: TradeDirection, entry: float, atr: float) -> float:
    buffer = CSPA_BE_BUFFER_ATR * atr
    if direction == "BUY":
        return entry + buffer
    return entry - buffer


def track_cspa_trade_outcome(
    pair_df: pd.DataFrame,
    start_index: int,
    direction: TradeDirection,
    entry: float,
    stop_loss: float,
    take_profit: float,
    bar_minutes: int,
    *,
    atr: float,
    max_holding_bars: int,
    pip_size: float = 0.0001,
) -> tuple[str, float, float, int, float]:
    """CSPA L5 追跡 — 建値ストップ + ATR トレーリング（原著の早期脱出）。

    Returns: (result, profit_r, profit_pips, holding_minutes, final_sl)
    """
    initial_risk = abs(entry - stop_loss)
    from strategies.bt_ohlcv import as_ohlcv

    ohlcv = as_ohlcv(pair_df)
    if initial_risk <= 0 or start_index < 0 or start_index >= ohlcv.length:
        return "LOSS", -1.0, 0.0, 0, stop_loss

    if not is_cspa_be_trail_enabled():
        from strategies.bt_l5 import track_cspa_fixed_sl_np

        return track_cspa_fixed_sl_np(
            ohlcv,
            start_index,
            direction,
            entry,
            stop_loss,
            take_profit,
            bar_minutes,
            max_holding_bars=max_holding_bars,
            pip_size=pip_size,
        )

    end_index = min(start_index + max_holding_bars, ohlcv.length - 1)
    current_sl = stop_loss
    peak_favorable = entry
    extension_armed = False
    sl_at_breakeven = False
    trail_atr = max(atr, initial_risk * 0.25)
    be_sl = _cspa_breakeven_sl(direction, entry, trail_atr)

    for i in range(start_index + 1, end_index + 1):
        high = float(ohlcv.high[i])
        low = float(ohlcv.low[i])
        close = float(ohlcv.close[i])
        elapsed = (i - start_index) * bar_minutes

        if direction == "BUY":
            bar_mfe_r = (high - entry) / initial_risk
            bar_mae_r = (entry - low) / initial_risk
            peak_favorable = max(peak_favorable, high)
        else:
            bar_mfe_r = (entry - low) / initial_risk
            bar_mae_r = (high - entry) / initial_risk
            peak_favorable = min(peak_favorable, low)

        if CSPA_BE_ENABLED:
            if bar_mfe_r >= CSPA_BE_ARM_MFE_R:
                extension_armed = True
            close_r = _cspa_close_unrealized_r(direction, entry, close, initial_risk)
            rhythm_window = (i - start_index) <= CSPA_BE_RHYTHM_MAX_BARS
            if bar_mfe_r >= CSPA_BE_TRIGGER_MFE_R:
                current_sl = _cspa_ratchet_sl(direction, current_sl, be_sl)
                sl_at_breakeven = True
            elif extension_armed and rhythm_window and close_r <= CSPA_BE_PULLBACK_CLOSE_R:
                current_sl = _cspa_ratchet_sl(direction, current_sl, be_sl)
                sl_at_breakeven = True

        if CSPA_TRAIL_ENABLED and sl_at_breakeven:
            if direction == "BUY":
                trail_sl = peak_favorable - CSPA_TRAIL_ATR_MULT * trail_atr
                trail_sl = max(trail_sl, be_sl)
            else:
                trail_sl = peak_favorable + CSPA_TRAIL_ATR_MULT * trail_atr
                trail_sl = min(trail_sl, be_sl)
            current_sl = _cspa_ratchet_sl(direction, current_sl, trail_sl)

        if direction == "BUY":
            sl_hit = low <= current_sl
            tp_hit = high >= take_profit
        else:
            sl_hit = high >= current_sl
            tp_hit = low <= take_profit

        if sl_hit and tp_hit:
            exit_price = current_sl
            profit_r = max(_cspa_profit_r(direction, entry, exit_price, initial_risk), -1.0)
            return _cspa_result_from_r(profit_r), profit_r, profit_r * initial_risk / pip_size, elapsed, current_sl

        if sl_hit:
            profit_r = max(_cspa_profit_r(direction, entry, current_sl, initial_risk), -1.0)
            return _cspa_result_from_r(profit_r), profit_r, profit_r * initial_risk / pip_size, elapsed, current_sl

        if tp_hit:
            profit_r = abs(take_profit - entry) / initial_risk
            return "WIN", profit_r, profit_r * initial_risk / pip_size, elapsed, current_sl

    last_close = float(ohlcv.close[end_index])
    profit_r = max(-1.0, min(2.4, _cspa_profit_r(direction, entry, last_close, initial_risk)))
    holding = (end_index - start_index) * bar_minutes
    return _cspa_result_from_r(profit_r), profit_r, profit_r * initial_risk / pip_size, holding, current_sl


def _cspa_result_from_r(profit_r: float) -> str:
    return "WIN" if profit_r >= 0.0 else "LOSS"


def _compute_cspa_excursions_fixed_sl(
    pair_df: pd.DataFrame,
    start_index: int,
    direction: TradeDirection,
    entry: float,
    stop_loss: float,
    take_profit: float,
    initial_risk: float,
    *,
    max_holding_bars: int,
) -> dict[str, float | str]:
    """固定 SL（BE/Trail OFF）— 旧 compute_cspa_trade_excursions ロジック。"""
    from strategies.bt_l5 import compute_trade_excursions_np
    from strategies.bt_ohlcv import as_ohlcv

    return compute_trade_excursions_np(
        as_ohlcv(pair_df),
        start_index,
        entry,
        stop_loss,
        take_profit,
        max_holding_bars=max_holding_bars,
        direction=direction,
    )


def compute_cspa_trade_excursions(
    pair_df: pd.DataFrame,
    start_index: int,
    direction: TradeDirection,
    entry: float,
    stop_loss: float,
    take_profit: float,
    bar_minutes: int,
    *,
    max_holding_bars: int,
    atr: float,
    pip_size: float = 0.0001,
) -> dict[str, float | str]:
    """L5 シャドー追跡と同一ルールで result_r / MFE / MAE / WIN|LOSS ラベルを算出。"""
    initial_risk = abs(entry - stop_loss)
    from strategies.bt_ohlcv import as_ohlcv

    ohlcv = as_ohlcv(pair_df)
    if initial_risk <= 0 or start_index < 0 or start_index >= ohlcv.length:
        return {
            "outcome_label": "LOSS",
            "result_r": -1.0,
            "mfe": 0.0,
            "mae": 0.0,
        }

    if not is_cspa_be_trail_enabled():
        return _compute_cspa_excursions_fixed_sl(
            pair_df,
            start_index,
            direction,
            entry,
            stop_loss,
            take_profit,
            initial_risk,
            max_holding_bars=max_holding_bars,
        )

    end_index = min(start_index + max_holding_bars, ohlcv.length - 1)
    mfe_r = 0.0
    mae_r = 0.0
    result, profit_r, _, _, final_sl = track_cspa_trade_outcome(
        pair_df,
        start_index,
        direction,
        entry,
        stop_loss,
        take_profit,
        bar_minutes,
        atr=atr,
        max_holding_bars=max_holding_bars,
        pip_size=pip_size,
    )

    for i in range(start_index + 1, end_index + 1):
        high = float(ohlcv.high[i])
        low = float(ohlcv.low[i])
        if direction == "BUY":
            mfe_r = max(mfe_r, (high - entry) / initial_risk)
            mae_r = max(mae_r, (entry - low) / initial_risk)
        else:
            mfe_r = max(mfe_r, (entry - low) / initial_risk)
            mae_r = max(mae_r, (high - entry) / initial_risk)

    _ = final_sl
    return {
        "outcome_label": result,
        "result_r": round(profit_r, 4),
        "mfe": round(mfe_r, 4),
        "mae": round(mae_r, 4),
    }


def build_cspa_bayes_log_row(
    *,
    trade_id: str,
    setup: CspaSetup,
    decision_source: str,
    executed: bool,
    candidate_score: float,
    bayes_probability: float,
    excursions: dict[str, float | str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "trade_id": trade_id,
        "timestamp": setup.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "pair": setup.pair,
        "direction": setup.direction,
        "decision_source": decision_source,
        "executed": executed,
        "candidate_score": round(candidate_score, 2),
        "bayes_probability": round(bayes_probability, 4),
        **setup.bayes_features.as_dict(),
        **excursions,
    }
    return row


def _prepare_df(df: Any):
    from strategies.bt_ohlcv import BtOhlcvFrame

    if isinstance(df, BtOhlcvFrame):
        return df if not df.empty else BtOhlcvFrame.make_empty()
    if df is None or df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close"])
    required = {"datetime", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OHLCV missing columns: {sorted(missing)}")
    work = df.sort_values("datetime").reset_index(drop=True)
    work["datetime"] = pd.to_datetime(work["datetime"])
    return work


def _as_prepared_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Skip sort/copy when ``detect_setups`` already prepared the frame."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close"])
    dt = df["datetime"]
    if pd.api.types.is_datetime64_any_dtype(dt) and dt.is_monotonic_increasing:
        return df
    return _prepare_df(df)


def prepare_trigger_df(df: pd.DataFrame) -> pd.DataFrame:
    return _prepare_df(df)


def prepare_structure_df(df: pd.DataFrame) -> pd.DataFrame:
    return _prepare_df(df)


def prepare_bias_df(df: pd.DataFrame) -> pd.DataFrame:
    return _prepare_df(df)


def prepare_monitor_df(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias for trigger OHLCV."""
    return prepare_trigger_df(df)


def _find_swings(
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
    dt_ns = pd.to_datetime(work["datetime"]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    swing_highs: list[SwingPoint] = []
    swing_lows: list[SwingPoint] = []

    for pivot in range(lookback, last_pivot + 1):
        left_h = highs[pivot - lookback : pivot]
        right_h = highs[pivot + 1 : pivot + lookback + 1]
        if highs[pivot] >= left_h.max() and highs[pivot] >= right_h.max():
            swing_highs.append(
                SwingPoint(
                    bar_index=pivot,
                    timestamp=pd.Timestamp(int(dt_ns[pivot]), unit="ns"),
                    price=float(highs[pivot]),
                    kind="HIGH",
                )
            )
        left_l = lows[pivot - lookback : pivot]
        right_l = lows[pivot + 1 : pivot + lookback + 1]
        if lows[pivot] <= left_l.min() and lows[pivot] <= right_l.min():
            swing_lows.append(
                SwingPoint(
                    bar_index=pivot,
                    timestamp=pd.Timestamp(int(dt_ns[pivot]), unit="ns"),
                    price=float(lows[pivot]),
                    kind="LOW",
                )
            )
    return swing_highs, swing_lows


def _swings_up_to(
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    up_to_bar_index: int,
    *,
    high_bar_indices: list[int] | None = None,
    low_bar_indices: list[int] | None = None,
) -> tuple[list[SwingPoint], list[SwingPoint]]:
    if high_bar_indices is None:
        high_bar_indices = [s.bar_index for s in swing_highs]
    if low_bar_indices is None:
        low_bar_indices = [s.bar_index for s in swing_lows]
    hi = bisect.bisect_right(high_bar_indices, up_to_bar_index)
    lo = bisect.bisect_right(low_bar_indices, up_to_bar_index)
    return swing_highs[:hi], swing_lows[:lo]


def _atr_at_bar(
    work: pd.DataFrame,
    bar_index: int,
    period: int = ATR_PERIOD,
    *,
    atr_series: pd.Series | None = None,
) -> float:
    if bar_index < 1 or len(work) < 2:
        return 0.0
    if atr_series is not None:
        if bar_index < 0 or bar_index >= len(atr_series):
            return 0.0
        val = float(atr_series.iloc[bar_index])
        return val if val > 0 else 0.0
    atr_computed = compute_atr(work.iloc[: bar_index + 1], period=period)
    if atr_computed is None or len(atr_computed) == 0:
        return 0.0
    val = float(atr_computed.iloc[-1])
    return val if val > 0 else 0.0


def _body_size(row: pd.Series) -> float:
    return abs(float(row["close"]) - float(row["open"]))


def _bar_index_as_of(work: pd.DataFrame, ts: pd.Timestamp) -> int:
    idx = work["datetime"].searchsorted(pd.Timestamp(ts), side="right") - 1
    return max(0, min(int(idx), len(work) - 1))


def classify_bias_dow_phase(
    bias_df: pd.DataFrame,
    bar_index: int,
    *,
    swing_highs: list[SwingPoint] | None = None,
    swing_lows: list[SwingPoint] | None = None,
    high_bar_indices: list[int] | None = None,
    low_bar_indices: list[int] | None = None,
) -> TrendPhase:
    """Dow trend on H1 (bias_tf): HH+HL uptrend, LL+LH downtrend, else range."""
    work = bias_df if swing_highs is not None else prepare_bias_df(bias_df)
    if work.empty or bar_index < 0 or bar_index >= len(work):
        return "NONE"

    if swing_highs is not None and swing_lows is not None:
        highs, lows = _swings_up_to(
            swing_highs,
            swing_lows,
            bar_index,
            high_bar_indices=high_bar_indices,
            low_bar_indices=low_bar_indices,
        )
    else:
        highs, lows = _find_swings(work, lookback=SWING_LOOKBACK_BIAS, up_to_bar_index=bar_index)
    if len(highs) < 2 or len(lows) < 2:
        return "NONE"

    h1, h2 = highs[-2], highs[-1]
    l1, l2 = lows[-2], lows[-1]
    if h2.price > h1.price and l2.price > l1.price:
        return "UPTREND"
    if h2.price < h1.price and l2.price < l1.price:
        return "DOWNTREND"
    return "RANGE"


classify_h4_dow_phase = classify_bias_dow_phase


def find_latest_impulse(
    structure_df: pd.DataFrame,
    bar_index: int,
    phase: TrendPhase,
    *,
    swing_highs: list[SwingPoint] | None = None,
    swing_lows: list[SwingPoint] | None = None,
    high_bar_indices: list[int] | None = None,
    low_bar_indices: list[int] | None = None,
) -> ImpulseLeg | None:
    """Most recent completed impulse leg on M15 aligned with H1 bias."""
    work = structure_df if swing_highs is not None else prepare_structure_df(structure_df)
    if phase not in ("UPTREND", "DOWNTREND") or bar_index < SWING_LOOKBACK_STRUCTURE * 4:
        return None

    up_to = bar_index - 1
    if swing_highs is not None and swing_lows is not None:
        highs, lows = _swings_up_to(
            swing_highs,
            swing_lows,
            up_to,
            high_bar_indices=high_bar_indices,
            low_bar_indices=low_bar_indices,
        )
    else:
        highs, lows = _find_swings(
            work, lookback=SWING_LOOKBACK_STRUCTURE, up_to_bar_index=up_to
        )
    if phase == "UPTREND":
        if len(highs) < 1 or len(lows) < 1:
            return None
        peak = highs[-1]
        prior_lows = [s for s in lows if s.bar_index < peak.bar_index]
        if not prior_lows:
            return None
        trough = prior_lows[-1]
        size = peak.price - trough.price
        if size <= 0:
            return None
        return ImpulseLeg(
            direction="UP",
            start_index=trough.bar_index,
            end_index=peak.bar_index,
            start_price=trough.price,
            end_price=peak.price,
            impulse_size=size,
        )

    if len(lows) < 1 or len(highs) < 1:
        return None
    trough = lows[-1]
    prior_highs = [s for s in highs if s.bar_index < trough.bar_index]
    if not prior_highs:
        return None
    peak = prior_highs[-1]
    size = peak.price - trough.price
    if size <= 0:
        return None
    return ImpulseLeg(
        direction="DOWN",
        start_index=peak.bar_index,
        end_index=trough.bar_index,
        start_price=peak.price,
        end_price=trough.price,
        impulse_size=size,
    )


def measure_retrace_ratio(
    structure_df: pd.DataFrame,
    impulse: ImpulseLeg,
    bar_index: int,
) -> float:
    """M15 インパルスに対する押し戻し深さ（0–1 の連続値）。

    Fib 水準そのものではなく「何 % 戻したか」という物差し。
    L2 粗ゲート・Bayes 特徴量 ``pullback_depth`` の共通ソース。
    """
    work = _as_prepared_ohlcv(structure_df)
    if impulse.impulse_size <= 0:
        return 0.0
    window = work.iloc[impulse.end_index : bar_index + 1]
    if window.empty:
        return 0.0
    if impulse.direction == "UP":
        correction_low = float(window["low"].min())
        return (impulse.end_price - correction_low) / impulse.impulse_size
    correction_high = float(window["high"].max())
    return (correction_high - impulse.end_price) / impulse.impulse_size


def m1_over_retraces_structure(
    trigger_df: pd.DataFrame,
    trigger_index: int,
    impulse: ImpulseLeg,
    structure_df: pd.DataFrame,
) -> bool:
    """PDF guard: M1 correction deeper than 61.8% invalidates M15 corner."""
    work = prepare_trigger_df(trigger_df)
    struct = prepare_structure_df(structure_df)
    if impulse.impulse_size <= 0 or trigger_index < 0 or trigger_index >= len(work):
        return False
    start_ts = pd.Timestamp(struct.iloc[impulse.end_index]["datetime"])
    end_ts = pd.Timestamp(work.iloc[trigger_index]["datetime"])
    seg = work.loc[(work["datetime"] >= start_ts) & (work["datetime"] <= end_ts)]
    if seg.empty:
        return False
    if impulse.direction == "UP":
        deepest = float(seg["low"].min())
        ratio = (impulse.end_price - deepest) / impulse.impulse_size
    else:
        highest = float(seg["high"].max())
        ratio = (highest - impulse.end_price) / impulse.impulse_size
    return ratio > FIB_RETRACE_MAX


def _prior_correction_ratio(
    structure_df: pd.DataFrame,
    impulse: ImpulseLeg,
    phase: TrendPhase,
    *,
    swing_highs: list[SwingPoint] | None = None,
    swing_lows: list[SwingPoint] | None = None,
    high_bar_indices: list[int] | None = None,
    low_bar_indices: list[int] | None = None,
) -> float | None:
    work = _as_prepared_ohlcv(structure_df)
    if swing_highs is not None and swing_lows is not None:
        highs, lows = _swings_up_to(
            swing_highs,
            swing_lows,
            impulse.start_index,
            high_bar_indices=high_bar_indices,
            low_bar_indices=low_bar_indices,
        )
    else:
        highs, lows = _find_swings(
            work, lookback=SWING_LOOKBACK_STRUCTURE, up_to_bar_index=impulse.start_index
        )
    high_arr = work["high"].to_numpy(dtype=np.float64, copy=False)
    low_arr = work["low"].to_numpy(dtype=np.float64, copy=False)
    if phase == "UPTREND":
        if len(highs) < 2 or len(lows) < 2:
            return None
        prev_peak = highs[-2]
        prev_trough = lows[-2]
        size = prev_peak.price - prev_trough.price
        if size <= 0:
            return None
        start = prev_trough.bar_index
        end = prev_peak.bar_index + 1
        if start >= end:
            return None
        return (prev_peak.price - float(np.min(low_arr[start:end]))) / size
    if len(lows) < 2 or len(highs) < 2:
        return None
    prev_trough = lows[-2]
    prev_peak = highs[-2]
    size = prev_peak.price - prev_trough.price
    if size <= 0:
        return None
    start = prev_peak.bar_index
    end = prev_trough.bar_index + 1
    if start >= end:
        return None
    return (float(np.max(high_arr[start:end])) - prev_trough.price) / size


def _volatility_regime_from_percentile(percentile: float) -> VolatilityRegime:
    if percentile < 0.33:
        return "LOW"
    if percentile > 0.66:
        return "HIGH"
    return "NORMAL"


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    return num / den if den > 0 else 0.0


def observe_compression_ratio(stagnation_high: float, stagnation_low: float, atr: float) -> float:
    """停滞帯幅 / ATR。理想 < 0.3、悪い > 0.8。"""
    if atr <= 0:
        return 0.0
    return round((stagnation_high - stagnation_low) / atr, 4)


def observe_range_decay_score(ranges: list[float]) -> float:
    """レンジ縮小の傾き（正 = 後半ほど狭い）。"""
    if len(ranges) < 2:
        return 0.0
    xs = [float(i) for i in range(len(ranges))]
    slope = _linear_slope(xs, ranges)
    return round(-slope, 6)


def observe_wick_balance(rows: list[pd.Series]) -> float:
    """停滞足の (上ヒゲ+下ヒゲ)/レンジ 平均。高い = 攻防が活発。"""
    ratios: list[float] = []
    for row in rows:
        open_ = float(row["open"])
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])
        total = high - low
        if total <= 0:
            continue
        upper = high - max(open_, close)
        lower = min(open_, close) - low
        ratios.append((upper + lower) / total)
    if not ratios:
        return 0.0
    return round(sum(ratios) / len(ratios), 4)


def normalize_compression_score(compression_ratio: float) -> float:
    if compression_ratio <= CSPA_COMPRESSION_IDEAL:
        return 1.0
    if compression_ratio >= CSPA_COMPRESSION_BAD:
        return 0.0
    span = max(CSPA_COMPRESSION_BAD - CSPA_COMPRESSION_IDEAL, 1e-9)
    return max(0.0, 1.0 - (compression_ratio - CSPA_COMPRESSION_IDEAL) / span)


def normalize_range_decay_score(range_decay: float, atr: float) -> float:
    if atr <= 0:
        return 0.0
    normalized = range_decay / max(atr * 0.15, 1e-9)
    return min(max(normalized, 0.0), 1.0)


def composite_stagnation_quality_score(
    compression_ratio: float,
    range_decay: float,
    wick_balance: float,
    atr: float,
) -> float:
    compression_score = normalize_compression_score(compression_ratio)
    decay_score = normalize_range_decay_score(range_decay, atr)
    wick_score = min(max(wick_balance, 0.0), 1.0)
    return round(
        compression_score * 0.4 + decay_score * 0.4 + wick_score * 0.2,
        4,
    )


def observe_overlap_ratio(work: pd.DataFrame, start_idx: int, end_idx: int) -> float:
    """修正波内部の隣接足オーバーラップ（絶対値）平均。"""
    if end_idx <= start_idx:
        return 0.0
    segment = work.iloc[start_idx : end_idx + 1]
    if len(segment) < 2:
        return 0.0
    overlaps: list[float] = []
    for i in range(1, len(segment)):
        prev = segment.iloc[i - 1]
        cur = segment.iloc[i]
        overlap = max(
            0.0,
            min(float(prev["high"]), float(cur["high"]))
            - max(float(prev["low"]), float(cur["low"])),
        )
        overlaps.append(overlap)
    return round(sum(overlaps) / len(overlaps), 6)


def observe_pullback_efficiency(retrace_distance: float, correction_bars: int) -> float:
    """押し幅 ÷ 修正時間。深く時間がかかるほど値が大きい。"""
    bars = max(correction_bars, 1)
    return round(retrace_distance / bars, 6)


def observe_correction_smoothness(work: pd.DataFrame, start_idx: int, end_idx: int) -> float:
    """修正波のレンジ変動係数の逆数 — 滑らかな調整ほど高い。"""
    if end_idx <= start_idx:
        return 0.5
    segment = work.iloc[start_idx : end_idx + 1]
    ranges = [float(row["high"]) - float(row["low"]) for _, row in segment.iterrows()]
    if len(ranges) < 2:
        return 0.5
    mean_r = sum(ranges) / len(ranges)
    if mean_r <= 0:
        return 0.5
    variance = sum((r - mean_r) ** 2 for r in ranges) / len(ranges)
    cv = (variance ** 0.5) / mean_r
    return round(max(0.0, 1.0 - min(cv, 1.0)), 4)


def normalize_overlap_score(overlap: float, atr: float) -> float:
    if atr <= 0:
        return 0.0
    return min(max(overlap / max(atr * 0.5, 1e-9), 0.0), 1.0)


def normalize_pullback_efficiency_score(efficiency: float, atr: float) -> float:
    """浅く速い調整（低 efficiency）を高評価。"""
    if atr <= 0:
        return 0.5
    norm = efficiency / atr
    return max(0.0, 1.0 - min(norm / 1.2, 1.0))


def composite_rhythm_score(
    overlap: float,
    smoothness: float,
    efficiency: float,
    atr: float,
) -> float:
    overlap_score = normalize_overlap_score(overlap, atr)
    efficiency_score = normalize_pullback_efficiency_score(efficiency, atr)
    smoothness_score = min(max(smoothness, 0.0), 1.0)
    return round(
        overlap_score * 0.5 + smoothness_score * 0.3 + efficiency_score * 0.2,
        4,
    )


def observe_breakout_velocity(row: pd.Series, direction: TradeDirection, atr: float) -> float:
    if atr <= 0:
        return 0.0
    open_ = float(row["open"])
    close = float(row["close"])
    body = close - open_ if direction == "BUY" else open_ - close
    return round(max(body, 0.0) / atr, 4)


def observe_follow_through(
    breakout_row: pd.Series,
    next_row: pd.Series | None,
    direction: TradeDirection,
) -> float:
    if next_row is None:
        return 0.0
    breakout_close = float(breakout_row["close"])
    next_close = float(next_row["close"])
    if direction == "BUY":
        return round(next_close - breakout_close, 6)
    return round(breakout_close - next_close, 6)


def observe_candle_imbalance(
    work: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    direction: TradeDirection,
) -> float:
    if end_idx < start_idx:
        return 0.5
    segment = work.iloc[start_idx : end_idx + 1]
    if segment.empty:
        return 0.5
    bullish = sum(1 for _, row in segment.iterrows() if float(row["close"]) > float(row["open"]))
    ratio = bullish / len(segment)
    if direction == "SELL":
        ratio = 1.0 - ratio
    return round(ratio, 4)


def composite_reacceleration_score(
    breakout_velocity: float,
    follow_through: float,
    candle_imbalance: float,
    atr: float,
) -> float:
    velocity_score = min(max(breakout_velocity / 1.2, 0.0), 1.0)
    follow_score = min(max(follow_through / max(atr * 0.35, 1e-9), 0.0), 1.0)
    imbalance_score = min(max(candle_imbalance, 0.0), 1.0)
    return round(
        velocity_score * 0.5 + follow_score * 0.3 + imbalance_score * 0.2,
        4,
    )


def _correction_overlap_ratio(work: pd.DataFrame, start_idx: int, end_idx: int) -> float:
    """後方互換: 正規化 overlap (0–1)。"""
    raw = observe_overlap_ratio(work, start_idx, end_idx)
    atr = _atr_at_bar(work, end_idx) if end_idx >= 0 else 0.0
    return normalize_overlap_score(raw, atr) if atr > 0 else 0.5


def build_pullback_rhythm(
    structure_df: pd.DataFrame,
    impulse: ImpulseLeg,
    struct_idx: int,
    retrace_ratio: float,
    *,
    structure_atr: pd.Series | None = None,
) -> PullbackRhythm:
    work = _as_prepared_ohlcv(structure_df)
    duration = max(1, struct_idx - impulse.end_index)
    atr = _atr_at_bar(work, struct_idx, atr_series=structure_atr)
    overlap = observe_overlap_ratio(work, impulse.end_index, struct_idx)
    smoothness = observe_correction_smoothness(work, impulse.end_index, struct_idx)
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


def build_trend_context(
    bias_df: pd.DataFrame,
    bias_idx: int,
    phase: TrendPhase,
    impulse: ImpulseLeg,
    struct_idx: int,
    timestamp: pd.Timestamp,
    *,
    bias_atr: pd.Series | None = None,
    bias_ema50: pd.Series | None = None,
    swing_highs: list[SwingPoint] | None = None,
    swing_lows: list[SwingPoint] | None = None,
    high_bar_indices: list[int] | None = None,
    low_bar_indices: list[int] | None = None,
    volatility_percentile: float | None = None,
) -> TrendContext:
    work = _as_prepared_ohlcv(bias_df)
    trend_age = max(0, struct_idx - impulse.start_index)
    vol_pct = (
        volatility_percentile
        if volatility_percentile is not None
        else _volatility_percentile(bias_df, timestamp, bias_atr=bias_atr, bias_idx=bias_idx)
    )
    regime = _volatility_regime_from_percentile(vol_pct)

    slope_component = 0.5
    breakout_component = 0.5
    imbalance_component = 0.5
    if not work.empty and bias_idx >= 5:
        closes = work["close"].astype(float)
        ema = bias_ema50 if bias_ema50 is not None else closes.ewm(span=50, adjust=False).mean()
        atr = _atr_at_bar(work, bias_idx, atr_series=bias_atr)
        if bias_idx >= 50 and atr > 0:
            slope_raw = float(ema.iloc[bias_idx] - ema.iloc[bias_idx - 5]) / (5.0 * atr)
            slope_component = min(max(0.5 + slope_raw, 0.0), 1.0)

        window = work.iloc[max(0, bias_idx - 10) : bias_idx + 1]
        bull = 0.0
        bear = 0.0
        for _, row in window.iterrows():
            body = float(row["close"]) - float(row["open"])
            if body >= 0:
                bull += body
            else:
                bear += abs(body)
        total = bull + bear
        if total > 0:
            imbalance_component = bull / total if phase == "UPTREND" else bear / total

        if swing_highs is not None and swing_lows is not None:
            highs, lows = _swings_up_to(
                swing_highs,
                swing_lows,
                bias_idx,
                high_bar_indices=high_bar_indices,
                low_bar_indices=low_bar_indices,
            )
        else:
            highs, lows = _find_swings(work, lookback=SWING_LOOKBACK_BIAS, up_to_bar_index=bias_idx)
        close = float(work.iloc[bias_idx]["close"])
        if phase == "UPTREND" and highs:
            breakout_component = min(max((close - highs[-1].price) / max(atr, 1e-9) + 0.5, 0.0), 1.0)
        elif phase == "DOWNTREND" and lows:
            breakout_component = min(max((lows[-1].price - close) / max(atr, 1e-9) + 0.5, 0.0), 1.0)

    momentum_score = round(
        slope_component * 0.4 + breakout_component * 0.3 + imbalance_component * 0.3,
        4,
    )
    return TrendContext(
        direction=phase,
        momentum_score=momentum_score,
        trend_age_bars=trend_age,
        volatility_regime=regime,
    )


def build_stagnation_quality(
    trigger_df: pd.DataFrame,
    stagnation: StagnationCluster,
    momentum: MomentumSignal,
    direction: TradeDirection,
) -> StagnationQuality:
    work = _as_prepared_ohlcv(trigger_df)
    atr = momentum.atr if momentum.atr > 0 else 1e-9
    compression = observe_compression_ratio(stagnation.zone_high, stagnation.zone_low, atr)

    rows = [work.iloc[i] for i in range(stagnation.start_index, stagnation.end_index + 1)]
    ranges = [max(float(r["high"]) - float(r["low"]), 1e-9) for r in rows]
    wick_balance = observe_wick_balance(rows)
    range_decay = observe_range_decay_score(ranges)
    composite = composite_stagnation_quality_score(compression, range_decay, wick_balance, atr)

    return StagnationQuality(
        compression_ratio=compression,
        wick_balance=wick_balance,
        range_decay_rate=range_decay,
        composite_score=composite,
    )


def build_reacceleration(
    trigger_df: pd.DataFrame,
    stagnation: StagnationCluster,
    momentum: MomentumSignal,
    direction: TradeDirection,
) -> Reacceleration:
    work = _as_prepared_ohlcv(trigger_df)
    row = work.iloc[momentum.bar_index]
    atr = momentum.atr if momentum.atr > 0 else 1e-9
    velocity = observe_breakout_velocity(row, direction, atr)

    next_row = work.iloc[momentum.bar_index + 1] if momentum.bar_index + 1 < len(work) else None
    follow = observe_follow_through(row, next_row, direction)

    imb_start = max(0, stagnation.start_index - CSPA_REACCEL_IMBALANCE_LOOKBACK)
    imbalance = observe_candle_imbalance(work, imb_start, momentum.bar_index, direction)
    composite = composite_reacceleration_score(velocity, follow, imbalance, atr)

    return Reacceleration(
        breakout_velocity=velocity,
        follow_through=follow,
        candle_imbalance=imbalance,
        composite_score=composite,
    )


def _scan_m15_consolidation_zones(
    structure_df: pd.DataFrame,
    up_to_bar: int,
    *,
    lookback: int = CSPA_CONSOLIDATION_LOOKBACK,
    structure_atr: pd.Series | None = None,
) -> list[ConsolidationZone]:
    """M15 スライディング窓: width < ATR×1.5 の区間を停滞帯として抽出。"""
    work = structure_df if structure_atr is not None else prepare_structure_df(structure_df)
    if structure_atr is None:
        structure_atr = compute_atr(work, period=ATR_PERIOD)
    window = CSPA_CONSOLIDATION_WINDOW
    start_i = max(window, up_to_bar - lookback + 1)
    zones: list[ConsolidationZone] = []
    seen_mids: set[float] = set()

    for i in range(start_i, up_to_bar + 1):
        seg_start = i - window
        segment = work.iloc[seg_start:i]
        if len(segment) < window:
            continue
        zone_high = float(segment["high"].max())
        zone_low = float(segment["low"].min())
        width = zone_high - zone_low
        atr_slice = structure_atr.iloc[seg_start:i]
        atr_mean = float(atr_slice.mean()) if len(atr_slice) else 0.0
        if atr_mean <= 0 or width >= atr_mean * CSPA_CONSOLIDATION_WIDTH_ATR:
            continue
        zone_mid = round((zone_high + zone_low) / 2.0, 6)
        if zone_mid in seen_mids:
            continue
        seen_mids.add(zone_mid)
        zones.append(
            ConsolidationZone(
                bar_start=seg_start,
                bar_end=i - 1,
                zone_high=zone_high,
                zone_low=zone_low,
                zone_mid=zone_mid,
            )
        )
    return zones


def find_next_consolidation_zone(
    structure_df: pd.DataFrame,
    struct_idx: int,
    entry_price: float,
    direction: TradeDirection,
    *,
    lookback: int = CSPA_CONSOLIDATION_LOOKBACK,
    structure_atr: pd.Series | None = None,
    zones: list[ConsolidationZone] | None = None,
) -> ConsolidationZone | None:
    """進行方向の最寄り停滞帯（center 基準）。"""
    if zones is None:
        zones = _scan_m15_consolidation_zones(
            structure_df,
            struct_idx,
            lookback=lookback,
            structure_atr=structure_atr,
        )
    if not zones:
        return None

    if direction == "BUY":
        candidates = [z for z in zones if z.zone_mid > entry_price]
        if not candidates:
            return None
        return min(candidates, key=lambda z: z.zone_mid - entry_price)
    candidates = [z for z in zones if z.zone_mid < entry_price]
    if not candidates:
        return None
    return max(candidates, key=lambda z: entry_price - z.zone_mid)


def _fx_sweep_rr_fallback() -> float:
    if CSPA_FX_SWEEP_RR:
        return float(CSPA_FX_SWEEP_RR)
    return DEFAULT_RISK_REWARD


def resolve_cspa_risk_corners(
    momentum: MomentumSignal,
    stagnation: StagnationCluster,
    direction: TradeDirection,
) -> tuple[float, float]:
    """SL 基準価格 — SWEEP は FX_logic の 2 本目 H/L、それ以外は停滞ゾーン。"""
    if momentum.trigger_type == "SWEEP_ENGULFING" and CSPA_FX_SWEEP_SL:
        if direction == "BUY":
            return stagnation.zone_high, momentum.trigger_low
        return momentum.trigger_high, stagnation.zone_low
    return stagnation.zone_high, stagnation.zone_low


def resolve_cspa_sl_buffer(
    atr: float,
    pair: str,
    momentum: MomentumSignal,
) -> float:
    """SL バッファ — SWEEP は pips バッファ（FX_logic）と ATR バッファの大きい方。"""
    atr_buffer = SL_ATR_BUFFER_RATIO * atr
    if momentum.trigger_type != "SWEEP_ENGULFING" or not CSPA_FX_SWEEP_SL:
        return atr_buffer
    pip_buffer = CSPA_FX_SWEEP_SL_BUFFER_PIPS * pip_size_for_pair(pair)
    return max(atr_buffer, pip_buffer)


def fx_sweep_min_risk_ok(
    entry: float,
    stop: float,
    pair: str,
    direction: TradeDirection,
) -> bool:
    """FX_logic M1 スキャル — Entry〜SL が最低 pips 以上（0 で無効）。"""
    if CSPA_FX_SWEEP_MIN_RISK_PIPS <= 0:
        return True
    pip = pip_size_for_pair(pair)
    risk = (entry - stop) if direction == "BUY" else (stop - entry)
    return risk >= CSPA_FX_SWEEP_MIN_RISK_PIPS * pip


def build_risk_levels_for_cspa(
    direction: TradeDirection,
    entry: float,
    stagnation: StagnationCluster,
    momentum: MomentumSignal,
    pair: str,
    atr: float,
    *,
    structure_df: pd.DataFrame | None = None,
    struct_idx: int | None = None,
    structure_atr: pd.Series | None = None,
    consolidation_zones: list[ConsolidationZone] | None = None,
) -> tuple[float, float, float, TpMode] | None:
    """CSPA セットアップ用 SL/TP — FX SWEEP 時のみ SL/RR を FX_logic に合わせる。"""
    corner_high, corner_low = resolve_cspa_risk_corners(momentum, stagnation, direction)
    sl_buffer = resolve_cspa_sl_buffer(atr, pair, momentum)
    rr_fallback = (
        _fx_sweep_rr_fallback()
        if momentum.trigger_type == "SWEEP_ENGULFING" and CSPA_FX_SWEEP_RR
        else DEFAULT_RISK_REWARD
    )
    use_consolidation_tp = CSPA_CONSOLIDATION_TP
    if momentum.trigger_type == "SWEEP_ENGULFING" and CSPA_FX_SWEEP_RR:
        use_consolidation_tp = False
    return build_risk_levels_with_target(
        direction,
        entry,
        corner_high,
        corner_low,
        atr,
        structure_df=structure_df if use_consolidation_tp else None,
        struct_idx=struct_idx if use_consolidation_tp else None,
        risk_reward_fallback=rr_fallback,
        structure_atr=structure_atr if use_consolidation_tp else None,
        consolidation_zones=consolidation_zones if use_consolidation_tp else None,
        sl_buffer=sl_buffer,
    )


def build_risk_levels_with_target(
    direction: TradeDirection,
    entry: float,
    corner_high: float,
    corner_low: float,
    atr: float,
    *,
    structure_df: pd.DataFrame | None = None,
    struct_idx: int | None = None,
    risk_reward_fallback: float = DEFAULT_RISK_REWARD,
    structure_atr: pd.Series | None = None,
    consolidation_zones: list[ConsolidationZone] | None = None,
    sl_buffer: float | None = None,
) -> tuple[float, float, float, TpMode] | None:
    """SL + TP。TP は次停滞帯優先、不可なら固定 RR フォールバック。"""
    buffer = sl_buffer if sl_buffer is not None else SL_ATR_BUFFER_RATIO * atr
    if direction == "BUY":
        stop = corner_low - buffer
        risk = entry - stop
        if risk <= 0:
            return None
        tp_mode: TpMode = "FIXED_RR"
        take = entry + risk * risk_reward_fallback
        if CSPA_CONSOLIDATION_TP and structure_df is not None and struct_idx is not None:
            zone = find_next_consolidation_zone(
                structure_df, struct_idx, entry, direction,
                structure_atr=structure_atr, zones=consolidation_zones,
            )
            if zone is not None:
                zone_tp = zone.zone_mid
                rr = (zone_tp - entry) / risk
                if CSPA_MIN_RR <= rr <= CSPA_MAX_RR:
                    take = zone_tp
                    tp_mode = "CONSOLIDATION"
                elif rr > CSPA_MAX_RR:
                    take = entry + risk * CSPA_MAX_RR
                    tp_mode = "CONSOLIDATION"
    else:
        stop = corner_high + buffer
        risk = stop - entry
        if risk <= 0:
            return None
        tp_mode = "FIXED_RR"
        take = entry - risk * risk_reward_fallback
        if CSPA_CONSOLIDATION_TP and structure_df is not None and struct_idx is not None:
            zone = find_next_consolidation_zone(
                structure_df, struct_idx, entry, direction,
                structure_atr=structure_atr, zones=consolidation_zones,
            )
            if zone is not None:
                zone_tp = zone.zone_mid
                rr = (entry - zone_tp) / risk
                if CSPA_MIN_RR <= rr <= CSPA_MAX_RR:
                    take = zone_tp
                    tp_mode = "CONSOLIDATION"
                elif rr > CSPA_MAX_RR:
                    take = entry - risk * CSPA_MAX_RR
                    tp_mode = "CONSOLIDATION"

    rr_actual = abs(take - entry) / risk
    if rr_actual < CSPA_MIN_RR:
        if direction == "BUY":
            take = entry + risk * CSPA_MIN_RR
        else:
            take = entry - risk * CSPA_MIN_RR
        tp_mode = "FIXED_RR"
        rr_actual = CSPA_MIN_RR

    return round(stop, 6), round(take, 6), round(rr_actual, 4), tp_mode


def correction_rhythm_ok(
    structure_df: pd.DataFrame,
    impulse: ImpulseLeg,
    phase: TrendPhase,
    current_ratio: float,
    *,
    swing_highs: list[SwingPoint] | None = None,
    swing_lows: list[SwingPoint] | None = None,
    high_bar_indices: list[int] | None = None,
    low_bar_indices: list[int] | None = None,
) -> bool:
    prev = _prior_correction_ratio(
        structure_df,
        impulse,
        phase,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        high_bar_indices=high_bar_indices,
        low_bar_indices=low_bar_indices,
    )
    if prev is None or prev <= 0:
        return True
    return current_ratio <= prev * CORRECTION_RHYTHM_MAX_RATIO


def detect_stagnation_cluster(
    trigger_df: pd.DataFrame,
    end_index: int,
    direction: TradeDirection,
    *,
    max_bars: int = STAGNATION_MAX_BARS,
    trigger_atr: pd.Series | None = None,
) -> StagnationCluster | None:
    work = trigger_df if trigger_atr is not None else prepare_trigger_df(trigger_df)
    if end_index < 1:
        return None

    stagnation_indices: list[int] = []
    for i in range(end_index - 1, max(end_index - max_bars - 1, -1), -1):
        if i < 0:
            break
        row = work.iloc[i]
        atr = _atr_at_bar(work, i, atr_series=trigger_atr)
        if atr <= 0:
            break
        body = _body_size(row)
        body_atr = body / atr
        open_ = float(row["open"])
        close = float(row["close"])
        high = float(row["high"])
        low = float(row["low"])

        is_small_body = body_atr <= STAGNATION_MAX_BODY_ATR
        rejection = False
        if direction == "BUY":
            lower_wick = min(open_, close) - low
            rejection = lower_wick >= body * 1.5 and body_atr <= STAGNATION_MAX_BODY_ATR * 1.2
        else:
            upper_wick = high - max(open_, close)
            rejection = upper_wick >= body * 1.5 and body_atr <= STAGNATION_MAX_BODY_ATR * 1.2

        if is_small_body or rejection:
            stagnation_indices.insert(0, i)
        else:
            break

    if len(stagnation_indices) < STAGNATION_MIN_BARS:
        return None

    bodies: list[float] = []
    zone_high = float("-inf")
    zone_low = float("inf")
    for i in stagnation_indices:
        row = work.iloc[i]
        atr = _atr_at_bar(work, i, atr_series=trigger_atr)
        bodies.append(_body_size(row) / atr if atr > 0 else 0.0)
        zone_high = max(zone_high, float(row["high"]))
        zone_low = min(zone_low, float(row["low"]))

    solid = len(stagnation_indices) >= 2 or (
        len(stagnation_indices) == 1 and bodies[0] <= STAGNATION_MAX_BODY_ATR * 0.8
    )
    return StagnationCluster(
        start_index=stagnation_indices[0],
        end_index=stagnation_indices[-1],
        bar_count=len(stagnation_indices),
        avg_body_atr=round(sum(bodies) / len(bodies), 4),
        zone_high=zone_high,
        zone_low=zone_low,
        solid_ground=solid,
    )


def detect_momentum_breakout(
    trigger_df: pd.DataFrame,
    bar_index: int,
    direction: TradeDirection,
    stagnation: StagnationCluster,
    *,
    trigger_atr: pd.Series | None = None,
) -> MomentumSignal:
    work = trigger_df if trigger_atr is not None else prepare_trigger_df(trigger_df)
    empty = MomentumSignal(
        detected=False,
        trigger_type="NONE",
        bar_index=bar_index,
        timestamp=pd.Timestamp.utcnow(),
        entry_price=0.0,
        trigger_high=0.0,
        trigger_low=0.0,
        body_atr=0.0,
        atr=0.0,
    )
    if bar_index < 1 or bar_index >= len(work):
        return empty

    row = work.iloc[bar_index]
    prev = work.iloc[bar_index - 1]
    open_ = float(row["open"])
    close = float(row["close"])
    high = float(row["high"])
    low = float(row["low"])
    atr = _atr_at_bar(work, bar_index, atr_series=trigger_atr)
    body = _body_size(row)
    if atr <= 0 or body < MOMENTUM_MIN_BODY_ATR * atr:
        return empty

    body_atr = body / atr
    ts = pd.Timestamp(row["datetime"])
    prev_open = float(prev["open"])
    prev_close = float(prev["close"])

    if direction == "BUY":
        if close <= open_ or close <= stagnation.zone_high:
            return empty
        trigger_type: MomentumType = "BODY_BREAK"
        if prev_close < prev_open and close > open_ and close >= prev_open:
            trigger_type = "ENGULFING"
        if min(open_, close) - low >= 2.0 * body:
            trigger_type = "PIN_BAR"
    else:
        if close >= open_ or close >= stagnation.zone_low:
            return empty
        trigger_type = "BODY_BREAK"
        if prev_close > prev_open and close < open_ and close <= prev_open:
            trigger_type = "ENGULFING"
        if high - max(open_, close) >= 2.0 * body:
            trigger_type = "PIN_BAR"

    return MomentumSignal(
        detected=True,
        trigger_type=trigger_type,
        bar_index=bar_index,
        timestamp=ts,
        entry_price=close,
        trigger_high=high,
        trigger_low=low,
        body_atr=round(body_atr, 4),
        atr=atr,
    )


def detect_sweep_engulfing_trigger(
    trigger_df: pd.DataFrame,
    bar_index: int,
    direction: TradeDirection,
    stagnation: StagnationCluster | None = None,
    *,
    trigger_atr: pd.Series | None = None,
) -> MomentumSignal:
    """FX_logic 2本足 Outside Bar Sweep — 1本目を包み込みつつ終値で前足 H/L を突破。

    買い: Low2 < Low1 かつ Close2 > High1（High2 では不可）かつ 2本目陽線。
    売り: High2 > High1 かつ Close2 < Low1（Low2 では不可）かつ 2本目陰線。
    停滞ゾーン: 買いは zone_low スウィープ時は zone_high 未突破でも可。売りは zone_high
    スウィープ時は zone_low 未突破でも可。
    同値 Low1/Low2 または High1/High2 は不成立。
    """
    work = trigger_df if trigger_atr is not None else prepare_trigger_df(trigger_df)
    empty = MomentumSignal(
        detected=False,
        trigger_type="NONE",
        bar_index=bar_index,
        timestamp=pd.Timestamp.utcnow(),
        entry_price=0.0,
        trigger_high=0.0,
        trigger_low=0.0,
        body_atr=0.0,
        atr=0.0,
    )
    if bar_index < 1 or bar_index >= len(work):
        return empty

    row = work.iloc[bar_index]
    prev = work.iloc[bar_index - 1]
    open_ = float(row["open"])
    close = float(row["close"])
    high = float(row["high"])
    low = float(row["low"])
    prev_high = float(prev["high"])
    prev_low = float(prev["low"])
    atr = _atr_at_bar(work, bar_index, atr_series=trigger_atr)
    body = _body_size(row)
    if atr <= 0 or body < MOMENTUM_MIN_BODY_ATR * atr:
        return empty

    bar_range = high - low
    prev_range = prev_high - prev_low
    if bar_range < CSPA_FX_SWEEP_MIN_RANGE_ATR * atr:
        return empty
    if prev_range > 0 and bar_range < prev_range * CSPA_FX_SWEEP_MIN_OUTSIDE_RATIO:
        return empty

    if direction == "BUY":
        if low >= prev_low:
            return empty
        if close <= prev_high:
            return empty
        if close <= open_:
            return empty
        if high <= prev_high:
            return empty
        if stagnation is not None:
            swept_zone_low = low < stagnation.zone_low
            if not swept_zone_low and close <= stagnation.zone_high:
                return empty
    else:
        if high <= prev_high:
            return empty
        if close >= prev_low:
            return empty
        if close >= open_:
            return empty
        if low >= prev_low:
            return empty
        if stagnation is not None:
            swept_zone_high = high > stagnation.zone_high
            if not swept_zone_high and close >= stagnation.zone_low:
                return empty

    ts = pd.Timestamp(row["datetime"])
    return MomentumSignal(
        detected=True,
        trigger_type="SWEEP_ENGULFING",
        bar_index=bar_index,
        timestamp=ts,
        entry_price=close,
        trigger_high=high,
        trigger_low=low,
        body_atr=round(body / atr, 4),
        atr=atr,
    )


def resolve_cspa_momentum_trigger(
    trigger_df: pd.DataFrame,
    bar_index: int,
    direction: TradeDirection,
    stagnation: StagnationCluster,
    *,
    trigger_atr: pd.Series | None = None,
) -> MomentumSignal:
    """M1 トリガー — Outside Bar Sweep を優先し、未成立時は従来の momentum breakout。"""
    sweep = detect_sweep_engulfing_trigger(
        trigger_df,
        bar_index,
        direction,
        stagnation,
        trigger_atr=trigger_atr,
    )
    if sweep.detected:
        return sweep
    return detect_momentum_breakout(
        trigger_df,
        bar_index,
        direction,
        stagnation,
        trigger_atr=trigger_atr,
    )


def build_risk_levels(
    direction: TradeDirection,
    entry: float,
    corner_high: float,
    corner_low: float,
    atr: float,
    *,
    risk_reward: float = DEFAULT_RISK_REWARD,
) -> tuple[float, float] | None:
    buffer = SL_ATR_BUFFER_RATIO * atr
    if direction == "BUY":
        stop = corner_low - buffer
        risk = entry - stop
        if risk <= 0:
            return None
        take = entry + risk * risk_reward
    else:
        stop = corner_high + buffer
        risk = stop - entry
        if risk <= 0:
            return None
        take = entry - risk * risk_reward
    return round(stop, 6), round(take, 6)


def _clamp_score(value: float, *, lo: float = 0.0, hi: float = 100.0) -> float:
    return round(max(lo, min(hi, value)), 2)


def score_cspa_bias_alignment(trade_direction: TradeDirection, h1_trend: str) -> float:
    """H1 方向感 × 執行方向（bias_tf）。"""
    trend = str(h1_trend or "NEUTRAL").upper()
    if trade_direction == "BUY":
        if trend == "BULL":
            return CSPA_SCORE_BIAS_MAX
        if trend == "NEUTRAL":
            return CSPA_SCORE_BIAS_MAX * 0.55
        return 0.0
    if trend == "BEAR":
        return CSPA_SCORE_BIAS_MAX
    if trend == "NEUTRAL":
        return CSPA_SCORE_BIAS_MAX * 0.55
    return 0.0


def score_cspa_retrace_beauty(retrace_ratio: float) -> float:
    """L2 粗スコア: 50% 中心の連続評価（38.2–61.8% 帯内）。

    Fib 帯は L2 候補絞り込み用の便宜上の参照。価格の「魔法の水準」ではない。
    最終判断は Bayes（``pullback_depth`` 連続特徴量）に委ねる設計。
    """
    if retrace_ratio < FIB_RETRACE_MIN or retrace_ratio > FIB_RETRACE_MAX:
        return 0.0
    half_width = max(CSPA_RETRACE_BEAUTY_HALF_WIDTH, 1e-9)
    dist = abs(retrace_ratio - FIB_RETRACE_IDEAL)
    linear = 1.0 - (dist / half_width)
    return _clamp_score(CSPA_SCORE_RETRACE_MAX * max(0.0, linear), hi=CSPA_SCORE_RETRACE_MAX)


def score_cspa_stagnation(
    bar_count: int,
    avg_body_atr: float,
    solid_ground: bool,
) -> float:
    """M1 停滞クラスタ — 地盤の堅さ（PDF Ch.3 / Ch.6）。"""
    if not solid_ground or bar_count < STAGNATION_MIN_BARS:
        return 0.0
    score = 8.0
    score += min(8.0, max(0, bar_count - 1) * 4.0)
    if avg_body_atr <= STAGNATION_MAX_BODY_ATR * 0.55:
        score += 4.0
    elif avg_body_atr <= STAGNATION_MAX_BODY_ATR:
        score += 2.0
    return _clamp_score(score, hi=CSPA_SCORE_STAGNATION_MAX)


def score_cspa_momentum(trigger_type: MomentumType, body_atr: float) -> float:
    """M1 勢い足 — パターン種別 + 実体強度。"""
    base = {
        "SWEEP_ENGULFING": 15.0,
        "ENGULFING": 14.0,
        "PIN_BAR": 11.0,
        "BODY_BREAK": 8.0,
        "NONE": 0.0,
    }.get(trigger_type, 0.0)
    if base <= 0:
        return 0.0
    excess = max(0.0, body_atr - MOMENTUM_MIN_BODY_ATR)
    span = max(CSPA_MOMENTUM_BODY_ATR_FULL - MOMENTUM_MIN_BODY_ATR, 1e-9)
    strength = min(6.0, (excess / span) * 6.0)
    return _clamp_score(base + strength, hi=CSPA_SCORE_MOMENTUM_MAX)


def score_cspa_structure_freshness(m15_bars_since_impulse: int) -> float:
    """M15 波の角の鮮度 — 押し目/戻り目形成からの経過本数。"""
    bars = int(m15_bars_since_impulse)
    peak_lo, peak_hi = CSPA_M15_FRESHNESS_PEAK_BARS
    if peak_lo <= bars <= peak_hi:
        return CSPA_SCORE_FRESHNESS_MAX
    if bars == 1:
        return CSPA_SCORE_FRESHNESS_MAX * 0.7
    if bars < peak_lo:
        return CSPA_SCORE_FRESHNESS_MAX * (0.5 + 0.1 * bars)
    if bars <= 20:
        decay = (bars - peak_hi) / max(20 - peak_hi, 1)
        return _clamp_score(CSPA_SCORE_FRESHNESS_MAX * (1.0 - 0.7 * decay), hi=CSPA_SCORE_FRESHNESS_MAX)
    if bars <= MAX_BARS_SINCE_IMPULSE:
        return 2.0
    return 0.0


def score_cspa_correction_rhythm(
    current_ratio: float,
    prior_ratio: float | None,
    *,
    rhythm: PullbackRhythm | None = None,
) -> float:
    """修正波リズム — 観測合成 rhythm_score を主、深さ急変はハード減点。"""
    if prior_ratio is not None and prior_ratio > 0:
        rhythm_ratio = current_ratio / prior_ratio
        if rhythm_ratio > CORRECTION_RHYTHM_MAX_RATIO:
            return 0.0

    if rhythm is None:
        return CSPA_SCORE_RHYTHM_MAX * 0.5

    return _clamp_score(rhythm.rhythm_score * CSPA_BREATH_COMPONENT_MAX, hi=CSPA_BREATH_COMPONENT_MAX)


def score_trend_context(ctx: TrendContext, trade_direction: TradeDirection) -> float:
    """H1 方向感スコア（breath 成分 25 点）。"""
    aligned = (trade_direction == "BUY" and ctx.direction == "UPTREND") or (
        trade_direction == "SELL" and ctx.direction == "DOWNTREND"
    )
    if not aligned:
        return 0.0
    score = ctx.momentum_score * 14.0
    if ctx.trend_age_bars <= CSPA_DOW_EARLY_BARS:
        score += 6.0
    elif ctx.trend_age_bars <= CSPA_TREND_MATURE_BARS:
        score += 3.0
    else:
        score -= min(6.0, (ctx.trend_age_bars - CSPA_TREND_MATURE_BARS) * 0.25)
    if ctx.volatility_regime == "NORMAL":
        score += 3.0
    elif ctx.volatility_regime == "LOW":
        score += 1.0
    return _clamp_score(score, hi=CSPA_BREATH_COMPONENT_MAX)


def score_stagnation_quality(
    quality: StagnationQuality,
    *,
    solid_ground: bool,
    bar_count: int,
) -> float:
    """停滞品質 — composite_score × 25（Bayes 主分析軸）。"""
    if not solid_ground or bar_count < STAGNATION_MIN_BARS:
        return 0.0
    return _clamp_score(quality.composite_score * CSPA_BREATH_COMPONENT_MAX, hi=CSPA_BREATH_COMPONENT_MAX)


def score_reacceleration(reaccel: Reacceleration, trigger_type: MomentumType) -> float:
    """再加速 — composite_score × 25（Bayes 主分析軸）。"""
    if trigger_type == "NONE":
        return 0.0
    return _clamp_score(reaccel.composite_score * CSPA_BREATH_COMPONENT_MAX, hi=CSPA_BREATH_COMPONENT_MAX)


def calc_structure_score(inp: CspaScoreInput) -> float:
    """骨格スコア 0–100: 方向整合 + 半値戻し + 鮮度 + インパルス減点。"""
    bias = score_cspa_bias_alignment(inp.trade_direction, inp.h1_trend)
    retrace = score_cspa_retrace_beauty(inp.retrace_ratio)
    freshness = score_cspa_structure_freshness(inp.m15_bars_since_impulse)
    impulse_penalty = score_cspa_impulse_quality(inp.impulse_size_atr)
    raw = (
        bias / CSPA_SCORE_BIAS_MAX * 30.0
        + retrace / CSPA_SCORE_RETRACE_MAX * 40.0
        + freshness / CSPA_SCORE_FRESHNESS_MAX * 30.0
        + impulse_penalty
    )
    return _clamp_score(raw)


def calc_market_breath_score(inp: CspaScoreInput) -> tuple[float, float, float, float, float]:
    """呼吸スコア 0–100 と 4 成分。"""
    ctx = inp.trend_context
    rhythm = inp.pullback_rhythm
    quality = inp.stagnation_quality
    reaccel = inp.reacceleration

    trend_score = score_trend_context(ctx, inp.trade_direction) if ctx else (
        min(score_cspa_bias_alignment(inp.trade_direction, inp.h1_trend) / CSPA_SCORE_BIAS_MAX, 1.0)
        * CSPA_BREATH_COMPONENT_MAX
    )
    rhythm_score = score_cspa_correction_rhythm(
        inp.retrace_ratio,
        inp.prior_retrace_ratio,
        rhythm=rhythm,
    )

    if quality:
        stag_score = score_stagnation_quality(
            quality,
            solid_ground=inp.stagnation_solid_ground,
            bar_count=inp.stagnation_bars,
        )
    else:
        stag_score = score_cspa_stagnation(
            inp.stagnation_bars,
            inp.stagnation_avg_body_atr,
            inp.stagnation_solid_ground,
        )
        if CSPA_V2_SCORING:
            stag_score = min(stag_score / CSPA_SCORE_STAGNATION_MAX, 1.0) * CSPA_BREATH_COMPONENT_MAX

    if reaccel:
        reaccel_score = score_reacceleration(reaccel, inp.momentum_type)
    else:
        reaccel_score = score_cspa_momentum(inp.momentum_type, inp.momentum_body_atr)
        if CSPA_V2_SCORING:
            reaccel_score = min(reaccel_score / CSPA_SCORE_MOMENTUM_MAX, 1.0) * CSPA_BREATH_COMPONENT_MAX

    # Bayes 主分析 3 軸 85% + H1 方向感 15%
    breath_core = stag_score + rhythm_score + reaccel_score
    breath_total = _clamp_score(breath_core + trend_score * 0.15, hi=100.0)
    return (
        breath_total,
        trend_score,
        stag_score,
        reaccel_score,
        rhythm_score,
    )


def score_cspa_impulse_quality(impulse_size_atr: float) -> float:
    """Impulse leg size vs M15 ATR — 極端に小さい波は減点（detect 内蔵、breakdown 非表示）。"""
    if impulse_size_atr >= 1.2:
        return 0.0
    if impulse_size_atr >= 0.8:
        return 0.0
    if impulse_size_atr < 0.5:
        return -8.0
    return -4.0


def calc_cspa_candidate_score(inp: CspaScoreInput) -> CspaScoreBreakdown:
    """CSPA L2 candidate_score。

    v2 (default): structure × 0.4 + market_breath × 0.6
    v1 legacy: 6 軸合算（CSPA_V2_SCORING=0）
    """
    bias = score_cspa_bias_alignment(inp.trade_direction, inp.h1_trend)
    retrace = score_cspa_retrace_beauty(inp.retrace_ratio)
    stagnation = score_cspa_stagnation(
        inp.stagnation_bars,
        inp.stagnation_avg_body_atr,
        inp.stagnation_solid_ground,
    )
    momentum = score_cspa_momentum(inp.momentum_type, inp.momentum_body_atr)
    freshness = score_cspa_structure_freshness(inp.m15_bars_since_impulse)
    rhythm = score_cspa_correction_rhythm(
        inp.retrace_ratio,
        inp.prior_retrace_ratio,
        rhythm=inp.pullback_rhythm,
    )
    impulse_penalty = score_cspa_impulse_quality(inp.impulse_size_atr)

    structure_score = calc_structure_score(inp)
    breath_total, trend_ctx_score, stag_qual_score, reaccel_score, rhythm_breath = calc_market_breath_score(
        inp
    )

    if CSPA_V2_SCORING:
        raw_total = (
            structure_score * CSPA_V2_STRUCTURE_WEIGHT
            + breath_total * CSPA_V2_BREATH_WEIGHT
            + float(inp.vp_location_score)
        )
    else:
        raw_total = (
            bias
            + retrace
            + stagnation
            + momentum
            + freshness
            + rhythm
            + impulse_penalty
            + float(inp.vp_location_score)
        )
        structure_score = raw_total
        breath_total = 0.0
        trend_ctx_score = 0.0
        stag_qual_score = stagnation
        reaccel_score = momentum
        rhythm_breath = rhythm

    total = _clamp_score(raw_total)
    return CspaScoreBreakdown(
        bias_alignment=bias,
        retrace_beauty=retrace,
        stagnation=stagnation,
        momentum=momentum,
        structure_freshness=freshness,
        correction_rhythm=rhythm,
        structure_score=round(structure_score, 2),
        market_breath_score=round(breath_total, 2),
        trend_context_score=round(trend_ctx_score, 2),
        stagnation_quality_score=round(stag_qual_score, 2),
        reacceleration_score=round(reaccel_score, 2),
        vp_location_score=float(inp.vp_location_score),
        total=total,
    )


def calc_cspa_candidate_score_total(inp: CspaScoreInput) -> float:
    return calc_cspa_candidate_score(inp).total


def passes_cspa_l2_gate(score: float) -> bool:
    return float(score) >= CSPA_L2_MIN_SCORE


def build_llm_context(setup: CspaSetup) -> dict[str, Any]:
    return {
        "setup_type": SETUP_TYPE,
        "pair": setup.pair,
        "direction": setup.direction,
        "timestamp": setup.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "bias_tf": CSPA_BIAS_TF,
        "structure_tf": CSPA_STRUCTURE_TF,
        "trigger_tf": CSPA_TRIGGER_TF,
        "bias_phase": setup.bias_phase,
        "impulse_direction": setup.impulse.direction,
        "impulse_size_pips": round(setup.impulse.impulse_size / pip_size_for_pair(setup.pair), 1),
        "retrace_ratio": round(setup.retrace_ratio, 4),
        "stagnation_bars": setup.stagnation.bar_count,
        "stagnation_solid_ground": setup.stagnation.solid_ground,
        "momentum_type": setup.momentum.trigger_type,
        "momentum_body_atr": setup.momentum.body_atr,
        "htf_aligned": setup.htf_aligned,
        "h1_trend_direction": setup.h1_trend,
        "candidate_score": int(setup.candidate_score),
        "score_breakdown": setup.score_breakdown.as_dict(),
        "l2_min_score": CSPA_L2_MIN_SCORE,
        "entry_price": round(setup.entry_price, 6),
        "stop_loss": round(setup.stop_loss, 6),
        "take_profit": round(setup.take_profit, 6),
        "tp_mode": setup.tp_mode,
        "risk_reward": round(setup.risk_reward, 2),
        "structure_score": setup.score_breakdown.structure_score,
        "market_breath_score": setup.score_breakdown.market_breath_score,
        "vp_location_score": setup.score_breakdown.vp_location_score,
    }


def _prepare_mtf_frames(
    trigger_df: Any,
    structure_df: Any | None,
    bias_df: Any | None,
) -> tuple[Any, Any, Any]:
    from strategies.bt_ohlcv import BtOhlcvFrame, resample_to_h1, resample_to_m15

    trigger = prepare_trigger_df(trigger_df)
    if isinstance(trigger, BtOhlcvFrame):
        if structure_df is not None and not structure_df.empty:
            structure = prepare_structure_df(structure_df)
        else:
            structure = resample_to_m15(trigger)
        if bias_df is not None and not bias_df.empty:
            bias = prepare_bias_df(bias_df)
        else:
            bias = resample_to_h1(trigger)
        return trigger, structure, bias

    if structure_df is not None and not structure_df.empty:
        structure = prepare_structure_df(structure_df)
    else:
        from main_platform import resample_to_m15

        structure = resample_to_m15(trigger)
    if bias_df is not None and not bias_df.empty:
        bias = prepare_bias_df(bias_df)
    else:
        from main_platform import resample_to_h1

        bias = resample_to_h1(trigger)
    return trigger, structure, bias


def detect_cspa_setups(
    df: pd.DataFrame,
    pair_name: str,
    h1_df: pd.DataFrame | None = None,
    *,
    m15_df: pd.DataFrame | None = None,
    progress_hook: Any | None = None,
    pure_bt: bool | None = None,
    resume_from_bar: int | None = None,
    initial_setups: list[CspaSetup] | None = None,
    initial_last_signal_bar: int = -999,
    on_checkpoint: Callable[[int, list[CspaSetup], int], None] | None = None,
    checkpoint_every: int = 0,
    loop_end: int | None = None,
) -> list[CspaSetup]:
    return CspaStrategy().detect_setups(
        df,
        pair_name,
        h1_df,
        m15_df=m15_df,
        progress_hook=progress_hook,
        pure_bt=pure_bt,
        resume_from_bar=resume_from_bar,
        initial_setups=initial_setups,
        initial_last_signal_bar=initial_last_signal_bar,
        on_checkpoint=on_checkpoint,
        checkpoint_every=checkpoint_every,
        loop_end=loop_end,
    )


def detect_cspa_setups_for_pair(
    df: pd.DataFrame,
    pair_name: str,
    h1_df: pd.DataFrame | None = None,
    *,
    m15_df: pd.DataFrame | None = None,
    progress_hook: Any | None = None,
    pure_bt: bool | None = None,
    resume_from_bar: int | None = None,
    initial_setups: list[CspaSetup] | None = None,
    initial_last_signal_bar: int = -999,
    on_checkpoint: Callable[[int, list[CspaSetup], int], None] | None = None,
    checkpoint_every: int = 0,
    loop_end: int | None = None,
) -> list[CspaSetup]:
    return detect_cspa_setups(
        df,
        pair_name,
        h1_df,
        m15_df=m15_df,
        progress_hook=progress_hook,
        pure_bt=pure_bt,
        resume_from_bar=resume_from_bar,
        initial_setups=initial_setups,
        initial_last_signal_bar=initial_last_signal_bar,
        on_checkpoint=on_checkpoint,
        checkpoint_every=checkpoint_every,
        loop_end=loop_end,
    )


class CspaStrategy(BaseStrategy):
    """Candlestick Price Action (CSPA) — M1 trigger + M15 structure + H1 bias."""

    @property
    def setup_type(self) -> str:
        return SETUP_TYPE

    def detect_setups(
        self,
        df: pd.DataFrame,
        pair_name: str,
        h1_df: pd.DataFrame | None = None,
        *,
        m15_df: pd.DataFrame | None = None,
        progress_hook: Any | None = None,
        pure_bt: bool | None = None,
        resume_from_bar: int | None = None,
        initial_setups: list[CspaSetup] | None = None,
        initial_last_signal_bar: int = -999,
        on_checkpoint: Callable[[int, list[CspaSetup], int], None] | None = None,
        checkpoint_every: int = 0,
        loop_end: int | None = None,
    ) -> list[CspaSetup]:
        if pair_name.upper() not in ALLOWED_PAIRS:
            return []

        pure = resolve_cspa_pure_mode(pure_bt)

        trigger, structure, bias = _prepare_mtf_frames(df, m15_df, h1_df)
        if trigger.empty or structure.empty or bias.empty:
            return []

        consolidation_cache: dict[int, list[ConsolidationZone]] = {}

        from strategies.archive.cspa_scan_context import build_scan_context, build_cspa_bayes_features_np
        from strategies.archive.cspa_scan_engine import (
            atr_at_bar_fast,
            build_pullback_rhythm_fast,
            build_reacceleration_fast,
            build_stagnation_quality_fast,
            build_trend_context_fast,
            classify_bias_dow_phase_fast,
            correction_rhythm_ok_fast,
            detect_stagnation_cluster_fast,
            evaluate_cspa_vp_location_fast,
            find_latest_impulse_fast,
            measure_retrace_ratio_fast,
            m1_over_retraces_structure_fast,
            prior_correction_ratio_fast,
            resolve_momentum_trigger_fast,
            scan_consolidation_zones_fast,
            volatility_percentile_fast,
        )

        scan_ctx = build_scan_context(trigger, structure, bias)
        scan_arrays = scan_ctx.arrays
        bias_swing_highs = scan_ctx.bias_swing_highs
        bias_swing_lows = scan_ctx.bias_swing_lows
        struct_swing_highs = scan_ctx.struct_swing_highs
        struct_swing_lows = scan_ctx.struct_swing_lows
        bias_high_indices = scan_ctx.bias_high_indices
        bias_low_indices = scan_ctx.bias_low_indices
        struct_high_indices = scan_ctx.struct_high_indices
        struct_low_indices = scan_ctx.struct_low_indices

        setups: list[CspaSetup] = list(initial_setups) if initial_setups else []
        start = TRIGGER_WARMUP_BARS
        loop_start = max(start, resume_from_bar) if resume_from_bar is not None else start
        loop_end_idx = loop_end if loop_end is not None else len(trigger)
        total = max(loop_end_idx - start, 1)
        last_signal_bar = initial_last_signal_bar

        for bar_index in range(loop_start, loop_end_idx):
            bars_done = bar_index - start + 1
            if progress_hook is not None:
                progress_hook(bars_done, total)

            if bars_done > 1 and bars_done % CSPA_SCAN_HEARTBEAT_EVERY == 0:
                logger.info(
                    "CSPA scan %s: bar %d/%d setups=%d",
                    pair_name.upper(),
                    bar_index,
                    len(trigger),
                    len(setups),
                )

            if bar_index > loop_start:
                prev_bars_done = (bar_index - 1) - loop_start + 1
                if on_checkpoint and checkpoint_every > 0 and prev_bars_done % checkpoint_every == 0:
                    on_checkpoint(bar_index, setups, last_signal_bar)

            struct_idx = int(scan_arrays.struct_idx_by_bar[bar_index])
            bias_idx = int(scan_arrays.bias_idx_by_bar[bar_index])
            if struct_idx < 0 or bias_idx < 0:
                continue

            phase = classify_bias_dow_phase_fast(scan_ctx, bias_idx)
            if phase in ("RANGE", "NONE"):
                continue

            impulse = find_latest_impulse_fast(scan_ctx, struct_idx, phase)
            if impulse is None:
                continue
            if struct_idx - impulse.end_index > MAX_BARS_SINCE_IMPULSE:
                continue

            retrace = measure_retrace_ratio_fast(scan_arrays, impulse, struct_idx)
            if not pure:
                if retrace < FIB_RETRACE_MIN or retrace > FIB_RETRACE_MAX:
                    continue
                if not correction_rhythm_ok_fast(
                    scan_arrays,
                    impulse,
                    phase,
                    retrace,
                    struct_swing_highs,
                    struct_swing_lows,
                    high_bar_indices=struct_high_indices,
                    low_bar_indices=struct_low_indices,
                    scan_ctx=scan_ctx,
                ):
                    continue
                if m1_over_retraces_structure_fast(scan_arrays, bar_index, impulse):
                    continue

            trade_dir: TradeDirection = "BUY" if phase == "UPTREND" else "SELL"
            stagnation = detect_stagnation_cluster_fast(
                scan_arrays,
                bar_index,
                trade_dir,
                max_bars=STAGNATION_MAX_BARS,
            )
            if stagnation is None:
                continue
            if not pure and not stagnation.solid_ground:
                continue

            momentum = resolve_momentum_trigger_fast(
                scan_arrays, bar_index, trade_dir, stagnation
            )
            if not momentum.detected:
                continue

            vp_allowed, vp_location_score, _vp_levels = evaluate_cspa_vp_location_fast(
                scan_arrays,
                momentum,
                pair_name.upper(),
                trade_dir,
                bar_index,
            )
            if (
                not pure
                and CSPA_VP_SWEEP_FILTER
                and momentum.trigger_type == "SWEEP_ENGULFING"
                and not vp_allowed
            ):
                continue

            if bar_index - last_signal_bar < stagnation.bar_count + 1:
                continue

            htf_direction = scan_ctx.htf_cache.direction_at_ns(
                int(scan_arrays.trigger.datetime_ns[bar_index])
            )
            aligned = not is_counter_trend(trade_dir, htf_direction)  # type: ignore[arg-type]
            if not pure and not aligned:
                continue

            zones = consolidation_cache.get(struct_idx)
            if zones is None:
                zones = scan_consolidation_zones_fast(scan_arrays, struct_idx)
                consolidation_cache[struct_idx] = zones

            levels = build_risk_levels_for_cspa(
                trade_dir,
                momentum.entry_price,
                stagnation,
                momentum,
                pair_name.upper(),
                momentum.atr,
                structure_df=None,
                struct_idx=struct_idx,
                structure_atr=None,
                consolidation_zones=zones,
            )
            if levels is None:
                continue
            stop_loss, take_profit, rr_actual, tp_mode = levels
            if momentum.trigger_type == "SWEEP_ENGULFING" and not fx_sweep_min_risk_ok(
                momentum.entry_price, stop_loss, pair_name.upper(), trade_dir
            ):
                continue

            m15_atr = atr_at_bar_fast(scan_arrays, "structure", struct_idx)
            impulse_size_atr = impulse.impulse_size / m15_atr if m15_atr > 0 else 0.0
            prior_retrace = prior_correction_ratio_fast(
                scan_arrays,
                impulse,
                phase,
                struct_swing_highs,
                struct_swing_lows,
                high_bar_indices=struct_high_indices,
                low_bar_indices=struct_low_indices,
                scan_ctx=scan_ctx,
            )
            vol_pct = volatility_percentile_fast(scan_arrays, bias_idx)
            trend_ctx = build_trend_context_fast(
                scan_arrays,
                bias_idx,
                phase,
                impulse,
                struct_idx,
                vol_pct,
                bias_swing_highs,
                bias_swing_lows,
                high_bar_indices=bias_high_indices,
                low_bar_indices=bias_low_indices,
            )
            pullback_rhythm = build_pullback_rhythm_fast(
                scan_arrays, impulse, struct_idx, retrace
            )
            stagnation_quality = build_stagnation_quality_fast(
                scan_arrays, stagnation, momentum, trade_dir
            )
            reacceleration = build_reacceleration_fast(
                scan_arrays, stagnation, momentum, trade_dir
            )
            score_inp = CspaScoreInput(
                trade_direction=trade_dir,
                h1_trend=htf_direction,  # type: ignore[arg-type]
                retrace_ratio=retrace,
                prior_retrace_ratio=prior_retrace,
                stagnation_bars=stagnation.bar_count,
                stagnation_avg_body_atr=stagnation.avg_body_atr,
                stagnation_solid_ground=stagnation.solid_ground,
                momentum_type=momentum.trigger_type,
                momentum_body_atr=momentum.body_atr,
                m15_bars_since_impulse=struct_idx - impulse.end_index,
                impulse_size_atr=impulse_size_atr,
                trend_context=trend_ctx,
                pullback_rhythm=pullback_rhythm,
                stagnation_quality=stagnation_quality,
                reacceleration=reacceleration,
                vp_location_score=vp_location_score,
            )
            breakdown = calc_cspa_candidate_score(score_inp)
            if not pure and not passes_cspa_l2_gate(breakdown.total):
                continue

            reason_codes: tuple[str, ...] = (
                f"CSPA_{phase}",
                f"CSPA_RETRACE_{int(retrace * 100)}",
                f"CSPA_{momentum.trigger_type}",
                f"CSPA_L2_{int(breakdown.total)}",
                f"CSPA_TP_{tp_mode}",
            )
            if momentum.trigger_type == "SWEEP_ENGULFING" and CSPA_FX_SWEEP_SL:
                reason_codes = (*reason_codes, "CSPA_FX_SL")
            if pure:
                reason_codes = (*reason_codes, "CSPA_PURE_BT")
            bayes_features = build_cspa_bayes_features_np(
                scan_ctx,
                pair=pair_name.upper(),
                bias_phase=phase,
                impulse=impulse,
                struct_idx=struct_idx,
                bias_idx=bias_idx,
                retrace_ratio=retrace,
                stagnation=stagnation,
                momentum=momentum,
                impulse_size_atr=impulse_size_atr,
                trend_context=trend_ctx,
                pullback_rhythm=pullback_rhythm,
                stagnation_quality=stagnation_quality,
                reacceleration=reacceleration,
                tp_mode=tp_mode,
                tp_rr_actual=rr_actual,
                structure_score=breakdown.structure_score,
                market_breath_score=breakdown.market_breath_score,
                volatility_percentile=vol_pct,
                vp_location_score=vp_location_score,
                trigger_bar_index=bar_index,
            )
            setups.append(
                CspaSetup(
                    timestamp=momentum.timestamp,
                    pair=pair_name.upper(),
                    direction=trade_dir,
                    bias_phase=phase,
                    impulse=impulse,
                    retrace_ratio=round(retrace, 4),
                    stagnation=stagnation,
                    momentum=momentum,
                    entry_price=momentum.entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward=rr_actual,
                    tp_mode=tp_mode,
                    candidate_score=breakdown.total,
                    score_breakdown=breakdown,
                    h1_trend=htf_direction,  # type: ignore[arg-type]
                    htf_aligned=aligned,
                    reason_codes=reason_codes,
                    bar_index=bar_index,
                    structure_bar_index=struct_idx,
                    sweep_distance=abs(momentum.trigger_high - momentum.trigger_low),
                    bayes_features=bayes_features,
                )
            )
            last_signal_bar = bar_index

        if on_checkpoint:
            on_checkpoint(loop_end_idx, setups, last_signal_bar)

        return setups

    def analyze_setup(
        self,
        setup: CspaSetup,
        gbp_setup: CspaSetup | None,
        eur_setup: CspaSetup | None,
        h1_gbp: pd.DataFrame,
        h1_eur: pd.DataFrame,
    ) -> StrategyResult:
        h1_ref = h1_gbp if setup.pair == CSPA_PAIR_PRIMARY else h1_eur
        htf = analyze_htf_trend(h1_ref, setup.timestamp)
        aligned = not is_counter_trend(setup.direction, htf.direction)
        if is_cspa_pure_bt_mode():
            aligned = True
        smt_feats = calc_smt_features(
            gbp_setup,
            eur_setup,
            pip_size=pip_size_for_pair(setup.pair),
        )
        llm_ctx = build_llm_context(setup)
        raw: dict[str, Any] = {
            **llm_ctx,
            "smt_intensity": smt_feats.intensity,
            "smt_diff": smt_feats.diff,
            "smt_leader": smt_feats.leader,
            "wick_ratio_pct": 0.0,
            "atr_ratio": round(setup.momentum.body_atr, 4),
            "has_bos": False,
            "both_sweep": gbp_setup is not None and eur_setup is not None,
            "htf_trend_direction": htf.direction,
            "reject_reason": "" if aligned else "REJECT_BY_HTF_TREND",
            "reason_codes": list(setup.reason_codes),
            "candidate_score": setup.candidate_score,
            "score_breakdown": setup.score_breakdown.as_dict(),
            "l2_min_score": CSPA_L2_MIN_SCORE,
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
    "CSPA_BIAS_TF",
    "CSPA_L2_MIN_SCORE",
    "CSPA_PAIR_PRIMARY",
    "CSPA_PAIR_SECONDARY",
    "CSPA_STRUCTURE_TF",
    "CSPA_TRIGGER_TF",
    "CSPA_V2_SCORING",
    "CSPA_BAYES_FEATURE_COLUMNS",
    "CSPA_BAYES_LOW_WEIGHT_FEATURES",
    "ConsolidationZone",
    "CspaBayesFeatures",
    "CspaScoreBreakdown",
    "CspaScoreInput",
    "CspaSetup",
    "CspaStrategy",
    "ImpulseLeg",
    "MomentumSignal",
    "PullbackRhythm",
    "Reacceleration",
    "SETUP_TYPE",
    "STRATEGY_ABBREV",
    "STRATEGY_FULL_NAME",
    "StagnationCluster",
    "StagnationQuality",
    "SwingPoint",
    "TrendContext",
    "TrendPhase",
    "build_cspa_bayes_features",
    "build_cspa_bayes_log_row",
    "build_llm_context",
    "build_pullback_rhythm",
    "build_reacceleration",
    "build_risk_levels",
    "build_risk_levels_with_target",
    "build_stagnation_quality",
    "build_trend_context",
    "calc_cspa_candidate_score",
    "calc_cspa_candidate_score_total",
    "calc_market_breath_score",
    "calc_structure_score",
    "classify_dow_phase_maturity",
    "compute_cspa_session_volume_profile",
    "compute_cspa_trade_excursions",
    "track_cspa_trade_outcome",
    "is_cspa_be_trail_enabled",
    "classify_bias_dow_phase",
    "classify_h4_dow_phase",
    "correction_rhythm_ok",
    "detect_cspa_setups",
    "detect_cspa_setups_for_pair",
    "find_next_consolidation_zone",
    "observe_compression_ratio",
    "observe_range_decay_score",
    "observe_wick_balance",
    "composite_stagnation_quality_score",
    "composite_rhythm_score",
    "composite_reacceleration_score",
    "detect_momentum_breakout",
    "detect_sweep_engulfing_trigger",
    "resolve_cspa_momentum_trigger",
    "build_risk_levels_for_cspa",
    "resolve_cspa_risk_corners",
    "fx_sweep_min_risk_ok",
    "detect_stagnation_cluster",
    "is_cspa_pure_bt_mode",
    "find_latest_impulse",
    "measure_retrace_ratio",
    "passes_cspa_l2_gate",
    "evaluate_cspa_vp_location",
    "prepare_bias_df",
    "prepare_monitor_df",
    "prepare_structure_df",
    "prepare_trigger_df",
    "resolve_cspa_session_type",
    "scale_cspa_take_profit",
    "score_cspa_bias_alignment",
    "score_cspa_correction_rhythm",
    "score_cspa_momentum",
    "score_cspa_retrace_beauty",
    "score_cspa_stagnation",
    "score_cspa_structure_freshness",
]
