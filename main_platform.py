"""
main_platform.py  (v3.2)
==============================
7層多層防御パイプライン（L0〜L7）前処理・検証スクリプト

Fintokei等プロップファーム向け「ロンドン・スイープ・リバーサル」戦略の
セットアップ検知から執行シミュレーション、監査ログ出力までを一括実行する。

v3.2 変更点:
  - L3.5 ベイズ → L4 LLM の実行順逆転（ベイズ REJECT 時 LLM スキップでレイテンシ削減）

v3.1 変更点:
  - Fintokei 2大プロファイル (challenge / funded) と閾値動的切替
  - 利益進捗連動型リスク縮小 (challenge: +8% 接近で 2.5%→0.5%)
  - H1 ベイズアクセラレーター廃止 — ベイズは降格・拒絶専用フィルター

v3.0 変更点:
  - 疎結合マルチ・レジーム: strategies/ + audit/ + main_platform
  - StrategyResult 仲介、戦略レジストリ、risk_manager 4.5% テーパー

v2.0 変更点:
  - SMT: smt_diff / smt_leader を追加（calc_smt_intensity の絶対値互換は維持）
  - L6 CSV: ML用生特徴量5列（smt_diff, smt_leader, wick_ratio_pct, atr_ratio, has_bos）
  - 日次DD: 4%ステップ半減 → multiplier_daily_dd による 0〜5% 連続テーパリング

v1.9 変更点:
  - 物理リスクテーブル: Optuna STABLE (Trial #114) の過学習リスクを踏まえ、
    v1.8 Human-in-the-loop と Optuna 最適解の中間値へ調整
    WEAK_SMT=26, NO_BOS=2, THIN_WICK=8, CORRELATION_FAIL=18

v1.8 変更点:
  - 2026-06-02 週末監査結果に基づくリスク重みの最適化
    WEAK_SMT 強化 (+10)、NO_BOS/THIN_WICK/CORRELATION_FAIL 緩和

v1.7 変更点:
  - M5ベースリスクを 3.5% に再調整
  - 日次DDブレーキ: 当日累積損失 >= 4% で lot_factor を半減（日次DD 5% 保護）
  - H1ベイズ・アクセラレーター / M5黄金フローは v1.6 から変更なし

v1.6 変更点:
  - H1専用ベイズ・アクセラレーター: bayes>=ALLOW_THRES で L4 CAUTION を ALLOW へ昇格
  - 救済時 reason_codes に BAYES_OVERRIDE を記録
  - M5: v1.5 黄金フローを完全隔離維持
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from audit.l2_threshold_manager import (
    DTPA_LLM_REJECT_BELOW,
    dtpa_confidence_lot_multiplier,
    dtpa_llm_decision,
    l2_reject_reason_tags,
    resolve_l2_min_candidate_score as resolve_strategy_l2_min,
)
from audit import risk_manager as audit_rm
from audit import twin_brake as audit_twin_brake
from audit import dd_throttling as audit_dd_throttle
from audit.live_sentinel import (
    evaluate_live_sentinel,
    is_live_sentinel_enabled,
    parse_server_time,
    sentinel_hold_signal,
    sentinel_panic_signal,
)
from audit.risk_manager import (
    AccountState,
    DAILY_DD_TAPER_MAX_PCT,
    MAX_DAILY_EXPOSURE_LIMIT_PCT,
    MAX_DAILY_DD_PCT as RM_MAX_DAILY_DD,
    MAX_MONTHLY_DD_PCT as RM_MAX_MONTHLY,
    PIP_SIZE,
    PIP_VALUE_PER_LOT,
    STARTING_EQUITY as RM_STARTING_EQUITY,
    lot_from_risk_budget,
    multiplier_daily_dd,
    compute_trade_risk_pct,
)
from strategies.base import StrategyResult
from strategies.base_strategy import BaseStrategy
from strategies.archive.london_continuation import ContinuationSetup
from strategies.archive.asian_session_liquidity_sweep import AlsSetup, SETUP_TYPE as ALS_SETUP_TYPE
from strategies.london_sweep_failure import LsfcSetup, SETUP_TYPE as LSFC_SETUP_TYPE
from strategies.archive.fvg_fill import (
    FvgFillSetup,
    FvgFillStrategy,
    SETUP_TYPE as FVG_FILL_SETUP_TYPE,
)
from strategies.archive.tokyo_range_expansion_failure import TrefSetup, SETUP_TYPE as TREF_SETUP_TYPE
from strategies.archive.vexp import SETUP_TYPE as VEXP_SETUP_TYPE, VexpSetup
from strategies.archive.dtpa import DtpaSetup, SETUP_TYPE as DTPA_SETUP_TYPE
from strategies.archive.cspa import CspaSetup, SETUP_TYPE as CSPA_SETUP_TYPE, is_cspa_pure_bt_mode, scale_cspa_take_profit, track_cspa_trade_outcome
from strategies.dinapoli import (
    DiNapoliSetup,
    SETUP_TYPE as DINAPOLI_SETUP_TYPE,
    is_dinapoli_defense_pure_mode,
    is_dinapoli_l4_bypass,
    is_dinapoli_pure_bt_mode,
)
from strategies.dbbs import (
    DbbsSetup,
    SETUP_TYPE as DBBS_SETUP_TYPE,
)
from strategies.dbbs_common import (
    is_dbbs_defense_pure_mode,
    is_dbbs_l4_bypass,
)
from strategies.ttm import (
    TtmSetup,
    SETUP_TYPE as TTM_SETUP_TYPE,
    is_ttm_l4_bypass,
    is_ttm_pure_data_mode,
)
from strategies.ttm_bayes_ev import (
    evaluate_ttm_ev_sizing_for_setup,
    evaluate_ttm_ev_with_runtime,
    is_ttm_ev_sizing_mode,
    should_reject_ttm_bottom20,
)
from strategies.archive.cspa_exit import build_cspa_exit_signal_fields
from strategies.archive.wyckoff_reversal import (
    SpringSetup,
    SETUP_TYPE as WYCKOFF_SETUP_TYPE,
    SETUP_TYPE_LEGACY as WYCKOFF_SETUP_TYPE_LEGACY,
    is_wyckoff_pure_bt_mode,
)
from strategies.archive.liquidity_grab_reversal import (
    LgrSetup,
    SETUP_TYPE as LGR_SETUP_TYPE,
    is_lgr_defense_pure_mode,
    is_lgr_l0_ev_baseline_mode,
    is_lgr_l4_bypass,
    is_lgr_pure_data_mode,
)
from archive.lgr.lgr_ev_position_sizing import evaluate_lgr_ev_sizing_for_setup, is_lgr_ev_sizing_enabled
from src.filters.dn_prop_gate_runtime import prop_gate_enabled, score_dn_prop_gate_from_setup
from src.filters.dn_prop_gate_v1 import dn_prop_gate_base_risk_frac
from archive.lgr.lgr_bayes_gate import (
    LGR_BAYES_REJECT_SOURCE,
    evaluate_lgr_bayes_gate,
    features_from_lgr_setup,
    is_lgr_bayes_gate_enabled,
)

WYCKOFF_SETUP_TYPES = frozenset({WYCKOFF_SETUP_TYPE, WYCKOFF_SETUP_TYPE_LEGACY})
LGR_SETUP_TYPES = frozenset({LGR_SETUP_TYPE})
TTM_SETUP_TYPES = frozenset({TTM_SETUP_TYPE})
from strategies.market_utils import (
    CORRELATED_PAIR,
    SMTFeatures,
    calc_smt_features,
    calc_smt_intensity,
    correlated_pair,
    pip_size_for_pair,
    pair_dataframe_slot,
    positional_index as _positional_index,
    uses_primary_dataframe,
)
from strategies.mtf_timestamp import (
    normalize_bar_timestamp,
    resolve_bar_position,
    resolve_track_start_index as resolve_track_start_index_by_timestamp,
)
from strategies import get_registered_strategies

SetupUnion = LsfcSetup | ContinuationSetup | AlsSetup | FvgFillSetup | TrefSetup | VexpSetup | DtpaSetup | CspaSetup | SpringSetup | LgrSetup | DbbsSetup | DiNapoliSetup

# =============================================================================
# ■ ユーザー設定定数（ここを書き換えるだけでモード切替）
# =============================================================================
MODE_H1 = False  # True=H1(3年BT) / False=M5(5年ベイズ推定用)
PROP_FIRM_PROFILE = "challenge"  # "challenge" | "funded" — Fintokei 2大プロファイル

PROJECT_ROOT = Path(r"C:\Prop_EA_Project")
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
JSON_DIR = LOG_DIR / "audit_json"
CSV_LOG_PATH = LOG_DIR / "final_pipeline_audit_log.csv"
CALENDAR_CACHE_PATH = PROJECT_ROOT / "cache" / "calendar.json"

if MODE_H1:
    GBPUSD_FILE = DATA_DIR / "gbpusd_h1_3y.csv"
    EURUSD_FILE = DATA_DIR / "eurusd_h1_3y.csv"
    TIMEFRAME_LABEL = "H1"
else:
    GBPUSD_FILE = DATA_DIR / "gbpusd_m5_5y.csv"
    EURUSD_FILE = DATA_DIR / "eurusd_m5_5y.csv"
    TIMEFRAME_LABEL = "M5"

# --- 口座・リスク設定 ---
STARTING_EQUITY = RM_STARTING_EQUITY
MAX_DAILY_DD_PCT = RM_MAX_DAILY_DD
MAX_MONTHLY_DD_PCT = RM_MAX_MONTHLY
BASE_RISK_PCT = 0.025 if MODE_H1 else 0.035  # challenge 上限参照 / funded は effective_base_risk で 1%
DAILY_DD_TAPER_HARD_PCT = DAILY_DD_TAPER_MAX_PCT  # 4.5% 安全線

# --- L1 環境チェック ---
DEFAULT_MINUTES_TO_NEWS = 45
NEWS_REJECT_THRESHOLD_MIN = 15

# --- L2 閾値（configure_prop_profile でプロファイル別に上書き） ---
L2_MIN_CANDIDATE_SCORE = 30


def resolve_l2_min_candidate_score(setup_type: str) -> int:
    """戦略別 L2 足切り（プロファイル既定 + l2_threshold_manager）。"""
    return resolve_strategy_l2_min(setup_type, L2_MIN_CANDIDATE_SCORE)


def calculate_vp_position_size(
    vp_zone: str,
) -> tuple[str, float, str, list[str]]:
    """
    VP レイヤー: 執行拒否せず l2_base_lot_factor のみ決定（全戦略共通）。

    Returns:
        l2_regime, l2_base_lot_factor, vp_zone_label, vp_size_tags
    """
    zone = str(vp_zone or "NEUTRAL").upper()
    if zone in ("EXTREME_BAD", "ADVERSE"):
        return "CAUTION", 0.85, "ADVERSE_MOMENTUM", ["VP_SIZE_MOMENTUM"]
    if zone == "SWEEP_ZONE":
        return "ALLOW", 1.0, "SWEEP_ZONE", ["VP_SIZE_MAX"]
    if zone == "FAVORABLE":
        return "CAUTION", 0.75, "FAVORABLE", ["VP_SIZE_FAVORABLE"]
    if zone == "NEUTRAL":
        return "ALLOW", 1.0, "NEUTRAL", ["VP_SIZE_MAX"]
    return "ALLOW", 1.0, zone, ["VP_SIZE_MAX"]


# 後方互換エイリアス
calculate_lsfc_vp_position_size = calculate_vp_position_size


def normalize_smt_score_for_l2(smt_intensity: float) -> float:
    """SMT intensity (pips scale) → 0.0–1.0 正規化スコア。"""
    return max(0.0, min(1.0, float(smt_intensity) / 10.0))


def resolve_adr_used_for_l2(
    setup: SetupUnion,
    raw: dict[str, Any],
) -> float:
    """ADR 消化率 (0=未使用, 1=使い切り)。adr_remaining が残量比率のとき 1-remaining。"""
    adr_remaining: float | None = None
    wf = getattr(setup, "wyckoff_features", None)
    if wf is not None:
        adr_raw = getattr(wf, "adr_remaining", None)
        if adr_raw is not None:
            adr_remaining = float(adr_raw)
    else:
        momentum = getattr(setup, "momentum", None)
        if momentum is not None:
            adr_raw = getattr(momentum, "adr_remaining", None)
            if adr_raw is not None:
                adr_remaining = float(adr_raw)
    if adr_remaining is None and raw.get("adr_remaining") is not None:
        adr_remaining = float(raw["adr_remaining"])
    if adr_remaining is None:
        return 0.50
    return max(0.0, min(1.0, 1.0 - float(adr_remaining)))


def map_location_score_to_vp_zone(location_score: int, *, is_allowed: bool = True) -> str:
    score = int(location_score)
    if score == -20:
        return "EXTREME_BAD"
    if score == 30:
        return "SWEEP_ZONE"
    if score == 10:
        return "FAVORABLE"
    return "NEUTRAL"


def resolve_vp_zone_for_l2(setup: SetupUnion, trigger_df: pd.DataFrame | None) -> str:
    vp_ctx = resolve_volume_profile_context(setup, trigger_df)
    if vp_ctx:
        return map_location_score_to_vp_zone(
            int(vp_ctx.get("location_score", 0)),
            is_allowed=bool(vp_ctx.get("is_allowed", True)),
        )
    return "NEUTRAL"


def resolve_session_type_for_l2(setup: SetupUnion, raw: dict[str, Any]) -> str:
    wf = getattr(setup, "wyckoff_features", None)
    if wf is not None:
        session = str(getattr(wf, "session_type", "") or "")
        if session:
            return session.upper()
    momentum = getattr(setup, "momentum", None)
    if momentum is not None:
        session = str(getattr(momentum, "session_type", "") or "")
        if session:
            return session.upper()
    session = raw.get("session_type")
    if session:
        return str(session).upper()
    return "UNKNOWN"


def uses_l2_math_position_sizer(setup_type: str) -> bool:
    """L2 数式サイザー + L4 SMT 文脈監査（LSFC/ALS 本番系）。"""
    if setup_type in (
        FVG_FILL_SETUP_TYPE,
        TREF_SETUP_TYPE,
        DTPA_SETUP_TYPE,
        CSPA_SETUP_TYPE,
        VEXP_SETUP_TYPE,
    ) or setup_type in WYCKOFF_SETUP_TYPES or setup_type in LGR_SETUP_TYPES or setup_type in TTM_SETUP_TYPES:
        return False
    if is_l4_bypass_setup_type(setup_type):
        return False
    return setup_type in (LSFC_SETUP_TYPE, ALS_SETUP_TYPE, "continuation", "main", "all")


def l2_regime_to_llm_decision(regime: str) -> str:
    if regime == "ALLOW":
        return "ALLOW"
    if regime.startswith("CAUTION"):
        return "CAUTION"
    return "REJECT"


# --- L3.5 ベイズ — 動的スタビライザー（configure_bayes_stabilizer で MODE 別に確定） ---
BAYES_PRIOR_ALPHA = 2.0
BAYES_PRIOR_BETA = 2.0
BAYES_REJECT_THRES = 0.40
BAYES_ALLOW_THRES = 0.55
BAYES_MIN_MATCH_SAMPLES = 3
BAYES_BASE_WIN_RATE = 0.50


def configure_bayes_stabilizer(mode_h1: bool) -> None:
    """
    動的ベイズスタビライザー: タイムフレームごとの事前確率地平線に合わせて
    L3.5 の閾値・学習安定化パラメータを自動割り当てする。
    """
    global BAYES_REJECT_THRES, BAYES_ALLOW_THRES, BAYES_MIN_MATCH_SAMPLES, BAYES_BASE_WIN_RATE

    if mode_h1:
        BAYES_REJECT_THRES = 0.30
        BAYES_ALLOW_THRES = 0.48
        BAYES_MIN_MATCH_SAMPLES = 5
        BAYES_BASE_WIN_RATE = 0.38
    else:
        BAYES_REJECT_THRES = 0.40
        BAYES_ALLOW_THRES = 0.55
        BAYES_MIN_MATCH_SAMPLES = 8
        BAYES_BASE_WIN_RATE = 0.49


configure_bayes_stabilizer(MODE_H1)


def configure_prop_profile(profile: str | None = None, mode_h1: bool | None = None) -> None:
    """
    Fintokei 2大プロファイルに応じ L2 / ベイズ ALLOW / ベースリスク参照値を切替。

    challenge: L2>=30, bayes ALLOW>=0.48, 利益連動 2.5%→0.5%
    funded:    L2>=30, bayes ALLOW>=0.55, 固定 1.0%, CAUTION lot×0.25
    """
    global PROP_FIRM_PROFILE, L2_MIN_CANDIDATE_SCORE, BAYES_ALLOW_THRES, BASE_RISK_PCT

    prof = audit_rm.normalize_profile(profile or PROP_FIRM_PROFILE)
    PROP_FIRM_PROFILE = prof
    use_h1 = MODE_H1 if mode_h1 is None else mode_h1
    configure_bayes_stabilizer(use_h1)

    L2_MIN_CANDIDATE_SCORE = audit_rm.PROFILE_L2_MIN_SCORE[prof]
    BAYES_ALLOW_THRES = audit_rm.PROFILE_BAYES_ALLOW[prof]
    if prof == "funded":
        BASE_RISK_PCT = audit_rm.FUNDED_BASE_RISK_PCT
    else:
        BASE_RISK_PCT = 0.025 if use_h1 else 0.035


configure_prop_profile(PROP_FIRM_PROFILE)

# --- M5 BOS チューニング ---
M5_BOS_MIN_BREAK_PIPS = 3.0
M5_BOS_CLOSE_SMOOTH_BARS = 3
M5_BOS_MIN_BODY_RATIO = 0.30       # 実体が足全体の30%未満ならダマシとみなす
M5_BOS_MAX_ADVERSE_WICK_RATIO = 0.55  # 逆方向ヒゲが55%超なら BOS 無効

# --- L3/L4 LLM ---
MODEL_VERSION = "gemini-3.1-flash-lite"  # GEMINI_MODEL / llm_auditor.resolve_gemini_model() と同期
LSFC_L4_MODEL_VERSION = "RULE_BASE_ONLY"
RULE_BASE_ONLY_SETUP_TYPES = frozenset({LSFC_SETUP_TYPE, ALS_SETUP_TYPE, VEXP_SETUP_TYPE, CSPA_SETUP_TYPE}) | WYCKOFF_SETUP_TYPES | LGR_SETUP_TYPES | TTM_SETUP_TYPES
BAYES_BYPASS_SETUP_TYPES = WYCKOFF_SETUP_TYPES | LGR_SETUP_TYPES | TTM_SETUP_TYPES
BAYES_BYPASS_NEUTRAL_PROBABILITY = 1.0
# CSPA L3.5: audit.cspa_bayes_gate.evaluate_cspa_bayes_gate (CSPABayesEngine 3-Tier)

# Pure BT 契約: 以下の decision_source / tags が 1 件でも出たらデータ無効。
PURE_BT_FORBIDDEN_DECISION_SOURCES = frozenset(
    {
        "REJECT_BY_L0",
        "REJECT_BY_L1",
        "REJECT_BY_L2",
        "REJECT_BY_BAYES",
        "REJECT_BY_HTF_TREND",
        "REJECT_BY_LLM",
        "REJECT_BY_L4",
        "REJECT_BY_DAILY_STOP",
        "MUTUAL_EXCLUSION_LOCK",
        "REJECT_BY_TREF_LOSS_PATTERN",
    }
)
PURE_BT_FORBIDDEN_TAGS = frozenset(
    {
        "DAILY_DD_BRAKE",
        audit_rm.REASON_PROFIT_CUSHION_BRAKE,
        "DAILY_EXPOSURE_LIMIT_EXCEEDED",
        "DAILY_EXPOSURE_CAPPED",
        audit_dd_throttle.REASON_DD_THROTTLING_HALF,
        audit_dd_throttle.REASON_DD_THROTTLING_QUARTER,
        audit_dd_throttle.REASON_RECOVERY_BOOST,
        audit_dd_throttle.REASON_DAILY_STOP,
        audit_rm.REASON_SYMBOL_ONE_POSITION_LIMIT,
        "TWIN_BRAKE_PROXIMITY",
        "TWIN_BRAKE_DAILY",
        "TWIN_BRAKE_DAILY_HARD_STOP",
        "TWIN_BRAKE_ACTIVE",
    }
)


def is_defense_pure_setup(setup_type: str) -> bool:
    """CSPA / WR / LGR pure BT — L0〜L4.5 防御を完全スキップするか。"""
    if setup_type == CSPA_SETUP_TYPE:
        return is_cspa_pure_bt_mode()
    if setup_type in WYCKOFF_SETUP_TYPES:
        return is_wyckoff_pure_bt_mode()
    if setup_type in LGR_SETUP_TYPES:
        return is_lgr_defense_pure_mode()
    if setup_type in TTM_SETUP_TYPES:
        return is_ttm_pure_data_mode()
    if setup_type == DINAPOLI_SETUP_TYPE:
        return is_dinapoli_defense_pure_mode()
    if setup_type == DBBS_SETUP_TYPE:
        return is_dbbs_defense_pure_mode()
    return False


def assert_pure_bt_pending_contract(pending: PendingEvaluation) -> None:
    """Pure BT 契約違反を検出（テスト / 短期 BT 検証用）。"""
    ds = pending.decision_source
    if ds.startswith("REJECT") or ds in PURE_BT_FORBIDDEN_DECISION_SOURCES:
        raise AssertionError(f"pure BT forbidden decision_source: {ds}")
    if ds not in ("ALLOW", "CAUTION"):
        raise AssertionError(f"pure BT unexpected decision_source: {ds}")
    if pending.is_reject:
        raise AssertionError("pure BT pending.is_reject must be False")
    if pending.bayes_probability != BAYES_BYPASS_NEUTRAL_PROBABILITY:
        raise AssertionError(
            f"pure BT bayes_probability must be {BAYES_BYPASS_NEUTRAL_PROBABILITY}, "
            f"got {pending.bayes_probability}"
        )
    if pending.lot_factor <= 0.0:
        raise AssertionError(f"pure BT lot_factor must be > 0, got {pending.lot_factor}")
    tag_set = set(pending.tags)
    forbidden_tags = tag_set & PURE_BT_FORBIDDEN_TAGS
    if forbidden_tags:
        raise AssertionError(f"pure BT forbidden tags: {sorted(forbidden_tags)}")


def is_bayes_bypass_setup_type(setup_type: str) -> bool:
    """L3.5 ベイズ（ハード拒否・ロット倍率・学習）を完全スキップする戦略。"""
    if setup_type in WYCKOFF_SETUP_TYPES:
        if is_wyckoff_pure_bt_mode():
            return True
        from audit.wyckoff_bayes_gate import is_wyckoff_l4_bypass

        return is_wyckoff_l4_bypass()
    if setup_type in LGR_SETUP_TYPES:
        return not is_lgr_bayes_gate_enabled()
    if setup_type == CSPA_SETUP_TYPE and is_cspa_pure_bt_mode():
        return True
    if setup_type == DINAPOLI_SETUP_TYPE:
        from strategies.dinapoli import is_dinapoli_generic_bayes_bypass

        return is_dinapoli_generic_bayes_bypass()
    if setup_type == DBBS_SETUP_TYPE:
        from strategies.dbbs_common import is_dbbs_generic_bayes_bypass

        return is_dbbs_generic_bayes_bypass()
    return setup_type in BAYES_BYPASS_SETUP_TYPES


def is_l4_bypass_setup_type(setup_type: str) -> bool:
    """LSFC / ALS / TREF 等、L4 Gemini を完全スキップする戦略。"""
    if setup_type == LSFC_SETUP_TYPE:
        from strategies.london_sweep_failure import is_lsfc_l4_bypass

        return is_lsfc_l4_bypass()
    if setup_type in WYCKOFF_SETUP_TYPES:
        if is_wyckoff_pure_bt_mode():
            return True
        from audit.wyckoff_bayes_gate import is_wyckoff_l4_bypass

        return is_wyckoff_l4_bypass()
    if setup_type in LGR_SETUP_TYPES:
        return is_lgr_l4_bypass()
    if setup_type in TTM_SETUP_TYPES:
        return is_ttm_l4_bypass()
    if setup_type == CSPA_SETUP_TYPE and is_cspa_pure_bt_mode():
        return True
    if setup_type == DINAPOLI_SETUP_TYPE and is_dinapoli_l4_bypass():
        return True
    if setup_type == DBBS_SETUP_TYPE and is_dbbs_l4_bypass():
        return True
    if setup_type == TREF_SETUP_TYPE:
        from strategies.archive.tokyo_range_expansion_failure import load_tref_config

        return load_tref_config().l4_bypass
    return setup_type in RULE_BASE_ONLY_SETUP_TYPES
LLM_LATENCY_MIN_MS = 300
LLM_LATENCY_MAX_MS = 450
USE_LLM_AUDITOR = False  # live: mt5_bridge 起動時に configure_live_runtime(True) で有効化


def configure_live_runtime(enable_llm: bool = True) -> None:
    """MT5 Bridge ライブ運用モード（LLM監査 ON/OFF）。バックテストは False のまま。"""
    global USE_LLM_AUDITOR
    USE_LLM_AUDITOR = bool(enable_llm)

RISK_TAG_WEIGHTS: dict[str, int] = {
    # -------------------------------------------------------------------------
    # レガシー / Optuna 用タグ重み（simulate_llm_risk_audit + param_optimizer）
    #
    # 【タグ二重管理 — 非対称性警告】
    # ライブ L4 は llm_auditor.GEMINI_ASSIGNABLE_TAGS + Gemini JSON 監査。
    # 本テーブルは Optuna が探索する simulate 経路専用であり、Gemini タグ体系と
    # 1:1 対応しない。AGAINST_HTF_TREND (Gemini 専用・重み25) はここに存在せず、
    # Optuna 最適化の対象外 — バックテスト BT 結果と Live 結果を直接比較しないこと。
    # -------------------------------------------------------------------------
    # v1.9: 物理リスクテーブル (v1.8 Human x Optuna STABLE 中間 — 過学習抑制)
    "HIGH_ATR": 15,
    "OVER_TRADING_WARNING": 20,
    "NEWS_SOON": 25,
    "LOW_LIQUIDITY": 10,
    "CONSECUTIVE_LOSSES": 15,
    "WEAK_SMT": 26,           # v1.8: 20 / Optuna: 32 → 中間 (重複時 REJECT 維持)
    "NO_BOS": 2,              # v1.8=Optuna=2 (構造未確定でも執行する優位性を確定)
    "CORRELATION_FAIL": 18,   # v1.8: 15 / Optuna: 23 → ややタイト
    "THIN_WICK": 8,           # v1.8: 3 / Optuna: 13 → 過剰防衛を緩和
    "BAYES_OVERRIDE": 0,
    "DAILY_DD_BRAKE": 0,
    "DAILY_EXPOSURE_LIMIT_EXCEEDED": 0,
    "LLM_TIMEOUT": 45,
    "LLM_TIMEOUT_FALLBACK": 50,
    "LLM_PARSE_ERROR": 35,
}

OPTIMIZABLE_RISK_TAGS: tuple[str, ...] = (
    # Optuna 探索対象 — GEMINI_ASSIGNABLE_TAGS の部分集合のみ（非対称）
    "WEAK_SMT",
    "NO_BOS",
    "THIN_WICK",
    "CORRELATION_FAIL",
)


@contextmanager
def risk_weight_override(overrides: dict[str, int] | None) -> Iterator[dict[str, int]]:
    """RISK_TAG_WEIGHTS を一時上書き（Optuna 等のオフライン最適化用）。"""
    backup = dict(RISK_TAG_WEIGHTS)
    if overrides:
        RISK_TAG_WEIGHTS.update(overrides)
    try:
        yield RISK_TAG_WEIGHTS
    finally:
        RISK_TAG_WEIGHTS.clear()
        RISK_TAG_WEIGHTS.update(backup)


# --- セッション時間（CSVのサーバー時刻ベース） ---
LONDON_SESSION_HOURS = range(15, 21)  # 15:00〜20:59
NY_ENTRY_HOUR = 21  # NY開始直後のエントリー判定バー

# --- L5 未来追跡 ---
MAX_HOLDING_BARS = 48  # 最大保有バー数（H1=48h / M5=4h）

CSV_COLUMNS = [
    "trade_id",
    "timestamp",
    "pair",
    "equity_before_trade",
    "equity_after_trade",
    "daily_dd_remaining_percent",
    "monthly_dd_remaining_percent",
    "setup_type",
    "candidate_score",
    "bayes_probability",
    "smt_intensity",
    "model_version",
    "reason_codes",
    "risk_score",
    "llm_latency_ms",
    "decision_source",
    "lot_factor",
    "llm_score",
    "final_lot_size",
    "entry_price",
    "stop_loss",
    "take_profit",
    "trade_result",
    "profit_loss",
    "profit_r",
    "holding_time",
    "shadow_result",
    "shadow_profit_r",
    "smt_diff",
    "smt_leader",
    "wick_ratio_pct",
    "atr_ratio",
    "has_bos",
    "wyckoff_features",
    "lgr_features",
    "ttm_features",
    "vp_zone",
    "l2_regime",
    "l2_base_lot_factor",
    "htf_trend",
    "divergence_direction",
    "l4_multiplier",
    "l4_smt_interpretation",
    "htf_counter_trend",
    "htf_lot_multiplier",
    "fvg_final_lot_factor",
    "ev_rank",
    "ev_lot_multiplier",
    "sized_result_r",
]


# =============================================================================
# データ読み込み
# =============================================================================
def load_ohlcv(filepath: Path) -> pd.DataFrame:
    """FT6エクスポートCSVを読み込み、datetimeインデックス付きDataFrameへ変換。"""
    df = pd.read_csv(filepath)
    df.columns = [c.strip("<>").upper() for c in df.columns]

    time_str = df["TIME"].astype(str).str.zfill(4)
    dt_str = df["DTYYYYMMDD"].astype(str) + time_str
    df["datetime"] = pd.to_datetime(dt_str, format="%Y%m%d%H%M")
    df = df.rename(
        columns={
            "OPEN": "open",
            "HIGH": "high",
            "LOW": "low",
            "CLOSE": "close",
            "VOL": "volume",
        }
    )
    df = df.sort_values("datetime").reset_index(drop=True)
    return df[["datetime", "open", "high", "low", "close", "volume"]]


def resample_to_h1(df: pd.DataFrame) -> pd.DataFrame:
    """M5データをH1へリサンプル（セットアップ検知用）。"""
    from strategies.bt_ohlcv import BtOhlcvFrame, resample_to_h1 as bt_resample_to_h1

    if isinstance(df, BtOhlcvFrame):
        return bt_resample_to_h1(df)
    indexed = df.set_index("datetime")
    h1 = indexed.resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return h1.dropna(subset=["open"]).reset_index()


def resample_to_m15(df: pd.DataFrame) -> pd.DataFrame:
    """M5/M1 データを M15 へリサンプル（下位足トリガー用）。"""
    from strategies.bt_ohlcv import BtOhlcvFrame, resample_to_m15 as bt_resample_to_m15

    if isinstance(df, BtOhlcvFrame):
        return bt_resample_to_m15(df)
    indexed = df.set_index("datetime")
    m15 = indexed.resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return m15.dropna(subset=["open"]).reset_index()


def resample_to_h4(df: pd.DataFrame) -> pd.DataFrame:
    """H1 データを H4 へリサンプル（DBBS ATR 参照用）。"""
    from strategies.bt_ohlcv import BtOhlcvFrame, resample_bars_ns

    if df.empty:
        return df
    if isinstance(df, BtOhlcvFrame):
        frame = df
    else:
        frame = BtOhlcvFrame.from_pandas(df)
    bar_ns = int(np.timedelta64(240, "m") / np.timedelta64(1, "ns"))
    return resample_bars_ns(frame, bar_ns).to_pandas()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range（ATR）を計算。"""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


# Strategy layer -> strategies/london_sweep_failure.py (LSFC)

# =============================================================================
# L3.5 ベイズ推定エンジン（H1 / M5 共通）
# =============================================================================
def _smt_bucket(smt_intensity: float) -> str:
    if smt_intensity < 3.0:
        return "SMT_WEAK"
    if smt_intensity < 7.0:
        return "SMT_MID"
    return "SMT_STRONG"


def _score_bucket(candidate_score: float) -> str:
    if candidate_score < 40.0:
        return "SCORE_LOW"
    if candidate_score < 60.0:
        return "SCORE_MID"
    return "SCORE_HIGH"


def _streak_bucket(consecutive_losses: int) -> str:
    if consecutive_losses == 0:
        return "STREAK_0"
    if consecutive_losses <= 2:
        return "STREAK_1_2"
    return "STREAK_3PLUS"


@dataclass
class BayesObservation:
    """ベイズ学習用の1件の観測（シャドー・実執行の結果を統合）。"""

    timestamp: pd.Timestamp
    smt_bucket: str
    score_bucket: str
    streak_bucket: str
    has_bos: bool
    both_sweep: bool
    won: bool


class BayesEngine:
    """
    L3.5: 条件付き勝率（事後確率）をウォークフォワードで推定。

    過去のシャドー/実執行結果を特徴量バケットごとに蓄積し、
    Beta-Binomial 共役事前分布で事後確率を算出する。
    現在イベントより未来の観測は一切参照しない（Look-ahead 排除）。
    """

    def __init__(self) -> None:
        self.observations: list[BayesObservation] = []

    def reset(self) -> None:
        """ウォークフォワード学習データを完全初期化（BT 再走・汚染除去用）。"""
        self.observations.clear()

    def record_outcome(
        self,
        timestamp: pd.Timestamp,
        smt_intensity: float,
        candidate_score: float,
        consecutive_losses: int,
        has_bos: bool,
        both_sweep: bool,
        won: bool,
    ) -> None:
        """L5追跡後: シャドー/実執行の勝敗を学習データへ追加。"""
        self.observations.append(
            BayesObservation(
                timestamp=timestamp,
                smt_bucket=_smt_bucket(smt_intensity),
                score_bucket=_score_bucket(candidate_score),
                streak_bucket=_streak_bucket(consecutive_losses),
                has_bos=has_bos,
                both_sweep=both_sweep,
                won=won,
            )
        )

    def _global_win_anchor(self, history: list[BayesObservation]) -> float:
        """全体履歴のベース勝率（ラプラス平滑化）。不足時はモード別ベース勝率へ。"""
        if not history:
            return BAYES_BASE_WIN_RATE
        wins = sum(1 for o in history if o.won)
        return (wins + BAYES_PRIOR_ALPHA) / (
            len(history) + BAYES_PRIOR_ALPHA + BAYES_PRIOR_BETA
        )

    def compute_probability(
        self,
        timestamp: pd.Timestamp,
        smt_intensity: float,
        candidate_score: float,
        consecutive_losses: int,
        has_bos: bool,
        both_sweep: bool,
    ) -> float:
        """
        条件付き勝率 P(Win | Features) を返す（0.0〜1.0）。

        v1.4 コールドスタート安定化:
          - 条件一致サンプルが BAYES_MIN_MATCH_SAMPLES 未満の場合、
            局所事後確率をグローバルベース勝率へ線形回帰（過剰 REJECT 防止）
          - 履歴ゼロ時は BAYES_BASE_WIN_RATE（H1=0.38 / M5=0.49）を返す
        """
        history = [o for o in self.observations if o.timestamp < timestamp]
        anchor = self._global_win_anchor(history)

        if not history:
            return round(BAYES_BASE_WIN_RATE, 4)

        target = {
            "smt_bucket": _smt_bucket(smt_intensity),
            "score_bucket": _score_bucket(candidate_score),
            "streak_bucket": _streak_bucket(consecutive_losses),
            "has_bos": has_bos,
            "both_sweep": both_sweep,
        }

        filter_layers: list[dict[str, Any]] = [
            target,
            {k: v for k, v in target.items() if k != "both_sweep"},
            {k: v for k, v in target.items() if k not in ("both_sweep", "has_bos")},
            {"smt_bucket": target["smt_bucket"], "score_bucket": target["score_bucket"]},
            {"smt_bucket": target["smt_bucket"]},
            {},
        ]

        matched: list[BayesObservation] = []
        for layer in filter_layers:
            layer_matched = [
                o for o in history
                if all(getattr(o, key) == val for key, val in layer.items())
            ]
            if layer_matched:
                matched = layer_matched
                break

        if not matched:
            return round(anchor, 4)

        wins = sum(1 for o in matched if o.won)
        sample_n = len(matched)
        local_posterior = (wins + BAYES_PRIOR_ALPHA) / (
            sample_n + BAYES_PRIOR_ALPHA + BAYES_PRIOR_BETA
        )

        if sample_n < BAYES_MIN_MATCH_SAMPLES:
            blend_weight = sample_n / BAYES_MIN_MATCH_SAMPLES
            posterior = blend_weight * local_posterior + (1.0 - blend_weight) * anchor
        else:
            posterior = local_posterior

        return round(posterior, 4)


def check_bayes_hard_reject(bayes_probability: float) -> bool:
    """L3.5 ハード拒否: REJECT 閾値未満なら LLM 監査をスキップ可能。"""
    return bayes_probability < BAYES_REJECT_THRES


def apply_bayes_downgrade(llm_decision: str, bayes_probability: float) -> str:
    """
    L3.5 降格のみ（LLM 通過後）。bayes >= REJECT_THRES を前提。

    - bayes < ALLOW_THRES かつ LLM=ALLOW → CAUTION
    - それ以外 → LLM 判定を維持（昇格なし）
    """
    if bayes_probability < BAYES_ALLOW_THRES and llm_decision == "ALLOW":
        return "CAUTION"
    return llm_decision


def apply_bayes_gate(llm_decision: str, bayes_probability: float) -> str:
    """
    L3.5 純粋安全フィルター（v3.1）: 降格・拒絶のみ。昇格は行わない。

    v3.2: ライブでは check_bayes_hard_reject → LLM → apply_bayes_downgrade の順で使用。
    本関数は後方互換・単体テスト用の合成ラッパー。
    """
    if check_bayes_hard_reject(bayes_probability):
        return "REJECT_BY_BAYES"
    return apply_bayes_downgrade(llm_decision, bayes_probability)


def resolve_model_version(setup_type: str) -> str:
    """L6 CSV model_version — L4 バイパス戦略は RULE_BASE_ONLY。"""
    if is_l4_bypass_setup_type(setup_type):
        return LSFC_L4_MODEL_VERSION
    return MODEL_VERSION


def _skipped_llm_audit() -> tuple[list[str], int, int, str, int, str]:
    """L4 LLM 監査をスキップした場合のプレースホルダー（Gemini 未呼び出し）。"""
    return [], 0, 0, "REJECT", 0, ""


def _rule_base_l4_bypass_result(
    htf_trend_direction: str,
    htf_would_block: bool,
    setup_type: str = LSFC_SETUP_TYPE,
) -> tuple[list[str], int, int, str, int, str, str]:
    """
    L4 完全バイパス: Gemini API を呼ばず decision_source=ALLOW（RULE_BASE_ONLY）。

    対象: LSFC（Strategy A）。
    """
    tags = ["L4_BYPASS"]
    if setup_type == LSFC_SETUP_TYPE:
        tags.append("LSFC_L4_BYPASS")
    elif setup_type == ALS_SETUP_TYPE:
        tags.append("ALS_L4_BYPASS")
    elif setup_type == TREF_SETUP_TYPE:
        tags.append("TREF_L4_BYPASS")
    elif setup_type == CSPA_SETUP_TYPE:
        tags.append("CSPA_L4_BYPASS")
    elif setup_type in WYCKOFF_SETUP_TYPES:
        tags.append("WYCKOFF_L4_BYPASS")
    elif setup_type in LGR_SETUP_TYPES:
        tags.append("LGR_L4_BYPASS")
    elif setup_type in TTM_SETUP_TYPES:
        tags.append("TTM_L4_BYPASS")
    elif setup_type == DBBS_SETUP_TYPE:
        tags.append("DBBS_L4_BYPASS")
    else:
        tags.append("RULE_BASE_L4_BYPASS")
    if htf_would_block:
        tags.append(f"HTF_TREND_{htf_trend_direction}")
        tags.append("HTF_WOULD_BLOCK")
    return tags, 0, 0, "ALLOW", 0, "", "ALLOW"


def has_rule_base_l4_bypass_tag(tags: list[str] | tuple[str, ...]) -> bool:
    """reason_codes に L4 直列バイパスが含まれるか。"""
    tag_set = set(tags)
    return bool(
        tag_set
        & {"L4_BYPASS", "LSFC_L4_BYPASS", "ALS_L4_BYPASS", "TREF_L4_BYPASS", "CSPA_L4_BYPASS", "WYCKOFF_L4_BYPASS", "LGR_L4_BYPASS", "TTM_L4_BYPASS", "DBBS_L4_BYPASS", "RULE_BASE_L4_BYPASS"}
    )


def merge_rule_base_l4_bypass_tags(
    tags: list[str],
    setup_type: str,
    decision_source: str,
    *,
    htf_trend_direction: str = "NEUTRAL",
    htf_would_block: bool = False,
) -> list[str]:
    """
    RULE_BASE_ONLY 戦略の ALLOW/CAUTION では L4 バイパスタグを必ず維持する。
    ピラミッド L5 経路や後段ロジックによる tags 消失を防ぐ。
    """
    if not is_l4_bypass_setup_type(setup_type):
        return tags
    if decision_source not in ("ALLOW", "CAUTION"):
        return tags
    if has_rule_base_l4_bypass_tag(tags):
        merged = list(tags)
    else:
        merged = list(_rule_base_l4_bypass_result(
            htf_trend_direction, htf_would_block, setup_type=setup_type
        )[0])
        for existing in tags:
            if existing not in merged:
                merged.append(existing)
    return merged


# 後方互換エイリアス
_lsfc_l4_bypass_result = _rule_base_l4_bypass_result


# AccountState -> audit/risk_manager.py

@dataclass
class PendingEvaluation:
    """同一タイムスタンプ内の Phase-1 執行判定結果（L5反映前）。"""

    trade_id: str
    setup_type: str
    setup: SetupUnion
    gbp_s: SetupUnion | None
    eur_s: SetupUnion | None
    equity_before: float
    daily_rem: float
    monthly_rem: float
    smt: float
    smt_diff: float
    smt_leader: str
    has_bos: bool
    candidate_score: float
    atr_ratio: float
    both_sweep: bool
    tags: list[str]
    risk_score: int
    latency: int
    decision_source: str
    is_reject: bool
    bayes_probability: float
    consecutive_losses_snapshot: int
    profile: str
    llm_eligible: bool
    risk_budget: float
    lot_size: float
    lot_factor: float
    trade_risk_pct: float
    minutes_to_news: int
    start_idx: int
    llm_confidence_score: int = 0
    llm_reason_summary: str = ""
    confidence_lot_multiplier: float = 0.0
    final_lot_size: float = 0.0
    force_close_at_timeout: bool = False
    timeout_server_hour: int = 0
    htf_trend_direction: str = "NEUTRAL"
    vp_zone: str = ""
    l2_regime: str = ""
    l2_base_lot_factor: float = 0.0
    htf_trend: str = "NEUTRAL"
    divergence_direction: str = ""
    l4_multiplier: float = 1.0
    l4_smt_interpretation: str = ""
    htf_counter_trend: bool = False
    htf_lot_multiplier: float = 1.0
    fvg_final_lot_factor: float = 0.0
    cspa_gate_reason: str = ""
    cspa_tp_multiplier: float = 1.0
    lgr_bayes_regime: str = ""
    lgr_bayes_reason: str = ""
    lgr_ev_score: float = 0.0
    lgr_ev_rank: float = 0.0
    lgr_ev_lot_multiplier: float = 1.0
    ttm_bayes_win_prob: float = 0.0
    ttm_ev_rank: float = 0.0
    ttm_ev_lot_multiplier: float = 1.0
    dn_ev_rank: float = 0.0
    dn_ev_bucket: str = ""
    dn_ev_rank_v2: float = 0.0
    dn_prop_gate_tier: str = ""
    dn_prop_gate_lot_multiplier: float = 1.0


# =============================================================================
# L1 環境チェック（ダミー）
# =============================================================================
def simulate_minutes_to_news(ts: pd.Timestamp) -> int:
    """
    主要指標発表までの分数をダミーシミュレーション。
    初期設定（DEFAULT_MINUTES_TO_NEWS=45分）を中心に、日ごとの擬似変動を加える。
    """
    rng = random.Random(int(ts.timestamp()) % 9973)
    offset = rng.randint(-40, 75)
    return max(5, DEFAULT_MINUTES_TO_NEWS + offset)


def resolve_minutes_to_news(ts: pd.Timestamp, override: int | None = None) -> int:
    """
    L1 用: MT5/API override → calendar.json キャッシュ → ダミー の順で分数を解決。

    calendar_service は別プロセス。ここでは cache ファイルの読み取りのみ（高速）。
    """
    if override is not None:
        return int(override)
    try:
        from calendar_service import get_minutes_to_next_news

        minutes, _, _ = get_minutes_to_next_news(ts, cache_path=CALENDAR_CACHE_PATH)
        if minutes is not None:
            return max(1, int(minutes))
    except Exception:
        pass
    return simulate_minutes_to_news(ts)


def _read_calendar_cache_for_live(
    bar_timestamp: pd.Timestamp,
) -> tuple[int | None, str, int | None]:
    """Live API 用: calendar.json から分数/重要度/ミリ秒を読み取る（読み取り専用）。"""
    try:
        from calendar_service import get_minutes_to_next_news

        return get_minutes_to_next_news(bar_timestamp, cache_path=CALENDAR_CACHE_PATH)
    except Exception:
        return None, "", None


# =============================================================================
# L3/L4 LLMリスク監査シミュレーション
# =============================================================================
def _setup_type_for_llm(setup: Any) -> str:
    if isinstance(setup, FvgFillSetup):
        return FVG_FILL_SETUP_TYPE
    if isinstance(setup, ContinuationSetup):
        return "LONDON_CONTINUATION"
    if isinstance(setup, LsfcSetup):
        return LSFC_SETUP_TYPE
    if isinstance(setup, AlsSetup):
        return ALS_SETUP_TYPE
    if isinstance(setup, TrefSetup):
        return TREF_SETUP_TYPE
    if isinstance(setup, VexpSetup):
        return VEXP_SETUP_TYPE
    if isinstance(setup, DtpaSetup):
        return DTPA_SETUP_TYPE
    if isinstance(setup, CspaSetup):
        return CSPA_SETUP_TYPE
    if isinstance(setup, SpringSetup):
        return WYCKOFF_SETUP_TYPE
    if isinstance(setup, LgrSetup):
        return LGR_SETUP_TYPE
    if isinstance(setup, DiNapoliSetup):
        return DINAPOLI_SETUP_TYPE
    if isinstance(setup, DbbsSetup):
        return DBBS_SETUP_TYPE
    return "LONDON_CONTINUATION"


def run_tref_llm_risk_audit(
    setup: TrefSetup,
    raw: dict[str, Any],
    *,
    minutes_to_news: int = 999,
) -> tuple[list[str], int, int, str, int, str]:
    """TREF 専用 Gemini 監査（Idempotency キャッシュ）。"""
    from audit.gemini_tref_auditor import audit_tref_setup

    audit_input = dict(raw)
    audit_input["candidate_score"] = raw.get("candidate_score", setup.candidate_score)
    audit_input["minutes_to_news"] = minutes_to_news
    result = audit_tref_setup(setup, audit_input)
    confidence = int(result.get("confidence_score", 0))
    tags = list(result.get("reason_codes", ["TREF_GEMINI_AUDIT"]))
    return (
        tags,
        int(result.get("risk_score", max(0, 100 - confidence))),
        int(result.get("llm_latency_ms", 0)),
        str(result.get("llm_decision", audit_rm.confidence_to_llm_decision(confidence))),
        confidence,
        str(result.get("reason_summary", "")),
    )


def run_dtpa_llm_risk_audit(
    setup: DtpaSetup,
    raw: dict[str, Any],
    *,
    minutes_to_news: int = 999,
) -> tuple[list[str], int, int, str, int, str]:
    """DTPA 専用 Gemini 監査（Idempotency キャッシュ）。"""
    from audit.gemini_dtpa_auditor import audit_dtpa_setup

    audit_input = dict(raw)
    audit_input["candidate_score"] = raw.get("candidate_score", setup.candidate_score)
    audit_input["minutes_to_news"] = minutes_to_news
    result = audit_dtpa_setup(setup, audit_input)
    confidence = int(result.get("confidence_score", 0))
    tags = list(result.get("reason_codes", ["DTPA_GEMINI_AUDIT"]))
    return (
        tags,
        int(result.get("risk_score", max(0, 100 - confidence))),
        int(result.get("llm_latency_ms", 0)),
        str(result.get("llm_decision", audit_rm.confidence_to_llm_decision(confidence))),
        confidence,
        str(result.get("reason_summary", "")),
    )


def run_fvg_llm_risk_audit(
    strategy: BaseStrategy,
    setup: FvgFillSetup,
    raw: dict[str, Any],
) -> tuple[list[str], int, int, str, int, str]:
    """FVG 専用 Gemini 監査（Bar-Lock + Idempotency キャッシュ）。"""
    from audit.gemini_fvg_auditor import audit_fvg_setup

    audit_input = dict(raw)
    audit_input["candidate_score"] = raw.get("candidate_score", 0.0)

    def _invoke() -> dict[str, Any]:
        return audit_fvg_setup(setup, audit_input)

    if isinstance(strategy, FvgFillStrategy):
        result = strategy.audit_with_bar_lock(setup.pair, setup.timestamp, _invoke)
    else:
        result = _invoke()

    confidence = int(result.get("confidence_score", 0))
    tags = list(result.get("reason_codes", ["FVG_GEMINI_AUDIT"]))
    return (
        tags,
        int(result.get("risk_score", max(0, 100 - confidence))),
        int(result.get("llm_latency_ms", 0)),
        str(result.get("llm_decision", audit_rm.confidence_to_llm_decision(confidence))),
        confidence,
        str(result.get("reason_summary", "")),
    )


def simulate_llm_risk_audit(
    setup: SetupUnion,
    account: AccountState,
    smt_intensity: float,
    has_bos: bool,
    both_sweep: bool,
    minutes_to_news: int,
    atr_ratio: float,
) -> tuple[list[str], int, int]:
    """コンテキストに応じたリスクタグ・risk_score・latencyを生成。"""
    tags: list[str] = []

    if atr_ratio > 1.5:
        tags.append("HIGH_ATR")
    if account.consecutive_losses >= 2:
        tags.append("CONSECUTIVE_LOSSES")
    if account.consecutive_losses >= 3:
        tags.append("OVER_TRADING_WARNING")
    if minutes_to_news <= 30:
        tags.append("NEWS_SOON")
    if smt_intensity < 3.0:
        tags.append("WEAK_SMT")
    if not has_bos:
        tags.append("NO_BOS")
    if not both_sweep:
        tags.append("CORRELATION_FAIL")
    if hasattr(setup, "wick_ratio_pct") and setup.wick_ratio_pct < 25.0:
        tags.append("THIN_WICK")
    if setup.timestamp.hour in (21, 22) and setup.timestamp.weekday() >= 5:
        tags.append("LOW_LIQUIDITY")

    risk_score = sum(RISK_TAG_WEIGHTS.get(t, 10) for t in tags)
    latency = random.randint(LLM_LATENCY_MIN_MS, LLM_LATENCY_MAX_MS)
    return tags, risk_score, latency


def run_llm_risk_audit(
    setup: SetupUnion,
    account: AccountState,
    smt_intensity: float,
    has_bos: bool,
    both_sweep: bool,
    minutes_to_news: int,
    atr_ratio: float,
    candidate_score: float | None = None,
    bayes_probability: float = 0.0,
    htf_trend_direction: str = "NEUTRAL",
    smt_leader: str = "NONE",
    smt_diff: float = 0.0,
    trigger_df: pd.DataFrame | None = None,
) -> tuple[list[str], int, int, str, int, str]:
    """
    L4 リスク監査: USE_LLM_AUDITOR=True なら Gemini 1.5 Flash、さもなければシミュレーション。

    Returns: tags, risk_score, llm_latency_ms, llm_decision, confidence_score, reason_summary
    """
    bypass_type = _setup_type_for_llm(setup)
    if is_l4_bypass_setup_type(bypass_type):
        tags, risk_score, latency, llm_decision, confidence, reason, _ = _rule_base_l4_bypass_result(
            htf_trend_direction,
            False,
            setup_type=bypass_type,
        )
        return tags, risk_score, latency, llm_decision, confidence, reason

    if USE_LLM_AUDITOR:
        try:
            from llm_auditor import get_auditor

            auditor = get_auditor()
            ctx = build_llm_audit_context(
                setup,
                smt_intensity,
                has_bos,
                both_sweep,
                minutes_to_news,
                atr_ratio,
                float(candidate_score or 0.0),
                bayes_probability,
                account,
                setup_type=_setup_type_for_llm(setup),
                htf_trend_direction=htf_trend_direction,
                smt_leader=smt_leader,
                smt_diff=smt_diff,
                trigger_df=trigger_df,
            )
            result = auditor.audit_trade(ctx)
            confidence = int(result.get("confidence_score", 0))
            return (
                result["reason_codes"],
                int(result.get("risk_score", max(0, 100 - confidence))),
                int(result["llm_latency_ms"]),
                str(result["llm_decision"]),
                confidence,
                str(result.get("reason_summary", result.get("thinking", ""))),
            )
        except Exception:
            pass

    tags, risk_score, latency = simulate_llm_risk_audit(
        setup, account, smt_intensity, has_bos, both_sweep, minutes_to_news, atr_ratio
    )
    confidence = max(0, min(100, 100 - risk_score))
    return (
        tags,
        risk_score,
        latency,
        audit_rm.confidence_to_llm_decision(confidence),
        confidence,
        "",
    )


def resolve_volume_profile_context(
    setup: SetupUnion,
    trigger_df: pd.DataFrame | None,
) -> dict[str, Any] | None:
    """SessionVolumeProfile (VP-VAR) → Gemini L4 / L2 用 volume_profile_context。"""
    if trigger_df is None or trigger_df.empty:
        return None
    try:
        from llm_auditor import build_volume_profile_context_from_levels
        from strategies.archive.cspa import compute_cspa_session_volume_profile, evaluate_cspa_vp_location, resolve_cspa_session_type
        from strategies.market_utils import pip_size_for_pair
        from volume_profile_analyzer import SessionVolumeProfile, normalize_trade_direction

        direction = normalize_trade_direction(str(setup.direction))
        pair = str(setup.pair)
        pip = pip_size_for_pair(pair)
        bar_index = _resolve_track_start_index(trigger_df, setup)

        if isinstance(setup, LsfcSetup):
            session_type = resolve_cspa_session_type(setup.timestamp)
            if session_type == "OFF_HOURS":
                session_type = "LONDON"
            levels = compute_cspa_session_volume_profile(
                trigger_df,
                setup.timestamp,
                pair,
                bar_index,
                session_type=session_type,
            )
            filter_price = float(setup.sweep_extreme)
            score_price = float(setup.entry_price)
            buffer_atr = float(setup.atr) * 0.15 if setup.atr > 0 else None
            profiler = SessionVolumeProfile.for_pair(pair)
            is_allowed, location_score = profiler.evaluate_vp_location(
                direction,
                levels,
                pip_size=pip,
                filter_price=filter_price,
                score_price=score_price,
                buffer_atr=buffer_atr,
            )
            trigger_price = filter_price
        else:
            momentum = getattr(setup, "momentum", None)
            if momentum is None:
                return None
            is_allowed, location_score, levels = evaluate_cspa_vp_location(
                trigger_df,
                momentum,
                pair,
                direction,  # type: ignore[arg-type]
                bar_index=bar_index,
            )
            trigger_price = float(getattr(setup, "trigger_price", momentum.entry_price))

        ctx = build_volume_profile_context_from_levels(
            levels=levels,
            direction=direction,
            trigger_price=trigger_price,
            is_allowed=is_allowed,
            location_score=location_score,
            pip_size=pip,
        )
        return ctx or None
    except Exception:
        return None


def build_llm_audit_context(
    setup: SetupUnion,
    smt_intensity: float,
    has_bos: bool,
    both_sweep: bool,
    minutes_to_news: int,
    atr_ratio: float,
    candidate_score: float,
    bayes_probability: float,
    account: AccountState,
    setup_type: str | None = None,
    htf_trend_direction: str = "NEUTRAL",
    smt_leader: str = "NONE",
    smt_diff: float = 0.0,
    trigger_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """L3.5 通過後に Gemini へ渡す監査コンテキスト（候補 CSV / バッチ監査用）。"""
    session_type: str | None = None
    adr_remaining: float | None = None
    wf = getattr(setup, "wyckoff_features", None)
    lf = getattr(setup, "lgr_features", None)
    if wf is not None:
        session_type = str(getattr(wf, "session_type", "") or "") or None
        adr_raw = getattr(wf, "adr_remaining", None)
        if adr_raw is not None:
            adr_remaining = float(adr_raw)
    elif lf is not None:
        session_type = str(getattr(lf, "session_type", "") or "") or None
        adr_raw = getattr(lf, "adr_remaining", None)
        if adr_raw is not None:
            adr_remaining = float(adr_raw)
    else:
        momentum = getattr(setup, "momentum", None)
        if momentum is not None:
            session_type = str(getattr(momentum, "session_type", "") or "") or None
            adr_raw = getattr(momentum, "adr_remaining", None)
            if adr_raw is not None:
                adr_remaining = float(adr_raw)

    ctx: dict[str, Any] = {
        "pair": setup.pair,
        "setup_type": setup_type or _setup_type_for_llm(setup),
        "direction": setup.direction,
        "timestamp": setup.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "smt_intensity": smt_intensity,
        "smt_leader": smt_leader,
        "smt_diff": smt_diff,
        "minutes_to_next_news": minutes_to_news,
        "recent_losses": account.consecutive_losses,
        "has_bos": has_bos,
        "both_sweep": both_sweep,
        "atr_ratio": atr_ratio,
        "wick_ratio_pct": getattr(setup, "wick_ratio_pct", 0.0),
        "candidate_score": candidate_score,
        "bayes_probability": bayes_probability,
        "htf_trend_direction": htf_trend_direction,
        "entry_price": float(getattr(setup, "entry_price", 0.0) or 0.0),
        "trigger_price": float(
            getattr(setup, "trigger_price", getattr(setup, "entry_price", 0.0)) or 0.0
        ),
    }
    if session_type is not None:
        ctx["session_type"] = session_type
    if adr_remaining is not None:
        ctx["adr_remaining"] = adr_remaining
    vp_ctx = resolve_volume_profile_context(setup, trigger_df)
    if vp_ctx is not None:
        ctx["volume_profile_context"] = vp_ctx
    if setup_type in WYCKOFF_SETUP_TYPES or _setup_type_for_llm(setup) in WYCKOFF_SETUP_TYPES:
        if wf is not None:
            ctx["wyckoff_features"] = wf.as_dict()
    if setup_type in LGR_SETUP_TYPES or _setup_type_for_llm(setup) in LGR_SETUP_TYPES:
        if lf is not None:
            ctx["lgr_features"] = lf.as_dict()
    return ctx


def build_llm_audit_context_from_pending(
    pending: PendingEvaluation,
    *,
    trigger_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    setup = pending.setup
    return build_llm_audit_context(
        setup,
        pending.smt,
        pending.has_bos,
        pending.both_sweep,
        pending.minutes_to_news,
        pending.atr_ratio,
        pending.candidate_score,
        pending.bayes_probability,
        AccountState(consecutive_losses=pending.consecutive_losses_snapshot),
        setup_type=pending.setup_type,
        htf_trend_direction=pending.htf_trend_direction,
        smt_leader=pending.smt_leader,
        smt_diff=pending.smt_diff,
        trigger_df=trigger_df,
    )


def risk_to_decision(risk_score: int) -> str:
    if risk_score >= 41:
        return "REJECT_BY_L4"
    if risk_score >= 21:
        return "CAUTION"
    return "ALLOW"


# Risk sizing -> audit/risk_manager.py


def calc_position_size(
    equity: float,
    candidate_score: float,
    monthly_dd_remaining: float,
    consecutive_losses: int,
    llm_decision: str,
    sl_distance: float,
    bayes_probability: float = 0.0,
    account: AccountState | None = None,
    *,
    skip_portfolio_multiplier: bool = False,
    skip_defense_sizing: bool = False,
) -> tuple[float, float, float]:
    acct = account or AccountState(profile=PROP_FIRM_PROFILE)
    if skip_defense_sizing:
        base_risk = audit_rm.pure_bt_flat_base_risk_pct(acct.profile)
    else:
        base_risk = acct.resolved_base_risk_pct()
    return audit_rm.calc_position_size(
        equity,
        candidate_score,
        monthly_dd_remaining,
        consecutive_losses,
        llm_decision,
        sl_distance,
        base_risk,
        bayes_probability,
        BAYES_ALLOW_THRES,
        BAYES_REJECT_THRES,
        acct.profile,
        skip_portfolio_multiplier=skip_portfolio_multiplier,
        skip_defense_sizing=skip_defense_sizing,
    )


def apply_daily_dd_brake(
    lot_factor: float,
    equity: float,
    sl_distance: float,
    daily_loss_pct: float = 0.0,
    account: AccountState | None = None,
) -> tuple[float, float, float]:
    acct = account or AccountState(profile=PROP_FIRM_PROFILE)
    base_risk = acct.resolved_base_risk_pct()
    return audit_rm.apply_daily_dd_brake(
        lot_factor, equity, sl_distance, base_risk, daily_loss_pct
    )


def compute_l45_multipliers(
    candidate_score: float,
    monthly_dd_remaining: float,
    consecutive_losses: int,
    llm_decision: str,
    bayes_probability: float,
    profile: str | None = None,
) -> dict[str, float]:
    return audit_rm.compute_l45_multipliers(
        candidate_score,
        monthly_dd_remaining,
        consecutive_losses,
        llm_decision,
        bayes_probability,
        BAYES_ALLOW_THRES,
        BAYES_REJECT_THRES,
        profile or PROP_FIRM_PROFILE,
    )


def build_strategy_registry(*, live_only: bool = False) -> list[BaseStrategy]:
    return get_registered_strategies(RISK_TAG_WEIGHTS, MODE_H1, live_only=live_only)


# =============================================================================
# L5 バックテスト＆シャドー追跡
# =============================================================================
def _compute_session_timeout_deadline(
    entry_timestamp: pd.Timestamp,
    timeout_server_hour: int,
) -> pd.Timestamp:
    """
    エントリー日 00:00 + timeout_server_hour 時間 → 絶対決済締切。

    例: entry=2023-02-01 21:00, timeout_server_hour=27 → 2023-02-02 03:00
    """
    return pd.Timestamp(entry_timestamp).normalize() + pd.Timedelta(hours=timeout_server_hour)


def _find_first_bar_at_or_after(
    df: pd.DataFrame,
    start_index: int,
    deadline: pd.Timestamp,
) -> int | None:
    from strategies.bt_ohlcv import as_ohlcv, find_first_bar_at_or_after_np, normalize_ts_ns

    arr = as_ohlcv(df)
    return find_first_bar_at_or_after_np(arr, start_index, normalize_ts_ns(deadline))


def _force_close_pnl(
    direction: str,
    entry: float,
    exit_price: float,
    risk: float,
) -> tuple[str, float, float]:
    if direction == "BUY":
        pnl_price = exit_price - entry
    else:
        pnl_price = entry - exit_price
    profit_r = max(-1.0, min(2.4, pnl_price / risk))
    profit_pips = pnl_price / PIP_SIZE
    result = "WIN" if profit_r > 0 else "LOSS"
    return result, profit_r, profit_pips


@dataclass(frozen=True)
class TradeTrackOutcome:
    """L5 trade tracking result with exit bar metadata and sync validation flags."""

    result: str
    profit_r: float
    profit_pips: float
    holding_minutes: int
    start_index: int
    exit_index: int
    exit_reason: str
    invalid_sync: bool = False
    sync_flags: tuple[str, ...] = ()

    def as_legacy_tuple(self) -> tuple[str, float, float, int]:
        return self.result, self.profit_r, self.profit_pips, self.holding_minutes


def _validate_l5_sync(
    outcome: TradeTrackOutcome,
    pair_df: pd.DataFrame,
    entry_timestamp: pd.Timestamp,
) -> tuple[bool, tuple[str, ...]]:
    """Flag INVALID_SYNC when exit time equals entry or dataset tail without TP/SL."""
    from strategies.bt_ohlcv import as_ohlcv, ts_ns_to_pd

    flags = list(outcome.sync_flags)
    entry_ts = normalize_bar_timestamp(entry_timestamp)
    exit_ts = normalize_bar_timestamp(ts_ns_to_pd(int(as_ohlcv(pair_df).datetime_ns[outcome.exit_index])))
    dataset_end_ts = normalize_bar_timestamp(ts_ns_to_pd(int(as_ohlcv(pair_df).datetime_ns[-1])))

    assert exit_ts >= entry_ts, (
        f"L5 sync violation: exit_timestamp {exit_ts} < entry_timestamp {entry_ts}"
    )

    if exit_ts == entry_ts and outcome.exit_reason not in ("TP_HIT", "SL_HIT", "SL_AND_TP_SAME_BAR"):
        if "INVALID_SYNC" not in flags:
            flags.append("INVALID_SYNC")

    if exit_ts == dataset_end_ts and outcome.exit_reason in (
        "DATASET_END",
        "SAME_BAR_NO_EXIT",
        "SAME_BAR_FORCE_CLOSE",
    ):
        if "INVALID_SYNC" not in flags:
            flags.append("INVALID_SYNC")

    invalid = outcome.invalid_sync or bool(flags)
    if invalid:
        assert outcome.result != "WIN" or outcome.exit_reason == "TP_HIT", (
            "INVALID_SYNC must not produce fabricated WIN without explicit TP_HIT"
        )
    return invalid, tuple(dict.fromkeys(flags))


def track_trade_outcome(
    df: pd.DataFrame,
    start_index: int,
    direction: str,
    entry: float,
    stop_loss: float,
    take_profit: float,
    bar_minutes: int,
    *,
    force_close_at_timeout: bool = False,
    timeout_server_hour: int = 0,
    entry_timestamp: pd.Timestamp | None = None,
    max_holding_bars: int | None = None,
) -> TradeTrackOutcome:
    """
    L5 シャドー追跡 — 次バー以降で SL/TP を明示検索。データ末尾の捏造 WIN は返さない。
    """
    from strategies.bt_l5 import track_trade_outcome_np
    from strategies.bt_ohlcv import as_ohlcv, normalize_ts_ns

    holding_limit = max_holding_bars if max_holding_bars is not None else MAX_HOLDING_BARS
    ohlcv = as_ohlcv(df)
    entry_ns = normalize_ts_ns(entry_timestamp) if entry_timestamp is not None else None
    outcome = track_trade_outcome_np(
        ohlcv,
        start_index,
        direction,
        entry,
        stop_loss,
        take_profit,
        bar_minutes,
        max_holding_bars=holding_limit,
        force_close_at_timeout=force_close_at_timeout,
        timeout_server_hour=timeout_server_hour,
        entry_timestamp_ns=entry_ns,
    )
    return TradeTrackOutcome(
        result=outcome.result,
        profit_r=outcome.profit_r,
        profit_pips=outcome.profit_pips,
        holding_minutes=outcome.holding_minutes,
        start_index=outcome.start_index,
        exit_index=outcome.exit_index,
        exit_reason=outcome.exit_reason,
        invalid_sync=outcome.invalid_sync,
        sync_flags=outcome.sync_flags,
    )


def _resolve_track_start_index(
    pair_df: pd.DataFrame,
    setup: SetupUnion,
) -> int:
    """追跡 DataFrame 上のエントリーバー位置を entry timestamp で同期（bar_index 流用禁止）。"""
    from strategies.bt_ohlcv import as_ohlcv, normalize_ts_ns, resolve_bar_position_np, resolve_track_start_index_np

    ts = normalize_bar_timestamp(setup.timestamp)
    arr = as_ohlcv(pair_df)
    pos = resolve_bar_position_np(arr, normalize_ts_ns(ts))
    if pos is not None:
        return pos
    if isinstance(setup, CspaSetup):
        bi = int(setup.bar_index)
        if 0 <= bi < arr.length:
            if normalize_ts_ns(arr.datetime_ns[bi]) == normalize_ts_ns(ts):
                return bi
    from strategies.archive.wyckoff_reversal import ReversalSetup

    if isinstance(setup, ReversalSetup):
        bi = int(setup.recovery_bar_index)
        if 0 <= bi < arr.length:
            return bi
    return resolve_track_start_index_np(arr, normalize_ts_ns(ts))


def _build_l0_exposure_reject_pending(
    setup: SetupUnion,
    gbp_s: SetupUnion | None,
    eur_s: SetupUnion | None,
    account: AccountState,
    equity_snapshot: float,
    daily_rem: float,
    monthly_rem: float,
    streak_snapshot: int,
    trade_id: str,
    gbp_df: pd.DataFrame,
    eur_df: pd.DataFrame,
) -> PendingEvaluation:
    """L0 当日エクスポージャー上限超過 — L1〜L4 をスキップした REJECT Pending。"""
    pair_df = gbp_df if uses_primary_dataframe(setup.pair) else eur_df
    start_idx = _resolve_track_start_index(pair_df, setup)
    return PendingEvaluation(
        trade_id=trade_id,
        setup_type=_setup_type_for_llm(setup),
        setup=setup,
        gbp_s=gbp_s,
        eur_s=eur_s,
        equity_before=equity_snapshot,
        daily_rem=daily_rem,
        monthly_rem=monthly_rem,
        smt=0.0,
        smt_diff=0.0,
        smt_leader="NONE",
        has_bos=False,
        candidate_score=0.0,
        atr_ratio=0.0,
        both_sweep=False,
        tags=["DAILY_EXPOSURE_LIMIT_EXCEEDED"],
        risk_score=0,
        latency=0,
        decision_source="REJECT_BY_L0",
        is_reject=True,
        bayes_probability=0.0,
        consecutive_losses_snapshot=streak_snapshot,
        profile=account.profile,
        llm_eligible=False,
        risk_budget=0.0,
        lot_size=0.0,
        lot_factor=0.0,
        trade_risk_pct=0.0,
        minutes_to_news=0,
        start_idx=start_idx,
    )


def _build_daily_stop_reject_pending(
    setup: SetupUnion,
    strategy: BaseStrategy,
    gbp_s: SetupUnion | None,
    eur_s: SetupUnion | None,
    account: AccountState,
    equity_snapshot: float,
    daily_rem: float,
    monthly_rem: float,
    streak_snapshot: int,
    trade_id: str,
    gbp_df: pd.DataFrame,
    eur_df: pd.DataFrame,
) -> PendingEvaluation:
    """同日2連敗後 — その日の残りエントリーを強制停止（DAILY_STOP）。"""
    pair_df = gbp_df if uses_primary_dataframe(setup.pair) else eur_df
    start_idx = _resolve_track_start_index(pair_df, setup)
    return PendingEvaluation(
        trade_id=trade_id,
        setup_type=strategy.setup_type,
        setup=setup,
        gbp_s=gbp_s,
        eur_s=eur_s,
        equity_before=equity_snapshot,
        daily_rem=daily_rem,
        monthly_rem=monthly_rem,
        smt=0.0,
        smt_diff=0.0,
        smt_leader="NONE",
        has_bos=False,
        candidate_score=0.0,
        atr_ratio=0.0,
        both_sweep=False,
        tags=[audit_dd_throttle.REASON_DAILY_STOP],
        risk_score=0,
        latency=0,
        decision_source="REJECT_BY_DAILY_STOP",
        is_reject=True,
        bayes_probability=0.0,
        consecutive_losses_snapshot=streak_snapshot,
        profile=account.profile,
        llm_eligible=False,
        risk_budget=0.0,
        lot_size=0.0,
        lot_factor=0.0,
        trade_risk_pct=0.0,
        minutes_to_news=0,
        start_idx=start_idx,
    )


def _build_lgr_prop_filter_reject_pending(
    setup: SetupUnion,
    strategy: BaseStrategy,
    gbp_s: SetupUnion | None,
    eur_s: SetupUnion | None,
    account: AccountState,
    equity_snapshot: float,
    daily_rem: float,
    monthly_rem: float,
    streak_snapshot: int,
    trade_id: str,
    gbp_df: pd.DataFrame,
    eur_df: pd.DataFrame,
    *,
    decision_source: str,
    tags: tuple[str, ...],
) -> PendingEvaluation:
    pair_df = gbp_df if uses_primary_dataframe(setup.pair) else eur_df
    start_idx = _resolve_track_start_index(pair_df, setup)
    return PendingEvaluation(
        trade_id=trade_id,
        setup_type=strategy.setup_type,
        setup=setup,
        gbp_s=gbp_s,
        eur_s=eur_s,
        equity_before=equity_snapshot,
        daily_rem=daily_rem,
        monthly_rem=monthly_rem,
        smt=0.0,
        smt_diff=0.0,
        smt_leader="NONE",
        has_bos=False,
        candidate_score=0.0,
        atr_ratio=0.0,
        both_sweep=False,
        tags=list(tags),
        risk_score=0,
        latency=0,
        decision_source=decision_source,
        is_reject=True,
        bayes_probability=0.0,
        consecutive_losses_snapshot=streak_snapshot,
        profile=account.profile,
        llm_eligible=False,
        risk_budget=0.0,
        lot_size=0.0,
        lot_factor=0.0,
        trade_risk_pct=0.0,
        minutes_to_news=0,
        start_idx=start_idx,
    )


def _build_mutual_exclusion_reject_pending(
    setup: SetupUnion,
    strategy: BaseStrategy,
    gbp_s: SetupUnion | None,
    eur_s: SetupUnion | None,
    account: AccountState,
    equity_snapshot: float,
    daily_rem: float,
    monthly_rem: float,
    streak_snapshot: int,
    trade_id: str,
    gbp_df: pd.DataFrame,
    eur_df: pd.DataFrame,
    blocking_setup_type: str,
) -> PendingEvaluation:
    """
    L2 層: 同一シンボル1ポジション制限。

    アクティブポジション保有中（entry <= ts < close）は同シンボルへの新規エントリーを遮断。
    """
    pair_df = gbp_df if uses_primary_dataframe(setup.pair) else eur_df
    start_idx = _resolve_track_start_index(pair_df, setup)
    return PendingEvaluation(
        trade_id=trade_id,
        setup_type=strategy.setup_type,
        setup=setup,
        gbp_s=gbp_s,
        eur_s=eur_s,
        equity_before=equity_snapshot,
        daily_rem=daily_rem,
        monthly_rem=monthly_rem,
        smt=0.0,
        smt_diff=0.0,
        smt_leader="NONE",
        has_bos=False,
        candidate_score=0.0,
        atr_ratio=0.0,
        both_sweep=False,
        tags=[
            audit_rm.mutual_exclusion_reason_tag(),
            f"BLOCKED_BY_{blocking_setup_type}",
        ],
        risk_score=0,
        latency=0,
        decision_source=audit_rm.DECISION_MUTUAL_EXCLUSION_LOCK,
        is_reject=True,
        bayes_probability=0.0,
        consecutive_losses_snapshot=streak_snapshot,
        profile=account.profile,
        llm_eligible=False,
        risk_budget=0.0,
        lot_size=0.0,
        lot_factor=0.0,
        trade_risk_pct=0.0,
        minutes_to_news=0,
        start_idx=start_idx,
    )


def _evaluate_setup_at_timestamp(
    strategy: BaseStrategy,
    setup: SetupUnion,
    gbp_s: SetupUnion | None,
    eur_s: SetupUnion | None,
    account: AccountState,
    equity_snapshot: float,
    daily_rem: float,
    monthly_rem: float,
    daily_loss_fraction: float,
    h1_gbp: pd.DataFrame,
    h1_eur: pd.DataFrame,
    gbp_df: pd.DataFrame,
    eur_df: pd.DataFrame,
    bayes_engine: BayesEngine,
    tref_bayes_filter: Any | None = None,
    minutes_to_news_override: int | None = None,
    htf_gbp: pd.DataFrame | None = None,
    htf_eur: pd.DataFrame | None = None,
) -> PendingEvaluation:
    """Phase-1: L0〜L4.5 を同一口座スナップショットで一括判定。

    v3.2: L3.5 ベイズ（ハード拒否）→ L4 LLM → L3.5 降格の順。ベイズ REJECT 時は LLM 未呼び出し。
    v3.4: L0 当日累積エクスポージャー上限 — 超過時は L1〜L4 をスキップして REJECT。
    v3.4: 同一シンボル1ポジション制限 — L5 確定区間と重複 → MUTUAL_EXCLUSION_LOCK。
    """
    trade_id = account.next_trade_id(setup.timestamp)
    streak_snapshot = account.consecutive_losses
    setup_type = strategy.setup_type
    defense_pure = is_defense_pure_setup(setup_type)
    lgr_baseline = setup_type in LGR_SETUP_TYPES and is_lgr_l0_ev_baseline_mode()
    entry_filter_bypass = defense_pure or lgr_baseline

    account.purge_closed_positions(setup.timestamp)

    # --- L2 相互排他 ---
    blocked, blocking_type = account.is_blocked_by_mutual_exclusion(
        setup.timestamp, setup.pair, setup_type
    )
    if blocked and blocking_type is not None and not entry_filter_bypass:
        return _build_mutual_exclusion_reject_pending(
            setup,
            strategy,
            gbp_s,
            eur_s,
            account,
            equity_snapshot,
            daily_rem,
            monthly_rem,
            streak_snapshot,
            trade_id,
            gbp_df,
            eur_df,
            blocking_type,
        )

    planned_risk_pct = account.resolved_base_risk_pct()
    l0_exposure_fail = account.would_exceed_daily_exposure(planned_risk_pct)
    l0_fail = daily_rem <= 1.0
    daily_stop_active = audit_dd_throttle.is_daily_stop_active(account)

    if daily_stop_active and not defense_pure:
        return _build_daily_stop_reject_pending(
            setup,
            strategy,
            gbp_s,
            eur_s,
            account,
            equity_snapshot,
            daily_rem,
            monthly_rem,
            streak_snapshot,
            trade_id,
            gbp_df,
            eur_df,
        )

    if lgr_baseline and isinstance(setup, LgrSetup):
        from archive.lgr.lgr_prop_controls import lgr_max_open_positions, session_open_minutes_reject

        if session_open_minutes_reject(int(setup.lgr_features.minutes_from_session_open)):
            return _build_lgr_prop_filter_reject_pending(
                setup,
                strategy,
                gbp_s,
                eur_s,
                account,
                equity_snapshot,
                daily_rem,
                monthly_rem,
                streak_snapshot,
                trade_id,
                gbp_df,
                eur_df,
                decision_source="REJECT_BY_SESSION_OPEN",
                tags=("LGR_SESSION_OPEN_FILTER",),
            )
        max_pos = lgr_max_open_positions()
        if max_pos is not None and len(account.open_positions) >= max_pos:
            return _build_lgr_prop_filter_reject_pending(
                setup,
                strategy,
                gbp_s,
                eur_s,
                account,
                equity_snapshot,
                daily_rem,
                monthly_rem,
                streak_snapshot,
                trade_id,
                gbp_df,
                eur_df,
                decision_source="REJECT_BY_MAX_POSITIONS",
                tags=("LGR_MAX_POSITIONS",),
            )

    if l0_exposure_fail and not defense_pure:
        return _build_l0_exposure_reject_pending(
            setup,
            gbp_s,
            eur_s,
            account,
            equity_snapshot,
            daily_rem,
            monthly_rem,
            streak_snapshot,
            trade_id,
            gbp_df,
            eur_df,
        )

    minutes_to_news = resolve_minutes_to_news(setup.timestamp, minutes_to_news_override)
    l1_fail = minutes_to_news <= NEWS_REJECT_THRESHOLD_MIN

    prev_htf_gbp = getattr(strategy, "_htf_gbp", None)
    prev_htf_eur = getattr(strategy, "_htf_eur", None)
    if htf_gbp is not None:
        strategy._htf_gbp = htf_gbp  # type: ignore[attr-defined]
    if htf_eur is not None:
        strategy._htf_eur = htf_eur  # type: ignore[attr-defined]
    try:
        strategy_result = strategy.analyze_setup(setup, gbp_s, eur_s, h1_gbp, h1_eur)
    finally:
        if htf_gbp is not None:
            strategy._htf_gbp = prev_htf_gbp  # type: ignore[attr-defined]
        if htf_eur is not None:
            strategy._htf_eur = prev_htf_eur  # type: ignore[attr-defined]

    raw = strategy_result.raw_features
    htf_trend_direction = str(raw.get("htf_trend_direction", "NEUTRAL"))
    htf_counter_trend = bool(raw.get("htf_counter_trend", False))
    htf_lot_multiplier = float(raw.get("htf_lot_multiplier", 1.0) or 1.0)
    fvg_final_lot_factor = 0.0
    htf_reject = (
        not is_l4_bypass_setup_type(setup_type)
        and setup_type != FVG_FILL_SETUP_TYPE
        and raw.get("reject_reason") == "REJECT_BY_HTF_TREND"
    )
    if setup_type in TTM_SETUP_TYPES:
        smt = 0.0
        smt_feats = SMTFeatures(intensity=0.0, diff=0.0, leader="UNK")
        has_bos = False
        atr_ratio = float(raw.get("pre_ttm_atr_ratio", 0.0))
        both_sweep = False
    elif setup_type == DBBS_SETUP_TYPE:
        smt = 0.0
        smt_feats = SMTFeatures(intensity=0.0, diff=0.0, leader="NONE")
        has_bos = False
        atr_ratio = float(raw.get("bb20_width_atr_ratio", 1.0) or 1.0)
        both_sweep = False
    else:
        smt = float(raw["smt_intensity"])
        smt_feats = SMTFeatures(
            intensity=smt,
            diff=float(raw["smt_diff"]),
            leader=str(raw["smt_leader"]),
        )
        has_bos = bool(raw["has_bos"])
        atr_ratio = float(raw["atr_ratio"])
        both_sweep = bool(raw["both_sweep"])

    candidate_score = strategy_result.candidate_score
    l2_min = resolve_l2_min_candidate_score(setup_type)
    l2_fail = candidate_score < l2_min
    if entry_filter_bypass:
        if defense_pure:
            l0_fail = False
        l1_fail = False
        l2_fail = False
        htf_reject = False

    tref_loss_pattern_reject = False
    tref_loss_pattern_reason: str | None = None
    cspa_gate: dict[str, Any] | None = None
    cspa_gate_reason = ""
    cspa_tp_multiplier = 1.0
    lgr_bayes_regime = ""
    lgr_bayes_reason = ""

    if defense_pure and not (
        setup_type in LGR_SETUP_TYPES
        and is_lgr_bayes_gate_enabled()
        and isinstance(setup, LgrSetup)
    ):
        bayes_probability = BAYES_BYPASS_NEUTRAL_PROBABILITY
        bayes_hard_reject = False
    elif setup_type in LGR_SETUP_TYPES and is_lgr_bayes_gate_enabled() and isinstance(setup, LgrSetup):
        lgr_bayes = evaluate_lgr_bayes_gate(features_from_lgr_setup(setup))
        bayes_probability = float(lgr_bayes["bayes_probability"])
        lgr_bayes_regime = str(lgr_bayes["bayes_regime"])
        lgr_bayes_reason = str(lgr_bayes["bayes_reason"])
        bayes_hard_reject = lgr_bayes_regime == "REJECT"
    elif setup_type == TREF_SETUP_TYPE and tref_bayes_filter is not None:
        from audit.tref_bayes_filter import TrefBayesFilter

        assert isinstance(tref_bayes_filter, TrefBayesFilter)
        tref_bayes_filter.register_event(setup.timestamp)
        score_breakdown = raw.get("score_breakdown") if isinstance(raw.get("score_breakdown"), dict) else {}
        bayes_probability = tref_bayes_filter.compute_probability(
            setup.timestamp,
            setup.pair,
            score_breakdown,
        )
        tref_loss_pattern_reject, tref_loss_pattern_reason = tref_bayes_filter.check_loss_pattern_reject(
            setup.timestamp,
            setup.pair,
            score_breakdown,
        )
        bayes_hard_reject = tref_bayes_filter.check_hard_reject(setup.timestamp, bayes_probability)
    elif setup_type == CSPA_SETUP_TYPE and isinstance(setup, CspaSetup):
        from audit.cspa_bayes_gate import evaluate_cspa_bayes_gate

        cspa_gate = evaluate_cspa_bayes_gate(setup.bayes_features.as_dict())
        bayes_probability = float(cspa_gate["bayes_probability"])
        bayes_hard_reject = cspa_gate["decision"] == "REJECT"
        cspa_gate_reason = str(cspa_gate["reason"])
        cspa_tp_multiplier = float(cspa_gate["tp_multiplier"])
    elif setup_type in WYCKOFF_SETUP_TYPES and not is_bayes_bypass_setup_type(setup_type):
        from audit.wyckoff_bayes_gate import check_wyckoff_bayes_hard_reject, is_wyckoff_bayes_strict_mode

        bayes_probability = bayes_engine.compute_probability(
            setup.timestamp,
            smt,
            candidate_score,
            streak_snapshot,
            has_bos,
            both_sweep,
        )
        if is_wyckoff_bayes_strict_mode():
            bayes_hard_reject = check_wyckoff_bayes_hard_reject(bayes_probability)
        else:
            bayes_hard_reject = False
    elif is_bayes_bypass_setup_type(setup_type):
        bayes_probability = BAYES_BYPASS_NEUTRAL_PROBABILITY
        bayes_hard_reject = False
    else:
        bayes_probability = bayes_engine.compute_probability(
            setup.timestamp,
            smt,
            candidate_score,
            streak_snapshot,
            has_bos,
            both_sweep,
        )
        bayes_hard_reject = check_bayes_hard_reject(bayes_probability)

    lgr_bayes_reject = (
        setup_type in LGR_SETUP_TYPES
        and is_lgr_bayes_gate_enabled()
        and lgr_bayes_regime == "REJECT"
    )
    dbbs_bear_kill_reject = (
        setup_type == DBBS_SETUP_TYPE
        and (
            strategy_result.strategy_action == "REJECT"
            or bool(raw.get("bear_kill_switch_active"))
        )
    )

    llm_confidence_score = 0
    llm_reason_summary = ""
    confidence_lot_multiplier = 0.0
    l2_math_sizing = False
    l2_regime = ""
    l2_base_lot_factor = 0.0
    l4_multiplier = 1.0
    vp_zone_label = "NEUTRAL"
    l4_smt_interpretation = ""
    divergence_direction_label = ""
    htf_trend_label = str(htf_trend_direction or "NEUTRAL").upper()

    if l0_fail:
        tags, risk_score, latency, llm_decision, llm_confidence_score, llm_reason_summary = _skipped_llm_audit()
        decision_source = "REJECT_BY_L0"
        llm_eligible = False
    elif l1_fail:
        tags, risk_score, latency, llm_decision, llm_confidence_score, llm_reason_summary = _skipped_llm_audit()
        decision_source = "REJECT_BY_L1"
        llm_eligible = False
    elif htf_reject:
        mismatch_tags = raw.get("reason_codes")
        if isinstance(mismatch_tags, list) and mismatch_tags:
            tags = [str(t) for t in mismatch_tags]
        else:
            tags = ["HTF_TREND_MISMATCH"]
        trend_tag = f"HTF_TREND_{htf_trend_direction}"
        if trend_tag not in tags:
            tags.insert(0, trend_tag)
        risk_score, latency, llm_decision, llm_confidence_score, llm_reason_summary = 0, 0, "REJECT", 0, ""
        decision_source = "REJECT_BY_HTF_TREND"
        llm_eligible = False
    elif lgr_bayes_reject:
        tags = [lgr_bayes_regime, lgr_bayes_reason]
        risk_score, latency, llm_decision, llm_confidence_score, llm_reason_summary = 0, 0, "REJECT", 0, ""
        decision_source = LGR_BAYES_REJECT_SOURCE
        llm_eligible = False
    elif dbbs_bear_kill_reject:
        tags = ["BEAR_KILL_SWITCH_V2"]
        last3 = raw.get("last_3_avg_r")
        if last3 is not None and np.isfinite(float(last3)):
            tags.append(f"LAST3_AVG_R_{float(last3):.2f}")
        risk_score, latency, llm_decision, llm_confidence_score, llm_reason_summary = 0, 0, "REJECT", 0, ""
        decision_source = "REJECT_BY_BEAR_KILL_SWITCH"
        llm_eligible = False
    elif l2_fail:
        tags, risk_score, latency, llm_decision, llm_confidence_score, llm_reason_summary = _skipped_llm_audit()
        tags = l2_reject_reason_tags(setup_type, candidate_score, l2_min)
        decision_source = "REJECT_BY_L2"
        llm_eligible = False
    elif tref_loss_pattern_reject:
        tags, risk_score, latency, llm_decision, llm_confidence_score, llm_reason_summary = _skipped_llm_audit()
        if tref_loss_pattern_reason:
            tags = [tref_loss_pattern_reason]
        decision_source = "REJECT_BY_TREF_LOSS_PATTERN"
        llm_eligible = False
    elif setup_type == CSPA_SETUP_TYPE and bayes_hard_reject:
        tags, risk_score, latency, llm_decision, llm_confidence_score, llm_reason_summary = _skipped_llm_audit()
        if cspa_gate_reason:
            tags = [cspa_gate_reason.split(":")[0]]
        decision_source = "REJECT_BY_BAYES"
        llm_eligible = False
    elif is_l4_bypass_setup_type(setup_type):
        # LSFC: L3.5 ベイズハード拒否より先に L4 直列バイパス（ピラミッド ON でも不変）
        llm_eligible = False
        (
            tags,
            risk_score,
            latency,
            llm_decision,
            llm_confidence_score,
            llm_reason_summary,
            decision_source,
        ) = _rule_base_l4_bypass_result(
            htf_trend_direction,
            bool(raw.get("htf_would_block")),
            setup_type=setup_type,
        )
    elif bayes_hard_reject:
        tags, risk_score, latency, llm_decision, llm_confidence_score, llm_reason_summary = _skipped_llm_audit()
        decision_source = "REJECT_BY_BAYES"
        llm_eligible = False
    elif setup_type == FVG_FILL_SETUP_TYPE:
        llm_eligible = True
        tags, risk_score, latency, llm_decision, llm_confidence_score, llm_reason_summary = run_fvg_llm_risk_audit(
            strategy,
            setup,  # type: ignore[arg-type]
            raw,
        )
        # FVG も他戦略と同じ L4 閾値（>=40 執行、85+ は L4.5 で 1.4x）。
        # FVG_EXECUTE_MIN_CONFIDENCE(85) は高確信度ロット帯の参照値であり、足切りには使わない。
        if llm_confidence_score < audit_rm.LLM_CONFIDENCE_REJECT_BELOW:
            decision_source = "REJECT_BY_LLM"
        else:
            preliminary = resolve_final_decision(l0_fail, l1_fail, l2_fail, llm_decision)
            if preliminary.startswith("REJECT"):
                decision_source = preliminary
            else:
                decision_source = apply_bayes_downgrade(llm_decision, bayes_probability)
    elif setup_type == TREF_SETUP_TYPE:
        llm_eligible = True
        tags, risk_score, latency, llm_decision, llm_confidence_score, llm_reason_summary = run_tref_llm_risk_audit(
            setup,  # type: ignore[arg-type]
            raw,
            minutes_to_news=minutes_to_news,
        )
        if llm_confidence_score < audit_rm.LLM_CONFIDENCE_REJECT_BELOW:
            decision_source = "REJECT_BY_LLM"
        else:
            preliminary = resolve_final_decision(l0_fail, l1_fail, l2_fail, llm_decision)
            if preliminary.startswith("REJECT"):
                decision_source = preliminary
            else:
                decision_source = apply_bayes_downgrade(llm_decision, bayes_probability)
    elif setup_type == DTPA_SETUP_TYPE:
        llm_eligible = True
        tags, risk_score, latency, llm_decision, llm_confidence_score, llm_reason_summary = run_dtpa_llm_risk_audit(
            setup,  # type: ignore[arg-type]
            raw,
            minutes_to_news=minutes_to_news,
        )
        if llm_confidence_score < DTPA_LLM_REJECT_BELOW:
            decision_source = "REJECT_BY_LLM"
        else:
            llm_decision = dtpa_llm_decision(llm_confidence_score)
            preliminary = resolve_final_decision(l0_fail, l1_fail, l2_fail, llm_decision)
            if preliminary.startswith("REJECT"):
                decision_source = preliminary
            else:
                decision_source = apply_bayes_downgrade(llm_decision, bayes_probability)
    elif uses_l2_math_position_sizer(setup_type):
        from llm_auditor import (
            audit_smt_context,
            build_divergence_direction,
            normalize_htf_trend_label,
            normalize_smt_leader_pair,
        )

        l2_math_sizing = True
        trigger_df = gbp_df if str(setup.pair).upper() == "GBPUSD" else eur_df
        vp_zone_raw = resolve_vp_zone_for_l2(setup, trigger_df)
        htf_trend_label = normalize_htf_trend_label(htf_trend_direction)
        leader_pair = normalize_smt_leader_pair(smt_feats.leader)
        divergence_direction_label = build_divergence_direction(leader_pair)
        session_type = resolve_session_type_for_l2(setup, raw)

        l2_regime, l2_base_lot_factor, vp_zone_label, vp_size_tags = calculate_vp_position_size(
            vp_zone_raw
        )

        l4_response = audit_smt_context(
            session_type=session_type,
            trade_direction=str(setup.direction),
            smt_leader=leader_pair,
            divergence_direction=divergence_direction_label,
            htf_trend=htf_trend_label,
            strategy_type=setup_type,
        )
        l4_multiplier = float(l4_response.get("multiplier", 1.0))
        l4_smt_interpretation = str(l4_response.get("smt_interpretation", "NEUTRAL"))
        latency = int(l4_response.get("llm_latency_ms", 0))
        llm_reason_summary = str(l4_response.get("reason", ""))
        llm_decision = l2_regime_to_llm_decision(l2_regime)
        llm_confidence_score = int(round(l4_multiplier * 100))
        tags = [
            f"L2_{l2_regime}",
            f"VP_{vp_zone_label}",
            f"L4_{l4_smt_interpretation}",
            *vp_size_tags,
        ]
        risk_score = max(0, 100 - llm_confidence_score)
        llm_eligible = True
        preliminary = resolve_final_decision(l0_fail, l1_fail, l2_fail, llm_decision)
        if preliminary.startswith("REJECT"):
            decision_source = preliminary
        else:
            decision_source = apply_bayes_downgrade(llm_decision, bayes_probability)
    else:
        llm_eligible = True
        trigger_df = gbp_df if str(setup.pair).upper() == "GBPUSD" else eur_df
        tags, risk_score, latency, llm_decision, llm_confidence_score, llm_reason_summary = run_llm_risk_audit(
            setup, account, smt, has_bos, both_sweep, minutes_to_news, atr_ratio, candidate_score,
            bayes_probability,
            htf_trend_direction=htf_trend_direction,
            smt_leader=smt_feats.leader,
            smt_diff=smt_feats.diff,
            trigger_df=trigger_df,
        )
        # v3.4: confidence < 40 は L0 層で強制 REJECT_BY_LLM（lot_factor=0）
        if llm_confidence_score < audit_rm.LLM_CONFIDENCE_REJECT_BELOW:
            decision_source = "REJECT_BY_LLM"
        else:
            preliminary = resolve_final_decision(l0_fail, l1_fail, l2_fail, llm_decision)
            if preliminary.startswith("REJECT"):
                decision_source = preliminary
            else:
                decision_source = apply_bayes_downgrade(llm_decision, bayes_probability)

    if (
        setup_type == FVG_FILL_SETUP_TYPE
        and htf_counter_trend
        and not decision_source.startswith("REJECT")
    ):
        decision_source = "CAUTION_HTF_COUNTER"
        if "CAUTION_HTF_COUNTER" not in tags:
            tags.append("CAUTION_HTF_COUNTER")

    is_reject = decision_source.startswith("REJECT")

    sl_distance = abs(setup.entry_price - setup.stop_loss)
    # L4.5 六連動: confidence 倍率は後段で適用するため m_llm=1.0（ALLOW）で算出
    if setup_type == DTPA_SETUP_TYPE:
        sizing_decision = "ALLOW" if llm_eligible and not is_reject else "REJECT"
    else:
        sizing_decision = "ALLOW" if llm_eligible and not is_reject else (
            decision_source if decision_source in ("ALLOW", "CAUTION", "CAUTION_HTF_COUNTER") else "REJECT"
        )
    risk_budget, lot_size, lot_factor = calc_position_size(
        equity_snapshot,
        candidate_score,
        monthly_rem,
        streak_snapshot,
        sizing_decision,
        sl_distance,
        bayes_probability,
        account,
        skip_portfolio_multiplier=defense_pure,
        skip_defense_sizing=defense_pure,
    )
    daily_loss_pct = daily_loss_fraction * 100.0
    m_daily = multiplier_daily_dd(daily_loss_pct)
    if m_daily < 1.0 and not defense_pure:
        risk_budget, lot_size, lot_factor = apply_daily_dd_brake(
            lot_factor, equity_snapshot, sl_distance, daily_loss_pct, account
        )
        if "DAILY_DD_BRAKE" not in tags:
            tags.append("DAILY_DD_BRAKE")

    # v3.4 オフェンス型アクセル: L4 confidence_score → lot_multiplier
    if llm_eligible and not is_reject and setup_type == DTPA_SETUP_TYPE:
        base_risk_pct = account.resolved_base_risk_pct()
        dtpa_conf_mult = dtpa_confidence_lot_multiplier(llm_confidence_score)
        lot_factor, risk_budget, lot_size, confidence_lot_multiplier = audit_rm.apply_confidence_lot_scaling_with_mult(
            dtpa_conf_mult,
            lot_factor,
            equity_snapshot,
            sl_distance,
            base_risk_pct,
        )
        if confidence_lot_multiplier <= 0.0:
            decision_source = "REJECT_BY_LLM"
            is_reject = True
            risk_budget = 0.0
            lot_size = 0.0
            lot_factor = 0.0
    elif llm_eligible and not is_reject and l2_math_sizing:
        base_risk_pct = account.resolved_base_risk_pct()
        final_lot_mult = round(l2_base_lot_factor * l4_multiplier, 4)
        confidence_lot_multiplier = final_lot_mult
        lot_factor, risk_budget, lot_size, confidence_lot_multiplier = audit_rm.apply_confidence_lot_scaling_with_mult(
            final_lot_mult,
            lot_factor,
            equity_snapshot,
            sl_distance,
            base_risk_pct,
        )
        if confidence_lot_multiplier <= 0.0:
            decision_source = "REJECT_BY_RISK_SCALE"
            is_reject = True
            risk_budget = 0.0
            lot_size = 0.0
            lot_factor = 0.0
    elif llm_eligible and not is_reject and llm_confidence_score >= audit_rm.LLM_CONFIDENCE_REJECT_BELOW:
        base_risk_pct = account.resolved_base_risk_pct()
        lot_factor, risk_budget, lot_size, confidence_lot_multiplier = audit_rm.apply_confidence_lot_scaling(
            llm_confidence_score,
            lot_factor,
            equity_snapshot,
            sl_distance,
            base_risk_pct,
        )
        if confidence_lot_multiplier <= 0.0:
            decision_source = "REJECT_BY_LLM"
            is_reject = True
            risk_budget = 0.0
            lot_size = 0.0
            lot_factor = 0.0

    base_risk_pct = account.resolved_base_risk_pct()
    if lot_factor > 0.0 and not is_reject and not defense_pure:
        lot_factor, risk_budget, lot_size, cushion_mult = audit_rm.apply_profit_cushion_brake(
            lot_factor,
            equity_snapshot,
            sl_distance,
            base_risk_pct,
            account.phase_start_equity,
            account.profile,
        )
        if cushion_mult < 1.0 and "PROFIT_CUSHION_BRAKE" not in tags:
            tags.append(audit_rm.REASON_PROFIT_CUSHION_BRAKE)

    if lot_factor > 0.0 and not is_reject and audit_twin_brake.is_twin_brake_enabled() and not defense_pure:
        lot_factor, risk_budget, lot_size, twin_bd = audit_twin_brake.apply_twin_brake_to_lot_factor(
            lot_factor,
            equity_snapshot,
            sl_distance,
            base_risk_pct,
            initial_balance=account.phase_start_equity,
            daily_dd_remaining_percent=daily_rem,
        )
        for tag in audit_twin_brake.twin_brake_reason_tags(twin_bd):
            if tag not in tags:
                tags.append(tag)

    if not defense_pure:
        lot_factor = audit_rm.apply_portfolio_lot_multiplier(lot_factor)
    if lot_factor > 0.0:
        lot_factor = audit_rm.apply_lot_factor_floor(lot_factor)
        risk_budget = round(equity_snapshot * base_risk_pct * lot_factor, 2)
        lot_size = audit_rm.lot_from_risk_budget(
            risk_budget, sl_distance, lot_factor
        )
    elif not is_reject:
        risk_budget = 0.0
        lot_size = 0.0

    if lot_factor > 0.0 and not is_reject and audit_dd_throttle.is_dd_throttling_enabled() and not defense_pure:
        dd_pct = account.current_drawdown_pct()
        lot_factor, risk_budget, lot_size, _throttle_mult, dd_tag = (
            audit_dd_throttle.apply_dynamic_dd_throttling(
                lot_factor,
                equity_snapshot,
                sl_distance,
                base_risk_pct,
                dd_pct,
                consecutive_losses=streak_snapshot,
            )
        )
        if dd_tag and dd_tag not in tags:
            tags.append(dd_tag)

    if lot_factor > 0.0 and not is_reject and account.recovery_boost_armed and not defense_pure:
        lot_factor, risk_budget, lot_size, _boost_mult, boost_tag = (
            audit_dd_throttle.apply_recovery_boost(
                lot_factor,
                equity_snapshot,
                sl_distance,
                base_risk_pct,
                recovery_boost_armed=True,
            )
        )
        account.recovery_boost_armed = False
        if boost_tag and boost_tag not in tags:
            tags.append(boost_tag)

    if (
        setup_type == FVG_FILL_SETUP_TYPE
        and lot_factor > 0.0
        and not is_reject
        and htf_lot_multiplier < 1.0
        and not defense_pure
    ):
        lot_factor = round(lot_factor * htf_lot_multiplier, 4)
        lot_factor = audit_rm.apply_lot_factor_floor(lot_factor)
        risk_budget = round(equity_snapshot * base_risk_pct * lot_factor, 2)
        lot_size = audit_rm.lot_from_risk_budget(
            risk_budget, sl_distance, lot_factor
        )
        fvg_final_lot_factor = lot_factor
    elif setup_type == FVG_FILL_SETUP_TYPE and lot_factor > 0.0 and not is_reject:
        fvg_final_lot_factor = lot_factor

    if (
        setup_type == CSPA_SETUP_TYPE
        and cspa_gate is not None
        and cspa_gate.get("decision") == "ALLOW"
        and lot_factor > 0.0
        and not is_reject
        and not defense_pure
    ):
        cspa_lot_mult = float(cspa_gate.get("lot_multiplier", 1.0))
        if cspa_lot_mult != 1.0:
            lot_factor = round(lot_factor * cspa_lot_mult, 4)
            lot_factor = audit_rm.apply_lot_factor_floor(lot_factor)
            base_risk_pct = account.resolved_base_risk_pct()
            risk_budget = round(equity_snapshot * base_risk_pct * lot_factor, 2)
            lot_size = audit_rm.lot_from_risk_budget(
                risk_budget, sl_distance, lot_factor
            )
        gate_tag = cspa_gate_reason.split(":")[0] if cspa_gate_reason else "CSPA_BAYES"
        if gate_tag and gate_tag not in tags:
            tags.append(gate_tag)

    lgr_ev_score = 0.0
    lgr_ev_rank = 0.0
    lgr_ev_lot_multiplier = 1.0
    if (
        setup_type in LGR_SETUP_TYPES
        and is_lgr_ev_sizing_enabled()
        and not is_reject
        and lot_factor > 0.0
        and isinstance(setup, LgrSetup)
    ):
        ev_eval = evaluate_lgr_ev_sizing_for_setup(setup)
        lgr_ev_score = float(ev_eval["ev_score"])
        lgr_ev_rank = float(ev_eval["ev_rank"])
        lgr_ev_lot_multiplier = float(ev_eval["lot_multiplier"])
        lot_factor = round(lot_factor * lgr_ev_lot_multiplier, 4)
        lot_factor = audit_rm.apply_lot_factor_floor(lot_factor)
        base_risk_pct = account.resolved_base_risk_pct()
        risk_budget = round(equity_snapshot * base_risk_pct * lot_factor, 2)
        lot_size = audit_rm.lot_from_risk_budget(
            risk_budget, sl_distance, lot_factor
        )
        if "LGR_EV_SIZING" not in tags:
            tags.append("LGR_EV_SIZING")
        if lgr_ev_rank >= 0.95 and "LGR_EV_TOP5" not in tags:
            tags.append("LGR_EV_TOP5")
        elif lgr_ev_rank >= 0.80 and "LGR_EV_TOP20" not in tags:
            tags.append("LGR_EV_TOP20")
        elif lgr_ev_rank >= 0.50 and "LGR_EV_TOP50" not in tags:
            tags.append("LGR_EV_TOP50")
        elif "LGR_EV_BOTTOM50" not in tags:
            tags.append("LGR_EV_BOTTOM50")

    ttm_bayes_win_prob = 0.0
    ttm_ev_rank = 0.0
    ttm_ev_lot_multiplier = 1.0
    if (
        setup_type in TTM_SETUP_TYPES
        and is_ttm_ev_sizing_mode()
        and not is_reject
        and lot_factor > 0.0
        and isinstance(setup, TtmSetup)
    ):
        train_end = os.getenv("TTM_EV_TRAIN_END")
        if train_end:
            ev_eval = evaluate_ttm_ev_sizing_for_setup(setup)
        else:
            ev_eval = evaluate_ttm_ev_with_runtime(setup)
        ttm_bayes_win_prob = float(ev_eval["bayes_win_prob"])
        ttm_ev_rank = float(ev_eval["ev_rank"])
        ttm_ev_lot_multiplier = float(ev_eval["ev_lot_multiplier"])
        bayes_probability = ttm_bayes_win_prob
        lot_factor = round(lot_factor * ttm_ev_lot_multiplier, 4)
        lot_factor = audit_rm.apply_lot_factor_floor(lot_factor)
        base_risk_pct = account.resolved_base_risk_pct()
        risk_budget = round(equity_snapshot * base_risk_pct * lot_factor, 2)
        lot_size = audit_rm.lot_from_risk_budget(
            risk_budget, sl_distance, lot_factor
        )
        if "TTM_EV_SIZING" not in tags:
            tags.append("TTM_EV_SIZING")
        if ttm_ev_rank >= 0.95 and "TTM_EV_TOP5" not in tags:
            tags.append("TTM_EV_TOP5")
        elif ttm_ev_rank >= 0.80 and "TTM_EV_TOP20" not in tags:
            tags.append("TTM_EV_TOP20")
        elif ttm_ev_rank >= 0.50 and "TTM_EV_TOP50" not in tags:
            tags.append("TTM_EV_TOP50")
        elif "TTM_EV_BOTTOM25" not in tags:
            tags.append("TTM_EV_BOTTOM25")
        from dataclasses import replace

        setup.ttm_features = replace(
            setup.ttm_features,
            bayes_win_prob=ttm_bayes_win_prob,
            ev_rank=ttm_ev_rank,
            ev_lot_multiplier=ttm_ev_lot_multiplier,
        )
        if should_reject_ttm_bottom20(ttm_ev_rank):
            decision_source = "REJECT_BY_TTM_EV_BOTTOM20"
            is_reject = True
            lot_factor = 0.0
            lot_size = 0.0
            risk_budget = 0.0
            trade_risk_pct = 0.0
            if "TTM_EV_BOTTOM20_REJECT" not in tags:
                tags.append("TTM_EV_BOTTOM20_REJECT")

    dn_ev_rank_v2 = 0.0
    dn_prop_gate_tier = ""
    dn_prop_gate_lot_multiplier = 1.0
    dn_ev_rank = 0.0
    dn_ev_bucket = ""
    if (
        setup_type == DINAPOLI_SETUP_TYPE
        and prop_gate_enabled()
        and not is_reject
        and lot_factor > 0.0
        and isinstance(setup, DiNapoliSetup)
    ):
        m15_df = gbp_df if uses_primary_dataframe(setup.pair) else eur_df
        h1_df = h1_gbp if uses_primary_dataframe(setup.pair) else h1_eur
        h4_df = None
        if htf_gbp is not None and htf_eur is not None:
            h4_df = htf_gbp if uses_primary_dataframe(setup.pair) else htf_eur
        gate_row = score_dn_prop_gate_from_setup(
            setup=setup,
            trade_id=trade_id,
            decision_source=decision_source,
            llm_confidence=llm_confidence_score,
            llm_reason=llm_reason_summary,
            minutes_to_news=minutes_to_news,
            m15_df=m15_df,
            h1_df=h1_df,
            h4_df=h4_df,
        )
        dn_ev_rank_v2 = float(gate_row.get("ev_rank_v2", 0.0))
        dn_prop_gate_tier = str(gate_row.get("dn_prop_gate_tier", ""))
        dn_prop_gate_lot_multiplier = float(gate_row.get("dn_prop_gate_lot_multiplier", 1.0))
        dn_ev_rank = dn_ev_rank_v2
        dn_ev_bucket = dn_prop_gate_tier
        lot_factor = round(lot_factor * dn_prop_gate_lot_multiplier, 4)
        lot_factor = audit_rm.apply_lot_factor_floor(lot_factor)
        base_risk_pct = dn_prop_gate_base_risk_frac()
        risk_budget = round(equity_snapshot * base_risk_pct * lot_factor, 2)
        lot_size = audit_rm.lot_from_risk_budget(
            risk_budget, sl_distance, lot_factor
        )
        if "DN_PROP_GATE" not in tags:
            tags.append("DN_PROP_GATE")
        if dn_prop_gate_tier == "Top5" and "DN_PROP_TOP5" not in tags:
            tags.append("DN_PROP_TOP5")
        elif dn_prop_gate_tier == "Top10" and "DN_PROP_TOP10" not in tags:
            tags.append("DN_PROP_TOP10")
        elif dn_prop_gate_tier == "Top20" and "DN_PROP_TOP20" not in tags:
            tags.append("DN_PROP_TOP20")
        elif dn_prop_gate_tier == "Middle" and "DN_PROP_MIDDLE" not in tags:
            tags.append("DN_PROP_MIDDLE")
        elif dn_prop_gate_tier == "Low" and "DN_PROP_LOW" not in tags:
            tags.append("DN_PROP_LOW")

    if (
        setup_type == DBBS_SETUP_TYPE
        and not is_reject
        and lot_factor > 0.0
        and isinstance(setup, DbbsSetup)
    ):
        edge_mult = float(raw.get("edge_risk_mult", 1.0) or 1.0)
        if edge_mult <= 0.0:
            decision_source = "REJECT_BY_BEAR_KILL_SWITCH"
            is_reject = True
            lot_factor = 0.0
            lot_size = 0.0
            risk_budget = 0.0
            trade_risk_pct = 0.0
            if "BEAR_KILL_SWITCH_V2" not in tags:
                tags.append("BEAR_KILL_SWITCH_V2")
        elif edge_mult < 1.0:
            lot_factor = round(lot_factor * edge_mult, 4)
            lot_factor = audit_rm.apply_lot_factor_floor(lot_factor)
            base_risk_pct = account.resolved_base_risk_pct()
            risk_budget = round(equity_snapshot * base_risk_pct * lot_factor, 2)
            lot_size = audit_rm.lot_from_risk_budget(
                risk_budget, sl_distance, lot_factor
            )
            if "DBBS_EDGE_RISK" not in tags:
                tags.append("DBBS_EDGE_RISK")

    trade_risk_pct = compute_trade_risk_pct(base_risk_pct, lot_factor)
    if not is_reject and trade_risk_pct > 0.0 and not defense_pure:
        if account.would_exceed_daily_exposure(trade_risk_pct):
            capped_lf, was_capped = audit_rm.cap_lot_factor_to_daily_exposure(
                lot_factor,
                base_risk_pct,
                account.daily_committed_risk_pct,
            )
            if was_capped and capped_lf > 0.0:
                lot_factor = audit_rm.apply_lot_factor_floor(capped_lf)
                trade_risk_pct = compute_trade_risk_pct(base_risk_pct, lot_factor)
                risk_budget = round(equity_snapshot * base_risk_pct * lot_factor, 2)
                lot_size = audit_rm.lot_from_risk_budget(
                    risk_budget, sl_distance, lot_factor
                )
                if "DAILY_EXPOSURE_CAPPED" not in tags:
                    tags.append("DAILY_EXPOSURE_CAPPED")
            else:
                tags = ["DAILY_EXPOSURE_LIMIT_EXCEEDED"]
                decision_source = "REJECT_BY_L0"
                is_reject = True
                trade_risk_pct = 0.0
                risk_budget = 0.0
                lot_size = 0.0
                lot_factor = 0.0

    if not is_reject and lot_size > 0.0 and lot_factor > 0.0:
        account.commit_daily_risk(trade_risk_pct)
        if lgr_baseline:
            from archive.lgr.lgr_prop_controls import lgr_max_open_positions
            from strategies.archive.liquidity_grab_reversal import LGR_EXEC_BAR_MINUTES, MAX_HOLDING_BARS

            if lgr_max_open_positions() is not None:
                account.register_open_position(
                    setup.timestamp,
                    setup.pair,
                    setup_type,
                    MAX_HOLDING_BARS * LGR_EXEC_BAR_MINUTES,
                )
        # L5 確定後に register_executed_position で区間登録

    pair_df = gbp_df if uses_primary_dataframe(setup.pair) else eur_df
    start_idx = _resolve_track_start_index(pair_df, setup)

    force_close_at_timeout = bool(raw.get("force_close_at_timeout", False))
    timeout_server_hour = int(raw.get("timeout_server_hour", 0) or 0)

    tags = merge_rule_base_l4_bypass_tags(
        tags,
        setup_type,
        decision_source,
        htf_trend_direction=htf_trend_direction,
        htf_would_block=bool(raw.get("htf_would_block")),
    )

    return PendingEvaluation(
        trade_id=trade_id,
        setup_type=strategy_result.setup_type,
        setup=setup,
        gbp_s=gbp_s,
        eur_s=eur_s,
        equity_before=equity_snapshot,
        daily_rem=daily_rem,
        monthly_rem=monthly_rem,
        smt=smt,
        smt_diff=smt_feats.diff,
        smt_leader=smt_feats.leader,
        has_bos=has_bos,
        candidate_score=candidate_score,
        atr_ratio=atr_ratio,
        both_sweep=both_sweep,
        tags=tags,
        risk_score=risk_score,
        latency=latency,
        decision_source=decision_source,
        is_reject=is_reject,
        bayes_probability=bayes_probability,
        consecutive_losses_snapshot=streak_snapshot,
        profile=account.profile,
        llm_eligible=llm_eligible,
        risk_budget=risk_budget,
        lot_size=lot_size,
        lot_factor=lot_factor,
        trade_risk_pct=trade_risk_pct,
        minutes_to_news=minutes_to_news,
        start_idx=start_idx,
        llm_confidence_score=llm_confidence_score,
        llm_reason_summary=llm_reason_summary,
        confidence_lot_multiplier=confidence_lot_multiplier,
        final_lot_size=lot_size if not is_reject else 0.0,
        force_close_at_timeout=force_close_at_timeout,
        timeout_server_hour=timeout_server_hour,
        htf_trend_direction=htf_trend_direction,
        vp_zone=vp_zone_label,
        l2_regime=l2_regime,
        l2_base_lot_factor=l2_base_lot_factor,
        htf_trend=htf_trend_label,
        divergence_direction=divergence_direction_label,
        l4_multiplier=l4_multiplier,
        l4_smt_interpretation=l4_smt_interpretation,
        htf_counter_trend=htf_counter_trend,
        htf_lot_multiplier=htf_lot_multiplier,
        fvg_final_lot_factor=fvg_final_lot_factor,
        cspa_gate_reason=cspa_gate_reason,
        cspa_tp_multiplier=cspa_tp_multiplier,
        lgr_bayes_regime=lgr_bayes_regime,
        lgr_bayes_reason=lgr_bayes_reason,
        lgr_ev_score=lgr_ev_score,
        lgr_ev_rank=lgr_ev_rank,
        lgr_ev_lot_multiplier=lgr_ev_lot_multiplier,
        ttm_bayes_win_prob=ttm_bayes_win_prob,
        ttm_ev_rank=ttm_ev_rank,
        ttm_ev_lot_multiplier=ttm_ev_lot_multiplier,
        dn_ev_rank=dn_ev_rank,
        dn_ev_bucket=dn_ev_bucket,
        dn_ev_rank_v2=dn_ev_rank_v2,
        dn_prop_gate_tier=dn_prop_gate_tier,
        dn_prop_gate_lot_multiplier=dn_prop_gate_lot_multiplier,
    )


def _apply_trade_outcome(
    pending: PendingEvaluation,
    account: AccountState,
    gbp_df: pd.DataFrame,
    eur_df: pd.DataFrame,
    bar_minutes: int,
    *,
    max_holding_bars: int | None = None,
) -> dict[str, Any]:
    """
    Phase-2: L5 未来追跡を実行し、口座状態を時系列順に更新。

    同一タイムスタンプの全ペアが Phase-1 完了後に呼ばれるため、
    2ペア目が1ペア目の未来結果を先読みするタイムトラベルは発生しない。
    """
    setup = pending.setup
    pair_df = pair_dataframe_slot(
        setup.pair,
        gbp_df,
        eur_df,
        setup_type=pending.setup_type,
    )
    holding_cap = max_holding_bars if max_holding_bars is not None else MAX_HOLDING_BARS
    sync_invalid = False
    sync_flags: tuple[str, ...] = ()

    if pending.setup_type == CSPA_SETUP_TYPE and isinstance(setup, CspaSetup):
        pip = pip_size_for_pair(setup.pair)
        effective_tp = scale_cspa_take_profit(
            setup.entry_price,
            setup.take_profit,
            setup.direction,
            pending.cspa_tp_multiplier,
        )
        shadow_result, shadow_profit_r, shadow_pips, holding, _final_sl = track_cspa_trade_outcome(
            pair_df,
            pending.start_idx,
            setup.direction,
            setup.entry_price,
            setup.stop_loss,
            effective_tp,
            bar_minutes,
            atr=setup.momentum.atr,
            max_holding_bars=holding_cap,
            pip_size=pip,
        )
    else:
        track_outcome = track_trade_outcome(
            pair_df,
            pending.start_idx,
            setup.direction,
            setup.entry_price,
            setup.stop_loss,
            setup.take_profit,
            bar_minutes,
            force_close_at_timeout=pending.force_close_at_timeout,
            timeout_server_hour=pending.timeout_server_hour,
            entry_timestamp=pd.Timestamp(setup.timestamp),
            max_holding_bars=max_holding_bars,
        )
        sync_invalid, sync_flags = _validate_l5_sync(
            track_outcome,
            pair_df,
            pd.Timestamp(setup.timestamp),
        )
        shadow_result = track_outcome.result
        shadow_profit_r = track_outcome.profit_r
        shadow_pips = track_outcome.profit_pips
        holding = track_outcome.holding_minutes

    executed = (not pending.is_reject) and pending.lot_factor > 0
    equity_before = pending.equity_before
    outcome_tags = list(pending.tags)
    for flag in sync_flags if pending.setup_type != CSPA_SETUP_TYPE else ():
        if flag not in outcome_tags:
            outcome_tags.append(flag)

    if executed:
        if sync_invalid:
            trade_result = "INVALID_SYNC"
            profit_r = 0.0
            profit_loss = 0.0
            equity_after = equity_before
            shadow_result_out = shadow_result if shadow_result in ("WIN", "LOSS") else "LOSS"
            shadow_profit_r_out = shadow_profit_r
        elif shadow_result in ("WIN", "LOSS"):
            trade_result = shadow_result
            profit_r = shadow_profit_r
            profit_loss = shadow_pips
            equity_after = account.equity + pending.risk_budget * profit_r
            account.equity = equity_after
            account.update_equity_high_water_mark()
            shadow_result_out = "NONE"
            shadow_profit_r_out = 0.0

            if trade_result == "LOSS":
                audit_dd_throttle.register_executed_streak(account, won=False)
            else:
                audit_dd_throttle.register_executed_streak(account, won=True)
            account.daily_profit_r = round(
                float(getattr(account, "daily_profit_r", 0.0) or 0.0) + float(profit_r),
                4,
            )

            if audit_rm.is_mutual_exclusion_enabled():
                account.register_executed_position(
                    setup.timestamp,
                    setup.pair,
                    pending.setup_type,
                    holding,
                )
        elif shadow_result == "EXIT_AT_SESSION_END":
            trade_result = "WIN" if shadow_profit_r > 0 else "LOSS"
            profit_r = shadow_profit_r
            profit_loss = shadow_pips
            equity_after = account.equity + pending.risk_budget * profit_r
            account.equity = equity_after
            account.update_equity_high_water_mark()
            shadow_result_out = "NONE"
            shadow_profit_r_out = 0.0
            if trade_result == "LOSS":
                audit_dd_throttle.register_executed_streak(account, won=False)
            else:
                audit_dd_throttle.register_executed_streak(account, won=True)
            account.daily_profit_r = round(
                float(getattr(account, "daily_profit_r", 0.0) or 0.0) + float(profit_r),
                4,
            )
            if audit_rm.is_mutual_exclusion_enabled():
                account.register_executed_position(
                    setup.timestamp,
                    setup.pair,
                    pending.setup_type,
                    holding,
                )
        else:
            trade_result = "INVALID_SYNC"
            profit_r = 0.0
            profit_loss = 0.0
            equity_after = equity_before
            shadow_result_out = "LOSS"
            shadow_profit_r_out = shadow_profit_r
            if "INVALID_SYNC" not in outcome_tags:
                outcome_tags.append("INVALID_SYNC")
    else:
        trade_result = "NOT_EXECUTED"
        profit_r = 0.0
        profit_loss = 0.0
        equity_after = equity_before
        shadow_result_out = shadow_result if shadow_result in ("WIN", "LOSS") else "LOSS"
        # v1.2: 連続R倍数を二値化せず track_trade_outcome の生値をそのまま記録
        shadow_profit_r_out = shadow_profit_r
        if shadow_result == "WIN":
            account.consecutive_losses = 0

    return {
        "trade_id": pending.trade_id,
        "timestamp": setup.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "pair": setup.pair,
        "equity_before_trade": round(equity_before, 2),
        "equity_after_trade": round(equity_after, 2),
        "daily_dd_remaining_percent": round(pending.daily_rem, 4),
        "monthly_dd_remaining_percent": round(pending.monthly_rem, 4),
        "setup_type": pending.setup_type,
        "candidate_score": pending.candidate_score,
        "bayes_probability": pending.bayes_probability,
        "smt_intensity": round(pending.smt, 2),
        "model_version": resolve_model_version(pending.setup_type),
        "reason_codes": json.dumps(outcome_tags, ensure_ascii=False),
        "risk_score": pending.risk_score,
        "llm_latency_ms": pending.latency,
        "decision_source": pending.decision_source,
        "lot_factor": pending.lot_factor,
        "llm_score": pending.llm_confidence_score,
        "final_lot_size": round(pending.final_lot_size if pending.final_lot_size > 0 else pending.lot_size, 4),
        "entry_price": round(setup.entry_price, 5),
        "stop_loss": round(setup.stop_loss, 5),
        "take_profit": round(setup.take_profit, 5),
        "trade_result": trade_result,
        "profit_loss": round(profit_loss, 2),
        "profit_r": round(profit_r, 2),
        "holding_time": holding,
        "shadow_result": shadow_result_out,
        "shadow_profit_r": shadow_profit_r_out,
        "smt_diff": round(pending.smt_diff, 4),
        "smt_leader": pending.smt_leader,
        "wick_ratio_pct": round(getattr(setup, "wick_ratio_pct", 0.0), 4),
        "atr_ratio": round(pending.atr_ratio, 4),
        "has_bos": pending.has_bos,
        "vp_zone": pending.vp_zone,
        "l2_regime": pending.l2_regime,
        "l2_base_lot_factor": round(pending.l2_base_lot_factor, 4),
        "htf_trend": pending.htf_trend,
        "divergence_direction": pending.divergence_direction,
        "l4_multiplier": round(pending.l4_multiplier, 4),
        "l4_smt_interpretation": pending.l4_smt_interpretation,
        "htf_counter_trend": pending.htf_counter_trend,
        "htf_lot_multiplier": round(pending.htf_lot_multiplier, 4),
        "fvg_final_lot_factor": round(pending.fvg_final_lot_factor, 4),
        "ev_rank": round(
            pending.dn_ev_rank_v2
            if isinstance(setup, DiNapoliSetup) and pending.dn_prop_gate_tier
            else (
                pending.dn_ev_rank
                if isinstance(setup, DiNapoliSetup) and pending.dn_ev_bucket
                else pending.ttm_ev_rank
            ),
            6,
        ),
        "ev_lot_multiplier": round(
            pending.dn_prop_gate_lot_multiplier
            if isinstance(setup, DiNapoliSetup) and pending.dn_prop_gate_tier
            else pending.ttm_ev_lot_multiplier,
            4,
        ),
        "sized_result_r": round(profit_r, 4),
        "_setup": setup,
        "_pending": pending,
    }


# =============================================================================
# パイプライン統合
# =============================================================================
def resolve_final_decision(
    l0_fail: bool,
    l1_fail: bool,
    l2_fail: bool,
    llm_decision: str,
) -> str:
    """各レイヤーの拒否優先順位で最終 decision_source を決定。"""
    if l0_fail:
        return "REJECT_BY_L0"
    if l1_fail:
        return "REJECT_BY_L1"
    if l2_fail:
        return "REJECT_BY_L2"
    return llm_decision


def run_pipeline(
    gbp_df: pd.DataFrame,
    eur_df: pd.DataFrame,
    h1_gbp: pd.DataFrame,
    h1_eur: pd.DataFrame,
    track_df: pd.DataFrame,
    bar_minutes: int,
) -> list[dict[str, Any]]:
    """
    全セットアップに対し L0〜L6 を実行し、監査レコードリストを返す。

    v1.2: 同一タイムスタンプ(ts)の複数ペアは
      Phase-1 → L0〜L4.5 を口座スナップショットで一括判定
      Phase-2 → L5 未来追跡を順次実行し equity を更新

    v3.0: StrategyResult 経由で L1-L3 を戦略層へ委譲。
      日次DDテーパーは audit/risk_manager (4.5% 安全線)。
    """
    account = AccountState(profile=PROP_FIRM_PROFILE)
    records: list[dict[str, Any]] = []
    bayes_engine = BayesEngine()

    strategies = build_strategy_registry()
    london = strategies[0]
    gbp_setups = london.detect_setups(gbp_df, "GBPUSD", h1_gbp)
    eur_setups = london.detect_setups(eur_df, "EURUSD", h1_eur)

    gbp_by_ts = {s.timestamp: s for s in gbp_setups}
    eur_by_ts = {s.timestamp: s for s in eur_setups}

    all_timestamps = sorted(set(gbp_by_ts.keys()) | set(eur_by_ts.keys()))

    for ts in all_timestamps:
        gbp_s = gbp_by_ts.get(ts)
        eur_s = eur_by_ts.get(ts)
        setups_at_ts = list(filter(None, [gbp_s, eur_s]))
        if not setups_at_ts:
            continue

        # --- Phase-0: このタイムスタンプ時点の口座スナップショットを固定 ---
        account.update_calendar(ts)
        equity_snapshot = account.equity
        daily_rem = account.daily_dd_remaining()
        monthly_rem = account.monthly_dd_remaining()
        current_daily_loss = account.daily_loss_fraction()

        # --- Phase-1: 未来結果を見る前に全ペアの執行判定を完了 ---
        pending_list: list[PendingEvaluation] = []
        for setup in setups_at_ts:
            pending_list.append(
                _evaluate_setup_at_timestamp(
                    london, setup, gbp_s, eur_s, account,
                    equity_snapshot, daily_rem, monthly_rem,
                    current_daily_loss,
                    h1_gbp, h1_eur, gbp_df, eur_df,
                    bayes_engine,
                )
            )

        # --- Phase-2: L5 追跡 → 口座更新 → L3.5学習 → L6 ログ出力 ---
        for pending in pending_list:
            raw_record = _apply_trade_outcome(
                pending, account, gbp_df, eur_df, bar_minutes
            )
            setup = raw_record.pop("_setup")
            pending_ctx = raw_record.pop("_pending")

            won = raw_record["trade_result"] == "WIN" or (
                raw_record["trade_result"] == "NOT_EXECUTED"
                and raw_record["shadow_result"] == "WIN"
            )
            if not is_bayes_bypass_setup_type(pending_ctx.setup_type) and pending_ctx.setup_type != CSPA_SETUP_TYPE:
                bayes_engine.record_outcome(
                    setup.timestamp,
                    pending_ctx.smt,
                    pending_ctx.candidate_score,
                    pending_ctx.consecutive_losses_snapshot,
                    pending_ctx.has_bos,
                    pending_ctx.both_sweep,
                    won,
                )

            record = {k: v for k, v in raw_record.items() if k in CSV_COLUMNS}
            records.append(record)

            _write_audit_json(pending.trade_id, record, setup, account, {
                "minutes_to_major_news": pending_ctx.minutes_to_news,
                "risk_budget": pending_ctx.risk_budget,
                "lot_size": pending_ctx.lot_size,
                "gbp_sweep": pending_ctx.gbp_s is not None,
                "eur_sweep": pending_ctx.eur_s is not None,
                "has_bos": pending_ctx.has_bos,
                "smt_diff": round(pending_ctx.smt_diff, 4),
                "smt_leader": pending_ctx.smt_leader,
                "wick_ratio_pct": round(setup.wick_ratio_pct, 4),
                "bayes_probability": pending_ctx.bayes_probability,
                "atr_ratio": round(pending_ctx.atr_ratio, 4),
                "timeframe": TIMEFRAME_LABEL,
            })

        account.last_event_timestamp = ts

    return records


def _json_safe(obj: Any) -> Any:
    """numpy/pandas型を標準JSON互換型へ変換。"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _write_audit_json(
    trade_id: str,
    record: dict[str, Any],
    setup: SetupUnion,
    account: AccountState,
    extra: dict[str, Any],
) -> None:
    """L6：生JSONアーカイブを個別保存。"""
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    if isinstance(setup, LsfcSetup):
        market_data: dict[str, Any] = {
            "pair": setup.pair,
            "direction": setup.direction,
            "pool_high": setup.pool_high,
            "pool_low": setup.pool_low,
            "sweep_level": setup.sweep_level,
            "sweep_extreme": setup.sweep_extreme,
            "failure_extreme": setup.failure_extreme,
            "failure_retracement_depth": setup.failure_retracement_depth,
            "atr": setup.atr,
            "sweep_distance": setup.sweep_distance,
        }
    elif isinstance(setup, AlsSetup):
        market_data = {
            "pair": setup.pair,
            "direction": setup.direction,
            "asia_high": setup.asia_high,
            "asia_low": setup.asia_low,
            "asia_equilibrium_price": setup.asia_equilibrium_price,
            "asia_range_pips": setup.asia_range_pips,
            "h1_atr": setup.h1_atr,
            "asia_range_atr_ratio": setup.asia_range_atr_ratio,
            "sweep_extreme": setup.sweep_extreme,
            "wick_ratio_pct": setup.wick_ratio_pct,
            "vwap_deviation_ratio": setup.vwap_deviation_ratio,
            "vwap": setup.vwap,
            "inside_return": setup.inside_return,
            "tp_target_type": setup.tp_target_type,
            "sweep_bar": {
                "open": setup.sweep_bar_open,
                "high": setup.sweep_bar_high,
                "low": setup.sweep_bar_low,
                "close": setup.sweep_bar_close,
            },
            "reason_codes": setup.reason_codes,
            "atr": setup.atr,
            "sweep_distance": setup.sweep_distance,
        }
    elif isinstance(setup, TrefSetup):
        market_data = {
            "pair": setup.pair,
            "direction": setup.direction,
            "range_high": setup.range_high,
            "range_low": setup.range_low,
            "range_width_pips": setup.range_width_pips,
            "expansion_depth_pips": setup.expansion_depth_pips,
            "bars_stayed_outside_m5": setup.bars_stayed_outside_m5,
            "payload": setup.payload,
            "score_breakdown": setup.score_breakdown,
            "sweep_distance": setup.sweep_distance,
        }
    elif isinstance(setup, DtpaSetup):
        market_data = {
            "pair": setup.pair,
            "direction": setup.direction,
            "bos_direction": setup.bos.direction,
            "broken_level": setup.bos.broken_level,
            "pullback_status": setup.pullback.status,
            "pa_trigger_type": setup.pa_trigger.trigger_type,
            "risk_reward": setup.risk_reward,
            "reason_codes": list(setup.reason_codes),
        }
    elif isinstance(setup, CspaSetup):
        market_data = {
            "pair": setup.pair,
            "direction": setup.direction,
            "h4_phase": setup.bias_phase,
            "bias_tf": "H1",
            "structure_tf": "M15",
            "trigger_tf": "M1",
            "retrace_ratio": setup.retrace_ratio,
            "stagnation_bars": setup.stagnation.bar_count,
            "momentum_type": setup.momentum.trigger_type,
            "risk_reward": setup.risk_reward,
            "reason_codes": list(setup.reason_codes),
            "score_breakdown": setup.score_breakdown.as_dict(),
        }
    elif isinstance(setup, SpringSetup):
        market_data = {
            "pair": setup.pair,
            "direction": setup.direction,
            "sc_price": setup.accumulation.sc_price,
            "ar_price": setup.accumulation.ar_price,
            "support_level": setup.accumulation.support_level,
            "spring_depth_atr": setup.spring_depth_atr,
            "spring_attempt_number": setup.spring_attempt_number,
            "risk_reward": setup.risk_reward,
            "wyckoff_features": setup.wyckoff_features.as_dict(),
            "reason_codes": list(setup.reason_codes),
        }
    elif isinstance(setup, LgrSetup):
        market_data = {
            "pair": setup.pair,
            "direction": setup.direction,
            "grab_price": setup.grab_price,
            "liquidity_pool_type": setup.liquidity_pool_type,
            "risk_reward": setup.risk_reward,
            "lgr_features": setup.lgr_features.as_dict(),
            "reason_codes": list(setup.reason_codes),
        }
    else:
        market_data = {
            "pair": setup.pair,
            "direction": setup.direction,
            "asia_high": setup.asia_high,
            "asia_low": setup.asia_low,
            "atr": setup.atr,
        }
    payload = _json_safe({
        "trade_id": trade_id,
        "audit_record": record,
        "market_data": market_data,
        "account_status": {
            "equity": account.equity,
            "consecutive_losses": account.consecutive_losses,
            "daily_start_equity": account.daily_start_equity,
            "monthly_start_equity": account.monthly_start_equity,
        },
        "pipeline_context": extra,
    })
    path = JSON_DIR / f"{trade_id}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def append_csv_log(records: list[dict[str, Any]]) -> None:
    """26カラムCSVへ追記（ヘッダー自動生成）。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records, columns=CSV_COLUMNS)
    header_needed = not CSV_LOG_PATH.exists()
    df.to_csv(CSV_LOG_PATH, mode="a", index=False, header=header_needed, encoding="utf-8-sig")


def _calc_pip_value(account_equity: float, base_risk_pct: float | None = None) -> float:
    """1Rあたりの金額換算（口座残高 × base_risk_pct）。"""
    pct = BASE_RISK_PCT if base_risk_pct is None else float(base_risk_pct)
    return round(float(account_equity) * pct, 2)


def export_backtest_html_report(
    df: pd.DataFrame,
    output_csv_path: Path | str,
    *,
    account_equity: float | None = None,
    ea_name: str = "Prop_EA_Project",
) -> str:
    """バックテスト監査CSVと同名パスへ MT5 準拠 HTML レポートを出力。"""
    import logging

    from reporting.html_report import BacktestHtmlReport

    equity = float(account_equity if account_equity is not None else STARTING_EQUITY)
    html_path = str(output_csv_path).replace(".csv", ".html")
    report = BacktestHtmlReport(
        df=df,
        ea_name=ea_name,
        output_path=html_path,
        pip_value=_calc_pip_value(equity),
    )
    path = report.generate()
    logging.getLogger(__name__).info("HTMLレポート保存: %s", path)
    return path


# =============================================================================
# L7 統計レポート
# =============================================================================
def print_l7_report(records: list[dict[str, Any]]) -> None:
    """RISK_TABLE調整用の統計をコンソール出力。"""
    if not records:
        print("\n[L7] イベントが検出されませんでした。")
        return

    total = len(records)
    executed = sum(1 for r in records if r["trade_result"] != "NOT_EXECUTED")
    shadow = total - executed

    print("\n" + "=" * 72)
    print(" L7: RISK_TABLE 調整用 統計レポート")
    print("=" * 72)
    print(f"  総イベント数       : {total}")
    print(f"  実際の執行件数     : {executed}")
    print(f"  見送り(シャドー)件数: {shadow}")

    print("\n--- decision_source 別集計 ---")
    ds_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "total_r": 0.0})
    for r in records:
        ds = r["decision_source"]
        ds_stats[ds]["count"] += 1
        if r["trade_result"] != "NOT_EXECUTED":
            ds_stats[ds]["total_r"] += r["profit_r"]
        else:
            ds_stats[ds]["total_r"] += r["shadow_profit_r"]

    print(f"  {'decision_source':<20} {'件数':>6} {'総R倍数':>10}")
    print("  " + "-" * 40)
    for ds in sorted(ds_stats.keys()):
        s = ds_stats[ds]
        print(f"  {ds:<20} {int(s['count']):>6} {s['total_r']:>10.2f}")

    print("\n--- reason_codes（リスクタグ）別シャドー分析 ---")
    tag_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0, "shadow_wins": 0, "shadow_total_r": 0.0}
    )
    for r in records:
        tags = json.loads(r["reason_codes"])
        shadow_r = r["shadow_profit_r"] if r["trade_result"] == "NOT_EXECUTED" else r["profit_r"]
        shadow_win = (r["shadow_result"] == "WIN") or (r["trade_result"] == "WIN")
        for tag in tags:
            tag_stats[tag]["count"] += 1
            tag_stats[tag]["shadow_total_r"] += shadow_r
            if shadow_win:
                tag_stats[tag]["shadow_wins"] += 1

    print(f"  {'タグ':<25} {'件数':>5} {'勝率':>8} {'期待R':>8}")
    print("  " + "-" * 50)
    for tag in sorted(tag_stats.keys()):
        s = tag_stats[tag]
        win_rate = s["shadow_wins"] / s["count"] * 100 if s["count"] else 0.0
        expectancy = s["shadow_total_r"] / s["count"] if s["count"] else 0.0
        print(f"  {tag:<25} {int(s['count']):>5} {win_rate:>7.1f}% {expectancy:>8.3f}")

    print("=" * 72 + "\n")


# =============================================================================
# Live API — MT5 Bridge 連携 (v1.7+)
# =============================================================================
LIVE_BAR_BUFFER_MAX = 600
BROKER_POSITION_SETUP_TYPE = "BROKER_POSITION"
BROKER_POSITION_HORIZON = pd.Timedelta(days=365)
LIVE_EXEC_BAR_MINUTES = 5  # PropEA_Bridge.mq5 uses PERIOD_M5


def normalize_pair_name(raw: str) -> str | None:
    """
    ブローカー固有サフィックスを除去し canonical ペア名へ正規化する。

    例: GBPUSDp / GBPUSD.pro / GBPUSD_m → GBPUSD
    """
    token = raw.upper().replace(".", "").replace("_", "").replace("-", "").replace(" ", "")
    if "GBPUSD" in token:
        return "GBPUSD"
    if "EURUSD" in token:
        return "EURUSD"
    if "AUDUSD" in token:
        return "AUDUSD"
    if "NZDUSD" in token:
        return "NZDUSD"
    return None


@dataclass
class LivePipelineState:
    """MT5ブリッジ用のセッション状態（口座・ベイズ・OHLCVバッファ）。"""

    account: AccountState
    bayes_engine: BayesEngine
    gbp_df: pd.DataFrame
    eur_df: pd.DataFrame
    tref_bayes: Any = None
    sentinel: Any = None  # LiveSentinelState — lazy import avoided in dataclass field

    @classmethod
    def create(cls) -> LivePipelineState:
        from audit.live_sentinel import LiveSentinelState
        from audit.tref_bayes_filter import TrefBayesFilter

        empty = pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
        return cls(
            account=AccountState(profile=PROP_FIRM_PROFILE),
            bayes_engine=BayesEngine(),
            tref_bayes=TrefBayesFilter(),
            gbp_df=empty.copy(),
            eur_df=empty.copy(),
            sentinel=LiveSentinelState.create(),
        )


def bars_payload_to_dataframe(bars: list[dict[str, Any]]) -> pd.DataFrame:
    """MT5から受信したOHLCV配列を DataFrame へ変換。"""
    if not bars:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    rows = []
    for bar in bars:
        ts = bar.get("time") or bar.get("datetime")
        rows.append(
            {
                "datetime": pd.Timestamp(ts),
                "open": float(bar["open"]),
                "high": float(bar["high"]),
                "low": float(bar["low"]),
                "close": float(bar["close"]),
                "volume": float(bar.get("volume", bar.get("tick_volume", 0))),
            }
        )
    df = pd.DataFrame(rows).sort_values("datetime").drop_duplicates("datetime", keep="last")
    return df.reset_index(drop=True)


def market_payload_to_bar(
    market: dict[str, Any],
    bar_time: str | None = None,
) -> dict[str, Any]:
    """単一 market オブジェクトを1本のバー dict へ正規化。"""
    ts = bar_time or market.get("time") or market.get("datetime")
    if ts is None:
        ts = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "time": ts,
        "open": market["open"],
        "high": market["high"],
        "low": market["low"],
        "close": market["close"],
        "volume": market.get("volume", 0),
    }


def upsert_live_bars(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """ライブバッファへ追記し、最大件数でトリム。"""
    if incoming.empty:
        return existing
    if existing.empty:
        merged = incoming.copy()
    else:
        merged = (
            pd.concat([existing, incoming], ignore_index=True)
            .sort_values("datetime")
            .drop_duplicates("datetime", keep="last")
        )
    if len(merged) > LIVE_BAR_BUFFER_MAX:
        merged = merged.iloc[-LIVE_BAR_BUFFER_MAX:].reset_index(drop=True)
    return merged


def sync_live_account(
    account: AccountState,
    equity: float,
    balance: float,
    bar_timestamp: pd.Timestamp,
    *,
    server_timestamp: pd.Timestamp | None = None,
) -> None:
    """MT5口座スナップショットを AccountState へ反映。"""
    calendar_ts = server_timestamp if server_timestamp is not None else bar_timestamp
    account.update_calendar(calendar_ts)
    account.equity = float(equity)
    _ = balance  # Sentinel / 将来の証拠金チェック用


def sync_broker_open_positions(
    account: AccountState,
    broker_positions: list[dict[str, Any]] | None,
    ts: pd.Timestamp,
) -> None:
    """
    Live VPS: MT5 から送られた open_positions を L2 同一シンボル1ポジション制限へ同期。

    ブローカー実保有が正。リストが空なら open_positions をクリアする。
    """
    if not audit_rm.is_mutual_exclusion_enabled():
        return
    account.purge_closed_positions(ts)
    if not broker_positions:
        account.open_positions.clear()
        return

    synced: list[audit_rm.OpenPosition] = []
    seen: set[str] = set()
    for item in broker_positions:
        if not isinstance(item, dict):
            continue
        pair = normalize_pair_name(str(item.get("pair", "")))
        if pair is None or pair in seen:
            continue
        seen.add(pair)
        entry_raw = item.get("entry_time") or item.get("entry_ts")
        try:
            entry = pd.Timestamp(entry_raw) if entry_raw else ts
        except (TypeError, ValueError):
            entry = ts
        setup_type = str(item.get("setup_type") or BROKER_POSITION_SETUP_TYPE).strip()
        synced.append(
            audit_rm.OpenPosition(
                pair=pair,
                setup_type=setup_type,
                entry_ts=entry,
                close_ts=ts + BROKER_POSITION_HORIZON,
            )
        )
    account.open_positions = synced


def _pair_dataframes(
    state: LivePipelineState,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """M5トラック + H1構造参照 DataFrame を返す。"""
    gbp_df = state.gbp_df
    eur_df = state.eur_df
    if MODE_H1:
        return gbp_df, eur_df, gbp_df, eur_df
    h1_gbp = resample_to_h1(gbp_df) if not gbp_df.empty else gbp_df
    h1_eur = resample_to_h1(eur_df) if not eur_df.empty else eur_df
    return gbp_df, eur_df, h1_gbp, h1_eur


def _detect_setups_for_live_strategy(
    strategy: BaseStrategy,
    gbp_df: pd.DataFrame,
    eur_df: pd.DataFrame,
    h1_gbp: pd.DataFrame,
    h1_eur: pd.DataFrame,
) -> tuple[list[SetupUnion], list[SetupUnion], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Live セットアップ検出 — 戦略ごとに exec/structure 足種を解決する。

    DBBS: exec M15 / structure H1 / ATR H4（M5 バッファからリサンプル）。
    """
    from strategies.dbbs import DbbsStrategy

    if isinstance(strategy, DbbsStrategy):
        m15_gbp = resample_to_m15(gbp_df) if not gbp_df.empty else gbp_df
        m15_eur = resample_to_m15(eur_df) if not eur_df.empty else eur_df
        h4_gbp = resample_to_h4(h1_gbp) if not h1_gbp.empty else h1_gbp
        h4_eur = resample_to_h4(h1_eur) if not h1_eur.empty else h1_eur
        gbp_setups = strategy.detect_setups(m15_gbp, "GBPUSD", h1_gbp, h4_gbp) if not m15_gbp.empty else []
        eur_setups = strategy.detect_setups(m15_eur, "EURUSD", h1_eur, h4_eur) if not m15_eur.empty else []
        return gbp_setups, eur_setups, m15_gbp, m15_eur, h1_gbp, h1_eur

    gbp_setups = strategy.detect_setups(gbp_df, "GBPUSD", h1_gbp) if not gbp_df.empty else []
    eur_setups = strategy.detect_setups(eur_df, "EURUSD", h1_eur) if not eur_df.empty else []
    return gbp_setups, eur_setups, gbp_df, eur_df, h1_gbp, h1_eur


def _find_active_setup(
    setups: list[SetupUnion],
    pair: str,
    bar_timestamp: pd.Timestamp,
) -> SetupUnion | None:
    """指定ペア・バー時刻に一致する最新セットアップを返す。"""
    same_day = [
        s
        for s in setups
        if s.pair == pair and s.timestamp.normalize() == bar_timestamp.normalize()
    ]
    if not same_day:
        return None
    exact = [s for s in same_day if s.timestamp == bar_timestamp]
    return exact[0] if exact else same_day[-1]


def evaluate_precomputed_setup(
    setup: SetupUnion,
    strategy: BaseStrategy,
    state: LivePipelineState,
    gbp_df: pd.DataFrame,
    eur_df: pd.DataFrame,
    setups_at_ts: list[SetupUnion],
    minutes_to_news_override: int | None = None,
    htf_gbp: pd.DataFrame | None = None,
    htf_eur: pd.DataFrame | None = None,
    structure_h1_gbp: pd.DataFrame | None = None,
    structure_h1_eur: pd.DataFrame | None = None,
) -> PendingEvaluation:
    """バックテスト用: 事前検出済みセットアップを L0〜L4.5 判定（Live 再検出なし）。"""
    if MODE_H1:
        h1_gbp, h1_eur = gbp_df, eur_df
    elif isinstance(setup, AlsSetup):
        h1_gbp = resample_to_m15(gbp_df) if not gbp_df.empty else gbp_df
        h1_eur = resample_to_m15(eur_df) if not eur_df.empty else eur_df
    elif isinstance(setup, TrefSetup):
        h1_gbp = resample_to_h1(gbp_df) if not gbp_df.empty else gbp_df
        h1_eur = resample_to_h1(eur_df) if not eur_df.empty else eur_df
    elif isinstance(setup, DbbsSetup):
        h1_gbp = resample_to_h1(gbp_df) if not gbp_df.empty else gbp_df
        h1_eur = resample_to_h1(eur_df) if not eur_df.empty else eur_df
        gbp_df = resample_to_m15(gbp_df) if not gbp_df.empty else gbp_df
        eur_df = resample_to_m15(eur_df) if not eur_df.empty else eur_df
    elif structure_h1_gbp is not None and structure_h1_eur is not None:
        h1_gbp, h1_eur = structure_h1_gbp, structure_h1_eur
    else:
        h1_gbp = resample_to_h1(gbp_df) if not gbp_df.empty else gbp_df
        h1_eur = resample_to_h1(eur_df) if not eur_df.empty else eur_df

    if uses_primary_dataframe(setup.pair):
        gbp_s: SetupUnion | None = setup
        eur_s = next(
            (
                s
                for s in setups_at_ts
                if s.pair == correlated_pair(setup.pair) and type(s) is type(setup)
            ),
            None,
        )
    else:
        eur_s = setup
        gbp_s = next(
            (
                s
                for s in setups_at_ts
                if s.pair == correlated_pair(setup.pair) and type(s) is type(setup)
            ),
            None,
        )

    equity_snapshot = state.account.equity
    daily_rem = state.account.daily_dd_remaining()
    monthly_rem = state.account.monthly_dd_remaining()
    daily_loss_fraction = state.account.daily_loss_fraction()

    return _evaluate_setup_at_timestamp(
        strategy,
        setup,
        gbp_s,
        eur_s,
        state.account,
        equity_snapshot,
        daily_rem,
        monthly_rem,
        daily_loss_fraction,
        h1_gbp,
        h1_eur,
        gbp_df,
        eur_df,
        state.bayes_engine,
        tref_bayes_filter=state.tref_bayes,
        minutes_to_news_override=minutes_to_news_override,
        htf_gbp=htf_gbp,
        htf_eur=htf_eur,
    )


def pending_to_trade_signal(pending: PendingEvaluation) -> dict[str, Any]:
    """Phase-1 判定結果を MT5 向けレスポンス dict へ変換。"""
    setup = pending.setup
    if pending.is_reject or pending.lot_size <= 0:
        action = "REJECT"
    elif pending.decision_source in ("ALLOW", "CAUTION"):
        action = setup.direction
    else:
        action = "HOLD"

    message = (
        f"{pending.decision_source} | score={pending.candidate_score:.1f} | "
        f"bayes={pending.bayes_probability:.2f} | htf={pending.htf_trend_direction} | "
        f"tags={','.join(pending.tags)}"
    )
    signal = {
        "action": action,
        "lot_size": float(pending.lot_size) if action in ("BUY", "SELL") else 0.0,
        "risk_budget": float(pending.risk_budget) if action in ("BUY", "SELL") else 0.0,
        "sl": float(round(setup.stop_loss, 5)),
        "tp": float(round(setup.take_profit, 5)),
        "entry": float(round(setup.entry_price, 5)),
        "message": message,
        "decision_source": pending.decision_source,
        "trade_id": pending.trade_id,
        "lot_factor": float(pending.lot_factor),
        "risk_score": pending.risk_score,
        "multipliers": compute_l45_multipliers(
            pending.candidate_score,
            pending.monthly_rem,
            pending.consecutive_losses_snapshot,
            pending.decision_source if pending.decision_source in ("ALLOW", "CAUTION") else "REJECT",
            pending.bayes_probability,
            pending.profile,
        ),
    }
    if pending.force_close_at_timeout and pending.timeout_server_hour > 0:
        signal["force_close_at_timeout"] = True
        signal["timeout_server_hour"] = pending.timeout_server_hour
    if pending.setup_type == CSPA_SETUP_TYPE and isinstance(setup, CspaSetup):
        signal.update(build_cspa_exit_signal_fields(setup))
    return signal


def evaluate_trade_signal_with_pending(
    payload: dict[str, Any],
    state: LivePipelineState | None = None,
) -> tuple[dict[str, Any], PendingEvaluation | None]:
    """
    MT5 JSON ペイロードを受け取り L0〜L4.5 判定を実行する。

    Returns:
        (trade_signal, pending) — セットアップ未検出時 pending は None
    """
    if state is None:
        state = LivePipelineState.create()

    market = payload["market"]
    calendar = payload.get("calendar", {})
    account_info = payload["account"]
    pair = normalize_pair_name(str(market.get("pair", "")))
    if pair is None:
        raw_pair = str(market.get("pair", "")).upper()
        return (
            {
                "action": "HOLD",
                "lot_size": 0.0,
                "sl": 0.0,
                "tp": 0.0,
                "message": f"Unsupported pair: {raw_pair}",
            },
            None,
        )

    bar_time = payload.get("bar_time")
    primary_bar = market_payload_to_bar(market, bar_time)
    bar_timestamp = pd.Timestamp(primary_bar["time"])

    server_time_raw = payload.get("server_time") or bar_time or str(bar_timestamp)
    server_timestamp = pd.Timestamp(parse_server_time(server_time_raw))
    balance = float(account_info.get("balance", RM_STARTING_EQUITY))
    equity = float(account_info.get("equity", RM_STARTING_EQUITY))
    spread_raw = payload.get("spread_points")
    spread_points: int | None = int(spread_raw) if spread_raw is not None else None

    incoming_primary = bars_payload_to_dataframe(payload.get("bars") or [primary_bar])
    if pair == "GBPUSD":
        state.gbp_df = upsert_live_bars(state.gbp_df, incoming_primary)
    else:
        state.eur_df = upsert_live_bars(state.eur_df, incoming_primary)

    corr_market = payload.get("correlated_market")
    if corr_market:
        corr_pair = normalize_pair_name(str(corr_market.get("pair", CORRELATED_PAIR[pair])))
        if corr_pair is None:
            corr_pair = CORRELATED_PAIR[pair]
        corr_bar = market_payload_to_bar(corr_market, payload.get("correlated_bar_time"))
        incoming_corr = bars_payload_to_dataframe(payload.get("correlated_bars") or [corr_bar])
        if corr_pair == "GBPUSD":
            state.gbp_df = upsert_live_bars(state.gbp_df, incoming_corr)
        elif corr_pair == "EURUSD":
            state.eur_df = upsert_live_bars(state.eur_df, incoming_corr)

    sync_live_account(
        state.account,
        equity=equity,
        balance=balance,
        bar_timestamp=bar_timestamp,
        server_timestamp=server_timestamp,
    )
    sync_broker_open_positions(
        state.account,
        payload.get("open_positions"),
        server_timestamp,
    )

    if is_live_sentinel_enabled():
        verdict = evaluate_live_sentinel(
            state.sentinel,
            server_timestamp.to_pydatetime(),
            balance,
            equity,
            spread_points=spread_points,
        )
        if verdict.panic_close:
            return sentinel_panic_signal(verdict.message, tags=verdict.tags), None
        if not verdict.entry_allowed:
            return sentinel_hold_signal(verdict.message, tags=verdict.tags), None

    gbp_df, eur_df, h1_gbp, h1_eur = _pair_dataframes(state)
    if gbp_df.empty and eur_df.empty:
        return (
            {
                "action": "HOLD",
                "lot_size": 0.0,
                "sl": 0.0,
                "tp": 0.0,
                "message": "Insufficient bar history — send bars or repeat requests to fill buffer",
            },
            None,
        )

    from strategies import is_live_strategy_mode, resolve_strategy_mode

    live_strategies = build_strategy_registry(live_only=True)
    if not live_strategies:
        return (
            {
                "action": "HOLD",
                "lot_size": 0.0,
                "sl": 0.0,
                "tp": 0.0,
                "message": "No live-eligible strategies registered (STRATEGY_LETTER_BY_MODE)",
            },
            None,
        )

    london = live_strategies[0]
    gbp_setups, eur_setups, track_gbp, track_eur, h1_gbp, h1_eur = _detect_setups_for_live_strategy(
        london, gbp_df, eur_df, h1_gbp, h1_eur
    )

    active = _find_active_setup(
        gbp_setups if pair == "GBPUSD" else eur_setups,
        pair,
        bar_timestamp,
    )
    matched_strategy = london
    if active is None:
        for strategy in live_strategies[1:]:
            gbp_setups, eur_setups, track_gbp, track_eur, h1_gbp, h1_eur = _detect_setups_for_live_strategy(
                strategy, gbp_df, eur_df, h1_gbp, h1_eur
            )
            active = _find_active_setup(
                gbp_setups if pair == "GBPUSD" else eur_setups,
                pair,
                bar_timestamp,
            )
            if active is not None:
                matched_strategy = strategy
                break
    if active is None:
        return (
            {
                "action": "HOLD",
                "lot_size": 0.0,
                "sl": 0.0,
                "tp": 0.0,
                "message": "No active setup at this bar",
            },
            None,
        )

    if not is_live_strategy_mode(resolve_strategy_mode(matched_strategy)):
        return (
            {
                "action": "HOLD",
                "lot_size": 0.0,
                "sl": 0.0,
                "tp": 0.0,
                "message": (
                    f"Strategy {resolve_strategy_mode(matched_strategy)} is not letter-registered "
                    "for live trading — no MT5 order"
                ),
            },
            None,
        )

    gbp_setups, eur_setups, track_gbp, track_eur, h1_gbp, h1_eur = _detect_setups_for_live_strategy(
        matched_strategy, gbp_df, eur_df, h1_gbp, h1_eur
    )
    gbp_s = _find_active_setup(gbp_setups, "GBPUSD", bar_timestamp)
    eur_s = _find_active_setup(eur_setups, "EURUSD", bar_timestamp)
    if active.pair == "GBPUSD":
        gbp_s = active
    else:
        eur_s = active

    minutes_to_news = int(calendar.get("minutes_to_next_news", DEFAULT_MINUTES_TO_NEWS))
    impact = str(calendar.get("news_impact_level", "LOW")).upper()
    if "minutes_to_next_news" not in calendar or calendar.get("minutes_to_next_news") == DEFAULT_MINUTES_TO_NEWS:
        cached_minutes, cached_impact, _ = _read_calendar_cache_for_live(bar_timestamp)
        if cached_minutes is not None:
            minutes_to_news = cached_minutes
        if cached_impact:
            impact = cached_impact
    if impact == "HIGH" and minutes_to_news <= NEWS_REJECT_THRESHOLD_MIN:
        minutes_to_news = NEWS_REJECT_THRESHOLD_MIN

    equity_snapshot = state.account.equity
    daily_rem = state.account.daily_dd_remaining()
    monthly_rem = state.account.monthly_dd_remaining()
    daily_loss_fraction = state.account.daily_loss_fraction()

    pending = _evaluate_setup_at_timestamp(
        matched_strategy,
        active,
        gbp_s,
        eur_s,
        state.account,
        equity_snapshot,
        daily_rem,
        monthly_rem,
        daily_loss_fraction,
        h1_gbp,
        h1_eur,
        track_gbp,
        track_eur,
        state.bayes_engine,
        tref_bayes_filter=state.tref_bayes,
        minutes_to_news_override=minutes_to_news,
    )
    return pending_to_trade_signal(pending), pending


def evaluate_trade_signal(
    payload: dict[str, Any],
    state: LivePipelineState | None = None,
) -> dict[str, Any]:
    """
    MT5 JSON ペイロードを受け取り L0〜L4.5 判定を実行し trade_signal を返す。

    payload 構造:
      market, calendar, account — 必須
      bar_time, bars, correlated_market, correlated_bars — 任意
    """
    signal, _ = evaluate_trade_signal_with_pending(payload, state)
    return signal


# =============================================================================
# メイン
# =============================================================================
def main() -> None:
    print(f"=== 7層多層防御パイプライン 起動 v3.2 ({TIMEFRAME_LABEL}モード) ===")
    print(f"  プロファイル: {PROP_FIRM_PROFILE} | L2 min={L2_MIN_CANDIDATE_SCORE}")
    print(f"  ベイズ: REJECT<{BAYES_REJECT_THRES} ALLOW>={BAYES_ALLOW_THRES} MIN_MATCH={BAYES_MIN_MATCH_SAMPLES} (降格・拒絶専用)")
    if PROP_FIRM_PROFILE == "challenge":
        print(
            f"  リスク: challenge 利益連動 {audit_rm.CHALLENGE_BASE_RISK_PCT_MAX*100:.1f}%"
            f"→{audit_rm.CHALLENGE_BASE_RISK_PCT_MIN*100:.1f}% (+{audit_rm.CHALLENGE_PROFIT_TARGET_PCT:.0f}% 目標)"
        )
    else:
        print(f"  リスク: funded 固定 {audit_rm.FUNDED_BASE_RISK_PCT*100:.1f}%")
    print(f"  日次DDテーパー 0→{DAILY_DD_TAPER_HARD_PCT}% lot×1.0→0.0")
    print(f"  GBPUSD: {GBPUSD_FILE.name}")
    print(f"  EURUSD: {EURUSD_FILE.name}")

    gbp_raw = load_ohlcv(GBPUSD_FILE)
    eur_raw = load_ohlcv(EURUSD_FILE)

    if MODE_H1:
        h1_gbp, h1_eur = gbp_raw, eur_raw
        track_gbp, track_eur = gbp_raw, eur_raw
        bar_minutes = 60
    else:
        h1_gbp = resample_to_h1(gbp_raw)
        h1_eur = resample_to_h1(eur_raw)
        track_gbp, track_eur = gbp_raw, eur_raw
        bar_minutes = 5

    records = run_pipeline(gbp_raw, eur_raw, h1_gbp, h1_eur, track_gbp, bar_minutes)

    if records:
        append_csv_log(records)
        print(f"\n[L6] CSVログ出力完了: {CSV_LOG_PATH}")
        print(f"[L6] JSONアーカイブ: {JSON_DIR} ({len(records)} 件)")
    else:
        print("\nセットアップイベントは検出されませんでした。")

    print_l7_report(records)


if __name__ == "__main__":
    main()
