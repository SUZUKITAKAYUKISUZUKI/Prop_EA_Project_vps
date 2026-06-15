"""
audit/l2_threshold_manager.py — 戦略別 L2 candidate_score 閾値

LSFC は L2 を緩和（スピード重視）。London Continuation（廃止）はプロファイル既定。
"""

from __future__ import annotations

import os

from strategies.archive.fvg_fill import FVG_L2_MIN_SCORE
from strategies.smrs_production import PRODUCTION_L2_MIN_SCORE, SMRS_L2_MIN_SCORE_ENV

FVG_SETUP_TYPE = "FVG_FILL"
LSFC_SETUP_TYPE = "LONDON_SWEEP_FAILURE_CONTINUATION"
ALS_SETUP_TYPE = "ASIAN_SESSION_LIQUIDITY_SWEEP"
TREF_SETUP_TYPE = "TOKYO_RANGE_EXPANSION_FAILURE"
VEXP_SETUP_TYPE = "VEXP_VOLATILITY_EXPANSION"
DTPA_SETUP_TYPE = "DTPA"
CSPA_SETUP_TYPE = "CSPA"
DBBS_SETUP_TYPE = "DBBS"
VAMR_SETUP_TYPE = "VAMR"
SMRS_SETUP_TYPE = "SMRS"
WYCKOFF_SETUP_TYPE = "WYCKOFF_REVERSAL"
WYCKOFF_SETUP_TYPE_LEGACY = "WYCKOFF_SPRING"
WYCKOFF_SETUP_TYPES = frozenset({WYCKOFF_SETUP_TYPE, WYCKOFF_SETUP_TYPE_LEGACY})
VEXP_L2_MIN_SCORE = int(os.getenv("VEXP_L2_MIN_SCORE", "80"))
WYCKOFF_L2_MIN_SCORE = int(os.getenv("WYCKOFF_L2_MIN_SCORE", "50"))
DTPA_L2_MIN_SCORE = int(os.getenv("DTPA_L2_MIN_SCORE", "70"))
CSPA_L2_MIN_SCORE = int(os.getenv("CSPA_L2_MIN_SCORE", "65"))
DBBS_L2_MIN_SCORE = int(os.getenv("DBBS_L2_MIN_SCORE", "0"))
VAMR_L2_MIN_SCORE = int(os.getenv("VAMR_L2_MIN_SCORE", "0"))
SMRS_L2_MIN_SCORE = int(os.getenv(SMRS_L2_MIN_SCORE_ENV, str(PRODUCTION_L2_MIN_SCORE)))
DTPA_LLM_REJECT_BELOW = int(os.getenv("DTPA_LLM_REJECT_BELOW", "65"))
DTPA_LLM_ALLOW_MIN = int(os.getenv("DTPA_LLM_ALLOW_MIN", "85"))
DTPA_LLM_CAUTION_MULT = float(os.getenv("DTPA_LLM_CAUTION_MULT", "0.5"))
DTPA_LLM_ALLOW_MULT = float(os.getenv("DTPA_LLM_ALLOW_MULT", "1.0"))

L2_SPEED_STRATEGIES = frozenset({LSFC_SETUP_TYPE, ALS_SETUP_TYPE, TREF_SETUP_TYPE})

L2_SPEED_MIN_SCORE = int(os.getenv("L2_SPEED_MIN_SCORE", "10"))


def resolve_l2_min_candidate_score(setup_type: str, profile_default: int = 30) -> int:
    """
    戦略別 L2 足切り閾値を返す。

    | 戦略 | 記号 | 閾値 |
    |------|------|------|
    | LSFC | A | 10（スピード重視） |
    | ALS | B | 10（スピード重視） |
    | London Continuation 他 | — | プロファイル既定（通常 30） |
    | VEXP | E | 80（固定） |
    | DTPA | F | 70（粗い L2 足切り） |
    | CSPA | G | 65（専用 L2 スコア） |
    | DBBS | B | 0（candidate_score 未使用） |
    | VAMR | D | 0（candidate_score 未使用） |
    | SMRS | E | 0（candidate_score 未使用） |
    | WR | H | 50（プロトタイプ） |

CSPA L3.5 ベイズ: ``audit.cspa_bayes_gate.evaluate_cspa_bayes_gate`` — CSPABayesEngine 3-Tier
    """
    if setup_type == VEXP_SETUP_TYPE:
        return VEXP_L2_MIN_SCORE
    if setup_type == DTPA_SETUP_TYPE:
        return DTPA_L2_MIN_SCORE
    if setup_type == CSPA_SETUP_TYPE:
        return CSPA_L2_MIN_SCORE
    if setup_type == DBBS_SETUP_TYPE:
        return DBBS_L2_MIN_SCORE
    if setup_type == VAMR_SETUP_TYPE:
        return VAMR_L2_MIN_SCORE
    if setup_type == SMRS_SETUP_TYPE:
        return SMRS_L2_MIN_SCORE
    if setup_type in WYCKOFF_SETUP_TYPES:
        return WYCKOFF_L2_MIN_SCORE
    if setup_type == FVG_SETUP_TYPE:
        return FVG_L2_MIN_SCORE
    if setup_type in L2_SPEED_STRATEGIES:
        return L2_SPEED_MIN_SCORE
    return profile_default


def verify_l2_rules(setup_type: str, candidate_score: float, profile_default: int = 30) -> tuple[str, list[str]]:
    """
    L2 判定。通過時 ``("ALLOW", [])``、拒否時 ``("REJECT_BY_L2", [tags...])``。
    """
    l2_min = resolve_l2_min_candidate_score(setup_type, profile_default)
    if candidate_score >= l2_min:
        return "ALLOW", []
    return "REJECT_BY_L2", l2_reject_reason_tags(setup_type, candidate_score, l2_min)


def l2_reject_reason_tags(setup_type: str, candidate_score: float, l2_min: int) -> list[str]:
    """L2 拒否時の監査タグ。"""
    if setup_type in L2_SPEED_STRATEGIES or setup_type == FVG_SETUP_TYPE:
        return ["CRITICAL_LOW_SCORE"]
    return [f"L2_SCORE_BELOW_{l2_min}"]


def dtpa_confidence_lot_multiplier(confidence_score: int) -> float:
    """DTPA L4: ≥85 → 1.0x / 65–84 → 0.5x / <65 → 0（REJECT）。"""
    score = max(0, min(100, int(confidence_score)))
    if score >= DTPA_LLM_ALLOW_MIN:
        return DTPA_LLM_ALLOW_MULT
    if score >= DTPA_LLM_REJECT_BELOW:
        return DTPA_LLM_CAUTION_MULT
    return 0.0


def dtpa_llm_decision(confidence_score: int) -> str:
    """DTPA L4 意思決定ラベル（ロット帯と同期）。"""
    score = max(0, min(100, int(confidence_score)))
    if score < DTPA_LLM_REJECT_BELOW:
        return "REJECT_BY_LLM"
    if score >= DTPA_LLM_ALLOW_MIN:
        return "ALLOW"
    return "CAUTION"
