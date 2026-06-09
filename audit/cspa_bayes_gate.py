"""
audit/cspa_bayes_gate.py — CSPA 専用 L3.5 多次元動的ベイズゲート

CSPA は汎用 ``bayes_probability`` 閾値ゲートではなく、
``audit.cspa_bayes_engine.CSPABayesEngine`` による 3-Tier 特徴量推論を使用する。

    Tier 1: reaccel_follow_through × reacceleration_score decile マトリクス → REJECT
    Tier 2: session × ATR レジーム → ベース lot/tp
    Tier 3: rhythm / market_breath / breakout_velocity → 動的補正

Pure BT（``CSPA_PURE_BT=1``）では本ゲートをスキップし、全件執行 + 特徴量収集のみ行う。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal, Mapping, TypedDict

from audit.cspa_bayes_engine import CSPABayesEngine

CspaBayesDecision = Literal["ALLOW", "REJECT"]

_ENGINE: CSPABayesEngine | None = None


class CspaBayesGateResult(TypedDict):
    """CSPA ベイズゲート評価結果。"""

    decision: CspaBayesDecision
    reason: str
    lot_multiplier: float
    tp_multiplier: float
    bayes_probability: float
    segment_avg_r: float


def _default_config_path() -> Path | None:
    raw = os.getenv("CSPA_BAYES_ENGINE_CONFIG", "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_file() else None


def get_cspa_bayes_engine() -> CSPABayesEngine:
    """モジュール共有の CSPABayesEngine シングルトン。"""
    global _ENGINE
    if _ENGINE is None:
        cfg_path = _default_config_path()
        _ENGINE = (
            CSPABayesEngine.from_json_path(cfg_path)
            if cfg_path is not None
            else CSPABayesEngine()
        )
    return _ENGINE


def reset_cspa_bayes_engine(config: dict[str, Any] | None = None) -> CSPABayesEngine:
    """テスト用: エンジンを再構築する。"""
    global _ENGINE
    _ENGINE = CSPABayesEngine(config)
    return _ENGINE


def evaluate_cspa_bayes_gate(features: Mapping[str, Any]) -> CspaBayesGateResult:
    """
    CSPA 特徴量辞書から執行判定と lot/tp 倍率を返す。

    ``bayes_probability`` には Tier 1 セグメント勝率（ログ互換用）を格納する。
    """
    engine = get_cspa_bayes_engine()
    feat = dict(features)
    rf = float(feat.get("reaccel_follow_through", 0.0))
    ra = float(feat.get("reacceleration_score", 0.0))
    segment_wr, segment_avg_r, _, _ = engine.tier1_segment_stats(rf, ra)

    result = engine.evaluate_trade(feat)
    return {
        "decision": result["decision"],
        "reason": result["reason"],
        "lot_multiplier": result["lot_multiplier"],
        "tp_multiplier": result["tp_multiplier"],
        "bayes_probability": round(segment_wr, 4),
        "segment_avg_r": round(segment_avg_r, 4),
    }


def check_cspa_bayes_hard_reject(features: Mapping[str, Any]) -> bool:
    """CSPA L3.5: 特徴量ベースのハード拒否判定。"""
    return evaluate_cspa_bayes_gate(features)["decision"] == "REJECT"


def cspa_bayes_reject_decision_source(features: Mapping[str, Any]) -> str:
    """監査 CSV 用 decision_source（REJECT 時）。"""
    if check_cspa_bayes_hard_reject(features):
        return "REJECT_BY_BAYES"
    return "ALLOW"
