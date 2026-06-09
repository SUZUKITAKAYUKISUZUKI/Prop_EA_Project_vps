"""
audit/wyckoff_bayes_gate.py — Wyckoff Reversal (WR) L3.5 ベイズ意思決定

プロトタイプ段階では全件 ALLOW（特徴量収集優先）。
WYCKOFF_BAYES_STRICT=1 で CSPA 同等の 3 段階閾値を有効化する。
"""

from __future__ import annotations

import os
from typing import Literal

WYCKOFF_BAYES_ALLOW_THRES = float(os.getenv("WYCKOFF_BAYES_ALLOW_THRES", "0.65"))
WYCKOFF_BAYES_CAUTION_THRES = float(os.getenv("WYCKOFF_BAYES_CAUTION_THRES", "0.50"))

WyckoffBayesDecision = Literal["ALLOW", "CAUTION", "REJECT"]


def is_wyckoff_l4_bypass() -> bool:
    """Wyckoff L4 Gemini バイパス（1=プロトタイプ / 0=本番パイプライン）。"""
    return os.getenv("WYCKOFF_L4_BYPASS", "1").strip().lower() in ("1", "true", "yes", "on")


def is_wyckoff_production_pipeline() -> bool:
    """7層フルパイプライン（L3.5 + L4 Gemini）を有効にする本番モード。"""
    return not is_wyckoff_l4_bypass()


def is_wyckoff_bayes_strict_mode() -> bool:
    return os.getenv("WYCKOFF_BAYES_STRICT", "0").strip().lower() in ("1", "true", "yes", "on")


def resolve_wyckoff_bayes_decision(bayes_probability: float) -> WyckoffBayesDecision:
    """Wyckoff L3.5: 初期は全件 ALLOW。strict 時のみ閾値判定。"""
    if not is_wyckoff_bayes_strict_mode():
        return "ALLOW"
    p = max(0.0, min(1.0, float(bayes_probability)))
    if p >= WYCKOFF_BAYES_ALLOW_THRES:
        return "ALLOW"
    if p >= WYCKOFF_BAYES_CAUTION_THRES:
        return "CAUTION"
    return "REJECT"


def check_wyckoff_bayes_hard_reject(bayes_probability: float) -> bool:
    return resolve_wyckoff_bayes_decision(bayes_probability) == "REJECT"
