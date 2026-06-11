"""
DiNapoli (DN) — rule-based EV rank V1.

Explainable weighted scoring on four entry features (no ML).
Logging only: no reject, no lot sizing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

MODEL_VERSION = "dn_bayes_ev_v1"

W_MOMENTUM = 0.40
W_PULLBACK = 0.25
W_EMA = 0.20
W_ATR = 0.15

# Calibrated on dn_bayes_features_pure_3y.csv (BT train quartiles)
DEFAULT_ATR_Q1 = 0.00055425
DEFAULT_ATR_Q2 = 0.000783
DEFAULT_ATR_Q3 = 0.0010885

EV_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("Bottom20", 0.00, 0.20),
    ("Low", 0.20, 0.50),
    ("Middle", 0.50, 0.80),
    ("Top20", 0.80, 0.95),
    ("Top5", 0.95, 1.001),
)


@dataclass(frozen=True)
class DnEvConfig:
    atr_q1: float = DEFAULT_ATR_Q1
    atr_q2: float = DEFAULT_ATR_Q2
    atr_q3: float = DEFAULT_ATR_Q3


def is_dn_ev_rank_enabled() -> bool:
    return os.getenv("DN_EV_RANK", "0").strip().lower() in ("1", "true", "yes", "on")


def score_momentum(momentum_score: float) -> float:
    m = float(momentum_score)
    if m <= 60.0:
        return 1.00
    if m <= 65.0:
        return 0.75
    if m <= 70.0:
        return 0.50
    return 0.25


def score_pullback(pullback_depth: float) -> float:
    p = float(pullback_depth)
    if p >= 0.56:
        return 1.00
    if p >= 0.52:
        return 0.75
    if p >= 0.46:
        return 0.50
    return 0.25


def score_alignment(ema_alignment_score: float) -> float:
    a = float(ema_alignment_score)
    if a >= 0.50:
        return 1.00
    if a >= 0.45:
        return 0.75
    if a >= 0.40:
        return 0.50
    return 0.25


def score_atr(atr_m15: float, *, cfg: DnEvConfig | None = None) -> float:
    cfg = cfg or DnEvConfig()
    a = float(atr_m15)
    if a >= cfg.atr_q2:
        return 1.00
    if a >= cfg.atr_q1:
        return 0.75
    return 0.25


def ev_bucket_from_rank(ev_rank: float) -> str:
    rank = max(0.0, min(1.0, float(ev_rank)))
    for name, lo, hi in EV_BUCKETS:
        if lo <= rank < hi:
            return name
    return "Top5"


def compute_dn_ev_rank(
    features: Mapping[str, Any],
    *,
    cfg: DnEvConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or DnEvConfig()
    s_m = score_momentum(float(features.get("momentum_score", 0) or 0))
    s_p = score_pullback(float(features.get("pullback_depth", 0) or 0))
    s_e = score_alignment(float(features.get("ema_alignment_score", 0) or 0))
    s_a = score_atr(float(features.get("atr_m15", 0) or 0), cfg=cfg)
    ev_rank = (
        W_MOMENTUM * s_m
        + W_PULLBACK * s_p
        + W_EMA * s_e
        + W_ATR * s_a
    )
    ev_rank = round(max(0.0, min(1.0, ev_rank)), 6)
    bucket = ev_bucket_from_rank(ev_rank)
    return {
        "ev_rank": ev_rank,
        "ev_bucket": bucket,
        "score_momentum": s_m,
        "score_pullback": s_p,
        "score_alignment": s_e,
        "score_atr": s_a,
        "model_version": MODEL_VERSION,
    }


def apply_dn_ev_to_row(row: dict[str, Any], *, cfg: DnEvConfig | None = None) -> dict[str, Any]:
    scored = compute_dn_ev_rank(row, cfg=cfg)
    row["ev_rank"] = scored["ev_rank"]
    row["ev_bucket"] = scored["ev_bucket"]
    return row
