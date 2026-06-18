"""Phase 5.6 — Risk Budget Optimizer (strategy-level challenge risk allocation)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from core.pass_probability import (
    AccountSnapshot,
    ChallengeState,
    estimate_expected_pass_days,
    estimate_pass_probability,
)
from core.prop_profiles import PropProfile
from core.utility_engine import ObjectiveMode, compute_utility
from prae.loaders import STRATEGY_COLUMN
from prae.metrics import apply_allocation_weights, profit_factor, sharpe_r


@dataclass(frozen=True)
class RiskBudgetResult:
    weights: dict[str, float]
    score: float
    pass_probability: float
    expected_pass_days: float
    utility: float
    top_candidates: pd.DataFrame


def _normalize_weights(raw: dict[str, float], strategies: tuple[str, ...]) -> dict[str, float]:
    vec = np.array([max(0.0, float(raw.get(s, 0.0))) for s in strategies], dtype=np.float64)
    total = vec.sum()
    if total <= 0:
        vec = np.ones(len(strategies), dtype=np.float64) / max(len(strategies), 1)
    else:
        vec = vec / total
    return {s: round(float(vec[i]), 4) for i, s in enumerate(strategies)}


def _random_weight_vector(
    n: int,
    rng: np.random.Generator,
    *,
    min_w: float,
    max_w: float,
) -> np.ndarray:
    for _ in range(64):
        w = rng.random(n) * (max_w - min_w) + min_w
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        if w.max() <= max_w + 1e-9:
            return w
    return np.ones(n) / n


def _score_allocation(
    trades: pd.DataFrame,
    strategies: tuple[str, ...],
    weights: dict[str, float],
    *,
    profile: PropProfile,
    account: AccountSnapshot,
    challenge: ChallengeState,
    mode: ObjectiveMode,
    config: dict[str, Any],
    prae_context: dict[str, Any] | None,
    mc_trials: int,
    horizon_trades: int | None = None,
    fast: bool = False,
    account_state: str | None = None,
) -> dict[str, float]:
    weighted = apply_allocation_weights(trades, weights)
    active = weighted[weighted["allocation_weight"] > 0]
    if active.empty:
        return {"score": -1e9, "pass_probability": 0.0, "expected_pass_days": 999.0, "utility": 0.0}

    # Penalize high risk-contribution strategies when DD headroom is low
    penalty = 0.0
    if prae_context:
        risk_contrib = prae_context.get("risk_contribution") or {}
        dd_headroom = max(0.0, profile.total_dd_limit - challenge.total_dd_used_percent)
        if dd_headroom < profile.total_dd_limit * 0.5:
            for s, w in weights.items():
                penalty += w * float(risk_contrib.get(s, 0.0)) / 100.0 * 0.15

    pass_res = estimate_pass_probability(
        active,
        profile=profile,
        account=account,
        challenge=challenge,
        trials=mc_trials,
        seed=int((config.get("monte_carlo") or {}).get("seed", 42)),
        horizon_trades=horizon_trades,
        fast=fast,
    )
    days_res = estimate_expected_pass_days(
        active,
        profile=profile,
        account=account,
        challenge=challenge,
        trials=max(10, mc_trials // 3),
        horizon_trades=horizon_trades,
        fast=fast,
    )
    util = compute_utility(
        pass_probability=pass_res.pass_probability,
        expected_pass_days=days_res.expected_pass_days,
        total_dd_used_pct=challenge.total_dd_used_percent,
        total_dd_limit=profile.total_dd_limit,
        mode=mode,
        config=config,
    )
    total_r = float(active["R"].sum()) if "R" in active.columns else float(active["profit_r"].sum())
    pf_val = profit_factor(active["R"]) if "R" in active.columns else 0.0
    sharpe_val = sharpe_r(active["R"]) if "R" in active.columns else 0.0

    if account_state:
        from src.objective_optimizer.objective_profiles import (
            compute_objective_score,
            metrics_from_pfoo_context,
        )

        metrics = metrics_from_pfoo_context(
            pass_probability=pass_res.pass_probability,
            expected_pass_days=days_res.expected_pass_days,
            total_dd_used_pct=challenge.total_dd_used_percent,
            total_dd_limit=profile.total_dd_limit,
            utility=util.utility,
            pf=pf_val if pf_val != float("inf") else 3.0,
            sharpe=sharpe_val,
            total_r=total_r,
        )
        score = compute_objective_score(account_state, metrics) - penalty
    else:
        score = util.utility - penalty
    return {
        "score": round(score, 4),
        "pass_probability": pass_res.pass_probability,
        "expected_pass_days": days_res.expected_pass_days,
        "utility": util.utility,
    }


def optimize_risk_budget(
    trades: pd.DataFrame,
    strategies: tuple[str, ...],
    *,
    profile: PropProfile,
    account: AccountSnapshot,
    challenge: ChallengeState,
    mode: ObjectiveMode = "BALANCED",
    config: dict[str, Any] | None = None,
    prae_context: dict[str, Any] | None = None,
    base_weights: dict[str, float] | None = None,
    account_state: str | None = None,
) -> RiskBudgetResult:
    """
    Allocate challenge risk budget across strategies using prop-firm utility.

    Uses PRAE risk contribution + correlation context as penalties/bonuses.
    """
    from audit.risk_manager import is_portfolio_allocation_enabled

    cfg = config or {}
    rb_cfg = cfg.get("risk_budget") or {}
    min_w = float(rb_cfg.get("min_weight", 0.0))
    max_w = float(rb_cfg.get("max_weight", 0.50))
    trials = int(rb_cfg.get("random_trials", 1500))
    search_mc = int(rb_cfg.get("search_mc_trials", 40))
    search_horizon = int(rb_cfg.get("search_horizon_trades", 250))
    top_n = int(rb_cfg.get("top_n", 10))
    seed = int((cfg.get("monte_carlo") or {}).get("seed", 42))
    rng = np.random.default_rng(seed)
    n = len(strategies)

    candidates: list[dict[str, Any]] = []

    def _eval(w: dict[str, float]) -> dict[str, float]:
        return _score_allocation(
            trades,
            strategies,
            w,
            profile=profile,
            account=account,
            challenge=challenge,
            mode=mode,
            config=cfg,
            prae_context=prae_context,
            mc_trials=search_mc,
            horizon_trades=search_horizon,
            fast=True,
            account_state=account_state,
        )

    equal = _normalize_weights(base_weights or {s: 1.0 for s in strategies}, strategies)
    equal_score = _eval(equal)

    if not is_portfolio_allocation_enabled():
        top_rows = [
            {
                "Rank": 1,
                "PassRate": round(equal_score["pass_probability"], 2),
                "ExpPassDays": round(equal_score["expected_pass_days"], 1),
                "Utility": round(equal_score["utility"], 4),
                "Score": round(equal_score["score"], 4),
                "Allocation": ", ".join(f"{k}:{v:.3f}" for k, v in equal.items()),
            }
        ]
        return RiskBudgetResult(
            weights=equal,
            score=float(equal_score["score"]),
            pass_probability=float(equal_score["pass_probability"]),
            expected_pass_days=float(equal_score["expected_pass_days"]),
            utility=float(equal_score["utility"]),
            top_candidates=pd.DataFrame(top_rows),
        )

    candidates.append({"weights": equal, **equal_score})

    for _ in range(trials):
        vec = _random_weight_vector(n, rng, min_w=min_w, max_w=max_w)
        w = {strategies[i]: round(float(vec[i]), 4) for i in range(n)}
        candidates.append({"weights": w, **_eval(w)})

    scored = pd.DataFrame(
        [
            {
                "score": c["score"],
                "pass_probability": c["pass_probability"],
                "expected_pass_days": c["expected_pass_days"],
                "utility": c["utility"],
                "weights_json": json.dumps(c["weights"], sort_keys=True),
            }
            for c in candidates
        ]
    ).sort_values("score", ascending=False).reset_index(drop=True)

    best = json.loads(scored.iloc[0]["weights_json"])
    top_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(scored.head(top_n).itertuples(index=False), start=1):
        w = json.loads(row.weights_json)
        alloc = ", ".join(f"{k}:{v:.3f}" for k, v in w.items())
        top_rows.append(
            {
                "Rank": rank,
                "PassRate": round(row.pass_probability, 2),
                "ExpPassDays": round(row.expected_pass_days, 1),
                "Utility": round(row.utility, 4),
                "Score": round(row.score, 4),
                "Allocation": alloc,
            }
        )

    best_row = scored.iloc[0]
    return RiskBudgetResult(
        weights=best,
        score=float(best_row["score"]),
        pass_probability=float(best_row["pass_probability"]),
        expected_pass_days=float(best_row["expected_pass_days"]),
        utility=float(best_row["utility"]),
        top_candidates=pd.DataFrame(top_rows),
    )
