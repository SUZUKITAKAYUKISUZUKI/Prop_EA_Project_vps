"""Lightweight allocation simulation for PAAE (PFOO-compatible metrics)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from prae.metrics import apply_allocation_weights, max_drawdown_r, profit_factor, sharpe_r
from src.adaptive_allocation.allocation_policy import STATE_OBJECTIVES


class AllocationSimulator:
    """Evaluate candidate weights against historical trades."""

    def evaluate(
        self,
        trades: pd.DataFrame,
        weights: dict[str, float],
        *,
        account_state: str = "funded",
    ) -> dict[str, Any]:
        if trades.empty or "R" not in trades.columns:
            return {
                "pass_rate": 0.0,
                "pass_days": 999.0,
                "max_dd": 0.0,
                "total_r": 0.0,
                "pf": 0.0,
                "sharpe": 0.0,
                "objective_score": 0.0,
            }

        work = trades.copy()
        if "strategy" not in work.columns:
            work["strategy"] = work.get("strategy_code", work.get("setup_type"))
        weighted = apply_allocation_weights(work, weights)
        active = weighted[weighted["allocation_weight"] > 0]
        if active.empty:
            return self.evaluate(trades, {k: 1.0 / max(len(weights), 1) for k in weights}, account_state=account_state)

        r = active["R"].astype(float)
        total_r = float(r.sum())
        dd = max_drawdown_r(r)
        pf_val = profit_factor(r)
        pf = pf_val if pf_val != float("inf") else 3.0
        sharpe = sharpe_r(r)
        wins = (r > 0).sum()
        pass_rate = float(wins / len(r) * 100.0) if len(r) else 0.0
        pass_days = max(7.0, 250.0 / max(len(r), 1) * 30.0)

        state = account_state.lower()
        if state == "challenge":
            objective = pass_rate / 100.0 * 0.6 + (30.0 / pass_days) * 0.4
        elif state == "recovery":
            objective = -dd
        else:
            objective = total_r / 100.0 * 0.5 + pf / 3.0 * 0.3 + sharpe / 2.0 * 0.2

        return {
            "pass_rate": round(pass_rate, 2),
            "pass_days": round(pass_days, 1),
            "max_dd": round(dd, 2),
            "total_r": round(total_r, 2),
            "pf": round(pf, 4),
            "sharpe": round(sharpe, 4),
            "objective_score": round(objective, 4),
            "priorities": list(STATE_OBJECTIVES.get(state, STATE_OBJECTIVES["funded"])["priorities"]),
        }

    def compare_candidates(
        self,
        trades: pd.DataFrame,
        candidates: dict[str, dict[str, float]],
        *,
        account_state: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for label, weights in candidates.items():
            metrics = self.evaluate(trades, weights, account_state=account_state)
            rows.append({"label": label, "weights": weights, **metrics})
        return sorted(rows, key=lambda r: r["objective_score"], reverse=True)
