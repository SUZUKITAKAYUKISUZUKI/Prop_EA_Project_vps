"""Compare strategic futures for AGE v4."""
from __future__ import annotations

from typing import Any

from src.ai_governor_v4.future_branch import FutureBranch


class ScenarioComparator:
    def compare(
        self,
        branches: list[FutureBranch],
        *,
        baseline: FutureBranch | None = None,
    ) -> dict[str, Any]:
        baseline = baseline or next((b for b in branches if b.action_type == "DO_NOTHING"), None)
        baseline_score = float(baseline.strategic_score) if baseline else 0.0

        rows: list[dict[str, Any]] = []
        for branch in branches:
            delta = round(branch.strategic_score - baseline_score, 1)
            rows.append(
                {
                    "branch_id": branch.branch_id,
                    "action_label": branch.action_label,
                    "strategic_score": branch.strategic_score,
                    "delta_vs_baseline": delta,
                    "rank_category": branch.rank_category,
                    "expected_benefit": branch.expected_benefit,
                    "expected_risk": branch.expected_risk,
                    "confidence": branch.confidence,
                }
            )

        rows.sort(key=lambda r: float(r.get("strategic_score") or 0), reverse=True)
        best = rows[0] if rows else {}
        return {
            "baseline_branch_id": baseline.branch_id if baseline else None,
            "baseline_score": baseline_score,
            "comparison": rows,
            "best_branch_id": best.get("branch_id"),
            "best_action": best.get("action_label"),
            "best_score": best.get("strategic_score"),
            "score_spread": round(
                float(rows[0].get("strategic_score") or 0) - float(rows[-1].get("strategic_score") or 0),
                1,
            )
            if len(rows) >= 2
            else 0.0,
        }

    def horizon_comparison(
        self,
        branches: list[FutureBranch],
        horizon: str,
    ) -> list[dict[str, Any]]:
        rows = []
        for branch in branches:
            metrics = branch.metrics_by_horizon.get(horizon) or {}
            rows.append(
                {
                    "branch_id": branch.branch_id,
                    "action_label": branch.action_label,
                    "health_score": metrics.get("health_score"),
                    "expected_dd": metrics.get("expected_dd"),
                    "expected_pf": metrics.get("expected_pf"),
                    "pass_probability": metrics.get("pass_probability"),
                    "recovery_probability": metrics.get("recovery_probability"),
                    "risk_budget_remaining": metrics.get("risk_budget_remaining"),
                }
            )
        return rows
