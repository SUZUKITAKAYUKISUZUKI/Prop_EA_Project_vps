"""PDTS-backed scenario projections for AGE v3."""
from __future__ import annotations

from typing import Any

from src.ai_governor.governor_context import GovernorContext


class ScenarioProjectionEngine:
    def project(self, context: GovernorContext) -> dict[str, Any]:
        pdts = context.pdts
        cmp = pdts.get("scenario_comparison") or {}
        current = cmp.get("current") or {}
        recommended = cmp.get("recommended") or {}
        gap = float(recommended.get("score") or 0) - float(current.get("score") or 0)
        fit_gain = float(recommended.get("portfolio_fit_gain") or 0)

        projections = []
        for days in (30, 60, 90):
            decay = 1.0 - (days / 180.0) * 0.3
            projections.append(
                {
                    "horizon_days": days,
                    "projected_score_delta": round(gap * decay, 2),
                    "projected_fit_gain": round(fit_gain * decay, 2),
                    "pass_rate_delta": round(
                        (float(recommended.get("pass_rate") or 0) - float(current.get("pass_rate") or 0)) * decay,
                        2,
                    ),
                }
            )

        return {
            "baseline_score": current.get("score"),
            "recommended_score": recommended.get("score"),
            "score_improvement": round(gap, 2),
            "portfolio_fit_gain": fit_gain,
            "projections": projections,
            "scenario_results": context.scenario_results[:5],
        }
