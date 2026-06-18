"""Allocation drift detection and reporting for PAAE."""
from __future__ import annotations

from typing import Any

from src.adaptive_allocation.allocation_constraints import AllocationConstraints


class AllocationReport:
    def __init__(self, constraints: AllocationConstraints | None = None) -> None:
        self._constraints = constraints or AllocationConstraints()

    def detect_drift(
        self,
        current: dict[str, float],
        recommended: dict[str, float],
    ) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []
        threshold = self._constraints.drift_threshold
        keys = set(current) | set(recommended)
        for key in sorted(keys):
            cur = float(current.get(key, 0.0))
            rec = float(recommended.get(key, 0.0))
            drift = round(abs(rec - cur), 4)
            if drift >= threshold:
                alerts.append(
                    {
                        "strategy": key,
                        "current_pct": round(cur * 100.0, 1),
                        "recommended_pct": round(rec * 100.0, 1),
                        "drift_pct": round(drift * 100.0, 1),
                        "direction": "increase" if rec > cur else "decrease",
                    }
                )
        return sorted(alerts, key=lambda a: a["drift_pct"], reverse=True)

    def weight_diff_table(
        self,
        current: dict[str, float],
        recommended: dict[str, float],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key in sorted(set(current) | set(recommended)):
            cur = float(current.get(key, 0.0))
            rec = float(recommended.get(key, 0.0))
            rows.append(
                {
                    "strategy": key,
                    "current_pct": round(cur * 100.0, 1),
                    "recommended_pct": round(rec * 100.0, 1),
                    "difference_pct": round((rec - cur) * 100.0, 1),
                }
            )
        return sorted(rows, key=lambda r: abs(r["difference_pct"]), reverse=True)

    def build_report(
        self,
        *,
        current_weights: dict[str, float],
        recommended_weights: dict[str, float],
        quality_scores: dict[str, float],
        strategy_risk: list[dict[str, Any]],
        health_report: dict[str, Any],
        drift_alerts: list[dict[str, Any]],
        last_rebalance: str | None,
        simulation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "current_weights": {k: round(v * 100.0, 1) for k, v in current_weights.items()},
            "recommended_weights": {k: round(v * 100.0, 1) for k, v in recommended_weights.items()},
            "quality_scores": quality_scores,
            "risk_ranking": [
                {"strategy": r["strategy"], "risk_score": r.get("risk_score")}
                for r in sorted(strategy_risk, key=lambda x: x.get("risk_score", 0), reverse=True)
            ],
            "quality_ranking": sorted(
                [{"strategy": k, "quality_score": v} for k, v in quality_scores.items()],
                key=lambda x: x["quality_score"],
                reverse=True,
            ),
            "drift_alerts": drift_alerts,
            "weight_diff": self.weight_diff_table(current_weights, recommended_weights),
            "health_impact": {
                "health_score": health_report.get("health_score"),
                "health_status": health_report.get("health_status"),
                "highest_risk_strategy": health_report.get("highest_risk_strategy"),
            },
            "last_rebalance": last_rebalance,
            "simulation": simulation or {},
        }
