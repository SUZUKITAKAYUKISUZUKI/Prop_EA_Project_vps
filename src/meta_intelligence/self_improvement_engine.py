"""Self-improvement notes for Portfolio OS modules."""
from __future__ import annotations

from typing import Any


class SelfImprovementEngine:
    def evaluate(
        self,
        *,
        trust_scores: dict[str, dict[str, Any]],
        drift: dict[str, Any],
        cace_v17_report: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        notes: list[dict[str, Any]] = []
        v17 = cace_v17_report or {}

        for alert in drift.get("drift_alerts") or []:
            notes.append(
                {
                    "module": alert.get("module"),
                    "issue": "trust_drift",
                    "recommendation": f"review_{str(alert.get('module', '')).lower()}_governance_rules",
                }
            )

        rec_accuracy = v17.get("recommendation_accuracy") or {}
        if float(rec_accuracy.get("PDTS") or 100) < 75:
            notes.append(
                {
                    "module": "PDTS",
                    "issue": "forecast_error_high",
                    "recommendation": "increase_monte_carlo_trials",
                }
            )

        for module, data in trust_scores.items():
            score = float(data.get("trust_score") or 0)
            components = data.get("components") or {}
            if score < 55:
                notes.append(
                    {
                        "module": module,
                        "issue": "low_trust_score",
                        "recommendation": f"audit_{module.lower()}_inputs_and_reduce_weight",
                    }
                )
            if float(components.get("calibration") or 100) < 65:
                notes.append(
                    {
                        "module": module,
                        "issue": "calibration_gap",
                        "recommendation": f"recalibrate_{module.lower()}_confidence_model",
                    }
                )

        if v17.get("overconfident"):
            notes.append(
                {
                    "module": "CACE",
                    "issue": "portfolio_overconfidence",
                    "recommendation": "reduce_confidence_for_profile_switch_decisions",
                }
            )

        if not notes:
            notes.append(
                {
                    "module": "PORTFOLIO_OS",
                    "issue": "none",
                    "recommendation": "continue_monitoring_module_trust_scores",
                }
            )
        return notes
