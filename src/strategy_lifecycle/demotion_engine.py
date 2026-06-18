"""Strategy demotion rules for SLM."""
from __future__ import annotations

from typing import Any


class DemotionEngine:
    def production_to_recovery(self, metrics: dict[str, Any]) -> tuple[bool, str]:
        if metrics.get("score", 100.0) < 60.0:
            return True, "score_below_60"
        if metrics.get("dd_contribution", 0.0) > 20.0:
            return True, "dd_contribution_above_20"
        if metrics.get("recovery_events", 0) >= 2:
            return True, "recovery_events_increased"
        if metrics.get("risk_score", 0.0) >= 70.0:
            return True, "prae_risk_warning"
        if metrics.get("portfolio_fit_score", 100.0) < 40.0:
            return True, "portfolio_fit_below_40"
        return False, "production_stable"

    def core_to_recovery(self, metrics: dict[str, Any]) -> tuple[bool, str]:
        if metrics.get("portfolio_fit_score", 100.0) < 50.0:
            return True, "portfolio_fit_below_50"
        if metrics.get("score", 100.0) < 60.0:
            return True, "score_below_60"
        return False, "core_stable"

    def next_stage(self, current: str, metrics: dict[str, Any]) -> tuple[str | None, str, bool]:
        if current == "CORE":
            ok, reason = self.core_to_recovery(metrics)
            return ("RECOVERY", reason, ok) if ok else (None, reason, False)
        if current == "PRODUCTION":
            ok, reason = self.production_to_recovery(metrics)
            return ("RECOVERY", reason, ok) if ok else (None, reason, False)
        return (None, "no_demotion_path", False)
