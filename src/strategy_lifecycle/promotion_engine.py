"""Strategy promotion rules for SLM."""
from __future__ import annotations

from typing import Any


class PromotionEngine:
    def incubation_to_candidate(self, metrics: dict[str, Any]) -> tuple[bool, str]:
        if metrics.get("oos_pf", 0.0) <= 1.2:
            return False, "oos_pf_below_1.2"
        if metrics.get("trade_count", 0) <= 300:
            return False, "trades_below_300"
        if metrics.get("oos_months", 0.0) < 6.0:
            return False, "oos_period_below_6m"
        return True, "qualified_for_candidate"

    def candidate_to_production(
        self,
        metrics: dict[str, Any],
        *,
        pdts_score: float | None = None,
    ) -> tuple[bool, str]:
        if metrics.get("pass_rate", 0.0) <= 95.0:
            return False, "pass_rate_below_95"
        if metrics.get("pf", 0.0) <= 1.4:
            return False, "pf_below_1.4"
        if metrics.get("score", 0.0) <= 75.0:
            return False, "score_below_75"
        if metrics.get("oos_pf", 0.0) <= 1.3:
            return False, "oos_pf_below_1.3"
        if pdts_score is not None and pdts_score <= 80.0:
            return False, "pdts_score_below_80"
        if metrics.get("portfolio_fit_score", 0.0) <= 60.0:
            return False, "portfolio_fit_below_60"
        return True, "qualified_for_production"

    def production_to_core(self, metrics: dict[str, Any]) -> tuple[bool, str]:
        if metrics.get("oos_months", 0.0) < 36.0:
            return False, "oos_below_36m"
        if metrics.get("pass_rate", 0.0) < 99.0:
            return False, "pass_rate_below_99"
        if metrics.get("portfolio_fit_score", 0.0) < 80.0:
            return False, "portfolio_fit_below_80"
        if metrics.get("score", 0.0) < 85.0:
            return False, "score_below_85"
        if int(metrics.get("recovery_events") or 0) > 1:
            return False, "recovery_events_above_1"
        return True, "qualified_for_core"

    def recovery_to_production(self, metrics: dict[str, Any]) -> tuple[bool, str]:
        if metrics.get("score", 0.0) <= 75.0:
            return False, "score_below_75"
        if metrics.get("pf", 0.0) <= 1.2:
            return False, "pf_not_recovered"
        if metrics.get("recovery_contribution", 100.0) > 15.0:
            return False, "recovery_not_resolved"
        if metrics.get("portfolio_fit_score", 0.0) <= 60.0:
            return False, "portfolio_fit_below_60"
        return True, "reinstated_to_production"

    def next_stage(
        self,
        current: str,
        metrics: dict[str, Any],
        *,
        pdts_score: float | None,
    ) -> tuple[str | None, str, bool]:
        if current == "INCUBATION":
            ok, reason = self.incubation_to_candidate(metrics)
            return ("CANDIDATE", reason, ok) if ok else (None, reason, False)
        if current == "CANDIDATE":
            ok, reason = self.candidate_to_production(metrics, pdts_score=pdts_score)
            return ("PRODUCTION", reason, ok) if ok else (None, reason, False)
        if current == "PRODUCTION":
            ok, reason = self.production_to_core(metrics)
            return ("CORE", reason, ok) if ok else (None, reason, False)
        if current == "RECOVERY":
            ok, reason = self.recovery_to_production(metrics)
            return ("PRODUCTION", reason, ok) if ok else (None, reason, False)
        return (None, "no_promotion_path", False)
