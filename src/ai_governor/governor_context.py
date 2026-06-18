"""Aggregated governance context for AGE."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.objective_optimizer import recommended_objective_label


@dataclass
class GovernorContext:
    """Single object aggregating all upstream Portfolio OS intelligence."""

    profile_id: str
    profile_name: str
    current_profile: str
    current_state: str
    source_state: str

    health_score: float
    health_status: str
    risk_score: float
    risk_level: str
    pass_rate: float
    dd_pct: float
    recovery_events: int
    portfolio_fit: float

    strategy_scores: dict[str, float] = field(default_factory=dict)
    strategy_stages: dict[str, str] = field(default_factory=dict)
    strategy_fit_scores: dict[str, float] = field(default_factory=dict)
    current_allocation: dict[str, float] = field(default_factory=dict)
    recommended_allocation: dict[str, float] = field(default_factory=dict)

    prae_v2: dict[str, Any] = field(default_factory=dict)
    state_analytics: dict[str, Any] = field(default_factory=dict)
    paae: dict[str, Any] = field(default_factory=dict)
    pdts: dict[str, Any] = field(default_factory=dict)
    slm: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    scenario_results: list[dict[str, Any]] = field(default_factory=list)
    objective_label: str = ""
    recommended_objective: str = ""

    @classmethod
    def from_payload(
        cls,
        *,
        profile_ctx: dict[str, Any],
        prae_v2: dict[str, Any],
        state_analytics: dict[str, Any],
        paae: dict[str, Any],
        pdts: dict[str, Any],
        slm: dict[str, Any],
    ) -> GovernorContext:
        profile_id = str(profile_ctx.get("profile_id") or "unknown")
        current_state = str(
            state_analytics.get("current_state")
            or profile_ctx.get("settings", {}).get("account_state")
            or "unknown"
        ).lower()
        health = prae_v2.get("health_report") or {}
        health_score = float(
            state_analytics.get("health_score")
            or health.get("health_score")
            or 50.0
        )
        strategy_risk = prae_v2.get("strategy_risk") or []
        top_risk = float(strategy_risk[0].get("risk_score") or 0.0) if strategy_risk else 0.0
        dd_attr = prae_v2.get("dd_attribution") or {}
        dd_pct = float(dd_attr.get("portfolio_dd_pct") or dd_attr.get("current_dd_pct") or 0.0)
        if dd_pct <= 0.0:
            for row in strategy_risk:
                dd_pct = max(dd_pct, float(row.get("max_dd") or 0.0))

        scenario_cmp = pdts.get("scenario_comparison") or {}
        baseline = scenario_cmp.get("current") or {}
        recommended = scenario_cmp.get("recommended") or {}
        pass_rate = float(recommended.get("pass_rate") or baseline.get("pass_rate") or 0.0)

        slm_strategies = slm.get("strategies") or []
        strategy_scores = {
            str(row.get("strategy")): float(row.get("score") or 0.0)
            for row in slm_strategies
            if row.get("strategy")
        }
        strategy_stages = {
            str(row.get("strategy")): str(row.get("stage") or "INCUBATION")
            for row in slm_strategies
            if row.get("strategy")
        }
        strategy_fit_scores = {
            str(row.get("strategy")): float(row.get("portfolio_fit_score") or 0.0)
            for row in slm_strategies
            if row.get("strategy")
        }
        fits = [v for v in strategy_fit_scores.values() if v > 0]
        portfolio_fit = round(sum(fits) / len(fits), 1) if fits else 0.0

        risk_level = _risk_level(top_risk, dd_pct)
        current_alloc = dict(profile_ctx.get("strategy_allocations") or {})
        recommended_alloc = dict(paae.get("recommended_weights") or current_alloc)

        scenarios = pdts.get("recommendation_ranking") or []
        if not scenarios:
            scenarios = [
                {"scenario": "current", **baseline},
                {"scenario": "recommended", **recommended},
            ]

        return cls(
            profile_id=profile_id,
            profile_name=str(profile_ctx.get("profile_name") or profile_id),
            current_profile=profile_id,
            current_state=current_state,
            source_state=current_state,
            health_score=health_score,
            health_status=str(
                state_analytics.get("health_status") or health.get("health_status") or "FAIR"
            ),
            risk_score=top_risk,
            risk_level=risk_level,
            pass_rate=pass_rate,
            dd_pct=dd_pct,
            recovery_events=int(
                state_analytics.get("recovery_events") or health.get("recovery_events") or 0
            ),
            portfolio_fit=portfolio_fit,
            strategy_scores=strategy_scores,
            strategy_stages=strategy_stages,
            strategy_fit_scores=strategy_fit_scores,
            current_allocation=current_alloc,
            recommended_allocation=recommended_alloc,
            prae_v2=prae_v2,
            state_analytics=state_analytics,
            paae=paae,
            pdts=pdts,
            slm=slm,
            profile=profile_ctx,
            scenario_results=list(scenarios),
            objective_label=recommended_objective_label(current_state),
            recommended_objective=recommended_objective_label(current_state),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile_name": self.profile_name,
            "current_profile": self.current_profile,
            "current_state": self.current_state,
            "health_score": self.health_score,
            "health_status": self.health_status,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "pass_rate": self.pass_rate,
            "dd_pct": self.dd_pct,
            "recovery_events": self.recovery_events,
            "portfolio_fit": self.portfolio_fit,
            "strategy_scores": self.strategy_scores,
            "strategy_stages": self.strategy_stages,
            "strategy_fit_scores": self.strategy_fit_scores,
            "current_allocation": self.current_allocation,
            "recommended_allocation": self.recommended_allocation,
            "objective_label": self.objective_label,
            "recommended_objective": self.recommended_objective,
            "scenario_results": self.scenario_results,
        }


def _risk_level(risk_score: float, dd_pct: float) -> str:
    if dd_pct >= 85.0 or risk_score >= 80.0:
        return "CRITICAL"
    if dd_pct >= 70.0 or risk_score >= 70.0:
        return "HIGH"
    if dd_pct >= 50.0 or risk_score >= 55.0:
        return "ELEVATED"
    if dd_pct >= 30.0 or risk_score >= 40.0:
        return "MODERATE"
    return "LOW"
