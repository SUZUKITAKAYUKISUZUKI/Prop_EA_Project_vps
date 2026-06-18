"""Strategy lifecycle forecast for AGE v3."""
from __future__ import annotations

from typing import Any

from src.ai_governor.governor_context import GovernorContext
from src.ai_governor_v3.forecast_config import ForecastConfig, DEFAULT_CONFIG


class LifecycleForecaster:
    def __init__(self, config: ForecastConfig | None = None) -> None:
        self._config = config or DEFAULT_CONFIG

    def forecast(self, context: GovernorContext) -> dict[str, Any]:
        strategies: dict[str, Any] = {}
        demotion_candidates: list[str] = []
        retirement_candidates: list[str] = []
        promotion_candidates: list[str] = []

        for sid, score in context.strategy_scores.items():
            fit = float(context.strategy_fit_scores.get(sid) or 50.0)
            stage = str(context.strategy_stages.get(sid) or "INCUBATION")
            decay = self._decay_rate(context, score, fit)
            projections: dict[str, Any] = {
                "current": {"stage": stage, "score": round(score, 1), "portfolio_fit": round(fit, 1)},
            }
            proj_score = score
            proj_fit = fit
            for days in self._config.forecast_days:
                proj_score = max(0.0, proj_score - decay["score"] * days)
                proj_fit = max(0.0, proj_fit - decay["fit"] * days)
                proj_stage = self._project_stage(stage, proj_score, proj_fit)
                projections[f"{days}d"] = {
                    "stage": proj_stage,
                    "score": round(proj_score, 1),
                    "portfolio_fit": round(proj_fit, 1),
                }

            flags: list[str] = []
            if any(projections[f"{d}d"]["portfolio_fit"] < 40 for d in self._config.forecast_days):
                flags.append("DEMOTION_CANDIDATE")
                demotion_candidates.append(sid)
            if any(projections[f"{d}d"]["portfolio_fit"] < 20 for d in self._config.forecast_days):
                flags.append("RETIREMENT_CANDIDATE")
                retirement_candidates.append(sid)
            if stage in {"CANDIDATE", "PRODUCTION"} and all(
                projections[f"{d}d"]["portfolio_fit"] >= 70 for d in self._config.forecast_days
            ):
                flags.append("PROMOTION_CANDIDATE")
                promotion_candidates.append(sid)

            strategies[sid] = {"projections": projections, "flags": flags}

        return {
            "strategies": strategies,
            "demotion_candidates": demotion_candidates,
            "retirement_candidates": retirement_candidates,
            "promotion_candidates": promotion_candidates,
        }

    def _decay_rate(self, context: GovernorContext, score: float, fit: float) -> dict[str, float]:
        risk_drag = context.risk_score / 1000.0
        recovery_drag = context.recovery_events * 0.02
        low_fit_drag = max(0.0, (60.0 - fit) / 3000.0)
        return {
            "score": round(0.03 + risk_drag + recovery_drag, 4),
            "fit": round(0.04 + risk_drag + low_fit_drag + recovery_drag, 4),
        }

    def _project_stage(self, current: str, score: float, fit: float) -> str:
        stage = current.upper()
        if fit < 20 or score < 40:
            return "RETIRED"
        if fit < 40 or score < 55:
            return "RECOVERY"
        if fit >= 80 and score >= 85 and stage == "PRODUCTION":
            return "CORE"
        if fit >= 60 and score >= 75 and stage in {"CANDIDATE", "INCUBATION"}:
            return "PRODUCTION"
        return stage
