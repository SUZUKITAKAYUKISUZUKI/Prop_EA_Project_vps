"""Daily briefing for RC2 live operations."""
from __future__ import annotations

from datetime import date
from typing import Any


class DailyBriefingEngine:
    def evaluate(
        self,
        *,
        ops_context: dict[str, Any],
        operational_score: float,
        live_readiness: float,
        user_action_load: int,
        required_actions: list[str],
        briefing_date: str | None = None,
    ) -> dict[str, Any]:
        ai_cio = ops_context.get("ai_cio_report") or {}
        cil = ops_context.get("cil_report") or {}
        production = ops_context.get("production_report") or {}

        system_health = float(
            production.get("system_health_score")
            or (production.get("end_to_end") or {}).get("end_to_end_score")
            or operational_score
        )

        return {
            "date": briefing_date or date.today().isoformat(),
            "profile_id": ops_context.get("profile_id"),
            "cio_opinion": ai_cio.get("cio_opinion"),
            "portfolio_state": list(ai_cio.get("portfolio_state") or cil.get("investment_state") or []),
            "top_opportunity": ai_cio.get("top_opportunity"),
            "top_risk": ai_cio.get("top_risk"),
            "required_actions": required_actions,
            "user_action_load": user_action_load,
            "system_health": round(system_health, 2),
            "readiness": round(live_readiness, 2),
            "operational_score": round(operational_score, 2),
            "cio_score": ai_cio.get("cio_score"),
            "recommended_profile": ai_cio.get("recommended_profile"),
        }
