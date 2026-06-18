"""Operational metrics and score for RC2."""
from __future__ import annotations

from typing import Any

from src.live_operations.config import (
    OPERATIONAL_SCORE_WEIGHTS,
    USER_ACTION_LOAD_IDEAL_MAX,
    USER_ACTION_LOAD_WARNING,
)


class OperationalMetricsEngine:
    def evaluate(
        self,
        *,
        ops_context: dict[str, Any],
        user_action_load: int,
        historical_reliability: float,
    ) -> dict[str, Any]:
        production = ops_context.get("production_report") or {}
        orl = ops_context.get("orl_report") or {}
        ai_cio = ops_context.get("ai_cio_report") or {}

        ai_cio_availability = float(
            production.get("ai_cio_availability")
            or (production.get("cio_consistency") or {}).get("ai_cio_availability")
            or (100.0 if ai_cio.get("cio_opinion") else 0.0)
        )
        system_health = float(
            production.get("system_health_score")
            or orl.get("system_health_score")
            or 85.0
        )
        recommendation_stability = float(
            (production.get("recommendation_chain") or {}).get("recommendation_consistency")
            or orl.get("recommendation_consistency")
            or 85.0
        )
        data_integrity = float(
            production.get("database_health")
            or (production.get("validation_results") or {}).get("data_integrity")
            or 90.0
        )
        user_action_score = self._user_action_score(user_action_load)

        operational_score = round(
            OPERATIONAL_SCORE_WEIGHTS["ai_cio_availability"] * ai_cio_availability
            + OPERATIONAL_SCORE_WEIGHTS["system_health"] * system_health
            + OPERATIONAL_SCORE_WEIGHTS["recommendation_stability"] * recommendation_stability
            + OPERATIONAL_SCORE_WEIGHTS["data_integrity"] * data_integrity
            + OPERATIONAL_SCORE_WEIGHTS["user_action_load"] * user_action_score
            + OPERATIONAL_SCORE_WEIGHTS["historical_reliability"] * historical_reliability,
            2,
        )

        return {
            "operational_score": operational_score,
            "user_action_load": user_action_load,
            "user_action_load_status": self._action_load_status(user_action_load),
            "ai_cio_availability": round(ai_cio_availability, 2),
            "system_health": round(system_health, 2),
            "recommendation_stability": round(recommendation_stability, 2),
            "data_integrity": round(data_integrity, 2),
            "historical_reliability": round(historical_reliability, 2),
            "operational_components": {
                "ai_cio_availability": ai_cio_availability,
                "system_health": system_health,
                "recommendation_stability": recommendation_stability,
                "data_integrity": data_integrity,
                "user_action_load": user_action_score,
                "historical_reliability": historical_reliability,
            },
        }

    def _user_action_score(self, load: int) -> float:
        if load <= USER_ACTION_LOAD_IDEAL_MAX:
            return 100.0
        if load >= USER_ACTION_LOAD_WARNING:
            return max(0.0, 100.0 - (load - USER_ACTION_LOAD_IDEAL_MAX) * 8)
        return max(50.0, 100.0 - (load - USER_ACTION_LOAD_IDEAL_MAX) * 15)

    def _action_load_status(self, load: int) -> str:
        if load <= USER_ACTION_LOAD_IDEAL_MAX:
            return "IDEAL"
        if load >= USER_ACTION_LOAD_WARNING:
            return "WARNING"
        return "ELEVATED"

    def count_required_actions(self, *, ai_cio: dict[str, Any]) -> tuple[int, list[str]]:
        actions: list[str] = []
        for rec in ai_cio.get("recommendations") or []:
            if rec.get("requires_approval", True):
                desc = str(rec.get("description") or rec.get("action") or "")
                if desc and desc.upper() != "NO_ACTION":
                    actions.append(desc)
        for item in ai_cio.get("recommended_actions") or []:
            if item and str(item) not in actions:
                actions.append(str(item))
        return len(actions), actions
