"""Live readiness scoring for RC2."""
from __future__ import annotations

from typing import Any

from src.live_operations.config import READINESS_THRESHOLDS, USER_ACTION_LOAD_IDEAL_MAX


class LiveReadinessEngine:
    def evaluate(
        self,
        *,
        operational_score: float,
        live_readiness_components: dict[str, Any],
        issues: dict[str, Any],
    ) -> dict[str, Any]:
        orl_ready = float(live_readiness_components.get("orl_readiness") or 0)
        production_ready = float(live_readiness_components.get("production_readiness") or 0)
        user_load = int(live_readiness_components.get("user_action_load") or 0)

        live_readiness = round(
            operational_score * 0.5 + orl_ready * 0.25 + production_ready * 0.25,
            2,
        )
        if user_load > USER_ACTION_LOAD_IDEAL_MAX:
            live_readiness = min(live_readiness, max(0.0, live_readiness - (user_load - 3) * 2))

        status = self._status(live_readiness)
        production_ready_flag = (
            live_readiness >= 90
            and operational_score >= 90
            and user_load <= USER_ACTION_LOAD_IDEAL_MAX
            and not issues.get("has_critical")
        )

        return {
            "live_readiness": live_readiness,
            "live_readiness_status": status,
            "portfolio_os_complete": production_ready_flag,
            "rc2_passed": production_ready_flag,
        }

    def _status(self, score: float) -> str:
        for threshold, label in READINESS_THRESHOLDS:
            if score >= threshold:
                return label
        return "NOT_READY"
