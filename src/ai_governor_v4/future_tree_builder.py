"""Build multi-future tree for AGE v4 dashboard."""
from __future__ import annotations

from typing import Any

from src.ai_governor.governor_context import GovernorContext
from src.ai_governor_v4.future_branch import FutureBranch
from src.ai_governor_v4.strategic_config import StrategicConfig, DEFAULT_STRATEGIC_CONFIG


class FutureTreeBuilder:
    def __init__(self, config: StrategicConfig | None = None) -> None:
        self._config = config or DEFAULT_STRATEGIC_CONFIG

    def build(
        self,
        context: GovernorContext,
        branches: list[FutureBranch],
    ) -> dict[str, Any]:
        root = {
            "node_id": "root",
            "label": f"{context.current_state.upper()} / Health {context.health_score:.0f}",
            "state": context.current_state,
            "health_score": context.health_score,
            "dd_pct": context.dd_pct,
            "children": [],
        }

        for branch in branches:
            child = {
                "node_id": branch.branch_id,
                "label": branch.action_label,
                "action_type": branch.action_type,
                "strategic_score": branch.strategic_score,
                "rank_category": branch.rank_category,
                "horizons": self._horizon_nodes(branch),
            }
            root["children"].append(child)

        return {
            "current_state": context.current_state,
            "current_profile": context.current_profile,
            "health_score": context.health_score,
            "dd_pct": context.dd_pct,
            "tree": root,
            "branch_count": len(branches),
            "forecast_horizons": list(self._config.forecast_days),
        }

    def _horizon_nodes(self, branch: FutureBranch) -> list[dict[str, Any]]:
        nodes = []
        for days in self._config.forecast_days:
            hkey = f"{days}d"
            metrics = branch.metrics_by_horizon.get(hkey) or {}
            nodes.append(
                {
                    "horizon_days": days,
                    "health_score": metrics.get("health_score"),
                    "expected_dd": metrics.get("expected_dd"),
                    "expected_pf": metrics.get("expected_pf"),
                    "pass_probability": metrics.get("pass_probability"),
                    "recovery_probability": metrics.get("recovery_probability"),
                    "risk_budget_remaining": metrics.get("risk_budget_remaining"),
                }
            )
        return nodes
