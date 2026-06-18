"""Executive roadmap scheduling for APM v1."""
from __future__ import annotations

from typing import Any


class ExecutiveScheduler:
    HORIZONS = (
        ("today", 0),
        ("week_2", 14),
        ("week_4", 28),
        ("week_6", 42),
        ("day_60", 60),
        ("day_90", 90),
    )

    def build(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        roadmap: list[dict[str, Any]] = []
        action_idx = 0
        for label, _days in self.HORIZONS:
            if action_idx >= len(actions):
                break
            action = actions[action_idx]
            if action.get("action_type") == "NO_ACTION":
                action_idx += 1
                continue
            roadmap.append(
                {
                    "horizon": label,
                    "action_type": action.get("action_type"),
                    "strategy": action.get("strategy"),
                    "description": self._describe(action),
                    "confidence": action.get("confidence"),
                    "status": action.get("status"),
                }
            )
            action_idx += 1
        return roadmap

    def _describe(self, action: dict[str, Any]) -> str:
        action_type = str(action.get("action_type") or "")
        strategy = action.get("strategy")
        if action_type == "PROMOTE_STRATEGY" and strategy:
            return f"PROMOTE {strategy}"
        if action_type == "ALLOCATION_REBALANCE":
            return "Increase Allocation"
        if action_type == "PROFILE_SWITCH":
            return "Switch Funded Profile"
        if action_type == "RETIRE_STRATEGY" and strategy:
            return f"Retire {strategy}"
        if action_type == "DEMOTE_STRATEGY" and strategy:
            return f"Demote {strategy}"
        return action_type.replace("_", " ").title()
