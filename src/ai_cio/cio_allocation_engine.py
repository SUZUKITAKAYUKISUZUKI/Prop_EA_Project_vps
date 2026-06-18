"""Allocation recommendations for AI CIO."""
from __future__ import annotations

from typing import Any

from src.ai_cio.config import PRIORITY_CAPITAL_PRESERVATION, PRIORITY_GROWTH


class CioAllocationEngine:
    def evaluate(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        recs: list[dict[str, Any]] = []
        states = set(context.get("investment_state") or [])
        capital = context.get("capital_detail") or {}

        if "UNDER_ALLOCATED" in states:
            recs.append(
                {
                    "category": "ALLOCATION",
                    "priority": PRIORITY_GROWTH,
                    "action": "INCREASE_ALLOCATION",
                    "description": "Portfolio is under-allocated relative to optimal weights",
                    "confidence": float(context.get("confidence_score") or 0),
                    "source": "CIL",
                    "requires_approval": True,
                }
            )
        if "OVER_ALLOCATED" in states:
            recs.append(
                {
                    "category": "ALLOCATION",
                    "priority": PRIORITY_CAPITAL_PRESERVATION,
                    "action": "REDUCE_ALLOCATION",
                    "description": "Portfolio is over-allocated; reduce exposure",
                    "confidence": float(context.get("confidence_score") or 0),
                    "source": "CIL",
                    "requires_approval": True,
                }
            )

        unused = float(capital.get("unused_capacity_pct") or 0)
        if unused >= 15 and "HIGH_GROWTH" in states:
            recs.append(
                {
                    "category": "ALLOCATION",
                    "priority": PRIORITY_GROWTH,
                    "action": "DEPLOY_UNUSED_CAPACITY",
                    "description": f"Deploy unused capacity ({unused:.0f}% available)",
                    "confidence": float(context.get("opportunity_score") or 0),
                    "source": "CIL Capital Efficiency",
                    "requires_approval": True,
                }
            )
        return recs
