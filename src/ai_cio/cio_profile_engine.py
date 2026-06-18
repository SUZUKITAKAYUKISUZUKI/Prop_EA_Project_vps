"""Profile recommendations for AI CIO."""
from __future__ import annotations

from typing import Any

from src.ai_cio.config import PRIORITY_SURVIVAL


class CioProfileEngine:
    def evaluate(self, context: dict[str, Any], *, recommended_profile: str) -> list[dict[str, Any]]:
        current = str(context.get("profile_id") or "FundedBalanced")
        if recommended_profile == current:
            return []

        states = set(context.get("investment_state") or [])
        priority = PRIORITY_SURVIVAL if "RECOVERY_MODE" in states or "HIGH_RISK" in states else 2

        return [
            {
                "category": "PROFILE",
                "priority": priority,
                "action": "PROFILE_SWITCH",
                "description": f"Switch profile from {current} to {recommended_profile}",
                "current_profile": current,
                "recommended_profile": recommended_profile,
                "confidence": float(context.get("confidence_score") or 0),
                "source": "AI CIO",
                "requires_approval": True,
            }
        ]
