"""Trust-weighted module recommendations."""
from __future__ import annotations

from typing import Any


class RecommendationEngine:
    def evaluate(
        self,
        *,
        trust_scores: dict[str, dict[str, Any]],
        rankings: list[dict[str, Any]],
        cace_v16_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        v16 = cace_v16_report or {}
        recommended_action = str(v16.get("recommended_action") or "NO_ACTION")
        trusted_modules = [
            item["module"]
            for item in rankings
            if float(item.get("trust_score") or 0) >= 70.0
        ]
        primary_module = rankings[0]["module"] if rankings else "AGE"

        return {
            "recommended_action": recommended_action,
            "primary_trusted_module": primary_module,
            "trusted_modules": trusted_modules,
            "governance_note": (
                f"Weight {primary_module} recommendations highest; "
                f"cross-check with {', '.join(trusted_modules[1:3]) or 'peer modules'}."
            ),
        }
