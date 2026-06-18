"""CIO portfolio opinion from CIL context."""
from __future__ import annotations

from typing import Any


class CioOpinionEngine:
    def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
        states = set(context.get("investment_state") or [])
        executive = float(context.get("executive_score") or 0)
        opportunity = float(context.get("opportunity_score") or 0)
        risk = float(context.get("risk_score") or 0)

        if "RECOVERY_MODE" in states:
            opinion = "RECOVERY"
        elif "HIGH_RISK" in states and risk < 65:
            opinion = "HIGH_RISK"
        elif risk < 60 or executive < 55:
            opinion = "DEFENSIVE"
        elif executive >= 88 and opportunity >= 85 and risk >= 75:
            opinion = "STRONG_BUY_PORTFOLIO"
        elif opportunity >= 78 and executive >= 72:
            opinion = "ACCUMULATE"
        elif executive >= 60:
            opinion = "MAINTAIN"
        elif executive < 50 and opportunity < 50:
            opinion = "NO_ACTION"
        else:
            opinion = "MAINTAIN"

        return {
            "cio_opinion": opinion,
            "opinion_rationale": self._rationale(opinion, states, executive, opportunity, risk),
        }

    def _rationale(
        self,
        opinion: str,
        states: set[str],
        executive: float,
        opportunity: float,
        risk: float,
    ) -> str:
        state_txt = ", ".join(sorted(states)) if states else "neutral"
        return (
            f"Opinion {opinion} based on executive score {executive:.0f}, "
            f"opportunity {opportunity:.0f}, risk posture {risk:.0f}, states: {state_txt}"
        )
