"""Recommendation prioritization for AI CIO."""
from __future__ import annotations

from typing import Any

from src.ai_cio.config import PRIORITY_CAPITAL_PRESERVATION, PRIORITY_GROWTH, PRIORITY_SURVIVAL


class CioPriorityEngine:
    def evaluate(
        self,
        *,
        context: dict[str, Any],
        strategy_recs: list[dict[str, Any]],
        allocation_recs: list[dict[str, Any]],
        profile_recs: list[dict[str, Any]],
        risk_recs: list[dict[str, Any]],
        portfolio_recs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        combined = strategy_recs + allocation_recs + profile_recs + risk_recs + portfolio_recs
        if not combined:
            return [
                {
                    "category": "NO_ACTION",
                    "priority": 99,
                    "action": "NO_ACTION",
                    "description": "Maintain current portfolio posture; no executive action required",
                    "confidence": float(context.get("confidence_score") or 0),
                    "source": "AI CIO",
                    "requires_approval": False,
                }
            ]

        return sorted(
            combined,
            key=lambda r: (int(r.get("priority") or 99), -float(r.get("confidence") or 0)),
        )

    def build_risk_recommendations(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        recs: list[dict[str, Any]] = []
        states = set(context.get("investment_state") or [])
        risk_detail = context.get("risk_detail") or {}

        if "RECOVERY_MODE" in states:
            recs.append(
                {
                    "category": "RISK",
                    "priority": PRIORITY_SURVIVAL,
                    "action": "ENTER_RECOVERY",
                    "description": "Enter recovery protocol to protect capital",
                    "confidence": float(context.get("confidence_score") or 0),
                    "source": "CIL",
                    "requires_approval": True,
                }
            )

        if float(context.get("risk_score") or 100) < 65 or "HIGH_RISK" in states:
            recs.append(
                {
                    "category": "RISK",
                    "priority": PRIORITY_SURVIVAL,
                    "action": "REDUCE_RISK",
                    "description": "Reduce portfolio risk exposure",
                    "confidence": float(context.get("confidence_score") or 0),
                    "source": "CIL",
                    "requires_approval": True,
                }
            )

        concentration = float(risk_detail.get("strategy_concentration") or 0)
        if "STRATEGY_CONCENTRATED" in states or concentration >= 0.4:
            recs.append(
                {
                    "category": "RISK",
                    "priority": PRIORITY_CAPITAL_PRESERVATION,
                    "action": "REDUCE_CONCENTRATION",
                    "description": "Reduce strategy concentration risk",
                    "confidence": float(context.get("confidence_score") or 0),
                    "source": "CIL",
                    "requires_approval": True,
                }
            )

        if "DIVERSIFIED" not in states and len(context.get("top_risks") or []) >= 2:
            recs.append(
                {
                    "category": "RISK",
                    "priority": PRIORITY_CAPITAL_PRESERVATION,
                    "action": "INCREASE_DIVERSIFICATION",
                    "description": "Increase portfolio diversification",
                    "confidence": float(context.get("confidence_score") or 0),
                    "source": "CIL",
                    "requires_approval": True,
                }
            )
        return recs

    def build_portfolio_recommendations(self, context: dict[str, Any], *, opinion: str) -> list[dict[str, Any]]:
        if opinion in {"NO_ACTION", "MAINTAIN"}:
            return []
        return [
            {
                "category": "PORTFOLIO",
                "priority": PRIORITY_GROWTH if opinion in {"ACCUMULATE", "STRONG_BUY_PORTFOLIO"} else PRIORITY_CAPITAL_PRESERVATION,
                "action": opinion,
                "description": f"Executive portfolio posture: {opinion.replace('_', ' ').title()}",
                "confidence": float(context.get("confidence_score") or 0),
                "source": "AI CIO Opinion",
                "requires_approval": True,
            }
        ]
