"""Executive report and CIO score for AI CIO v1."""
from __future__ import annotations

from typing import Any

from src.ai_cio.config import CIO_SCORE_WEIGHTS


class CioExecutiveReportEngine:
    def evaluate(self, context: dict[str, Any], *, opinion: dict[str, Any], memory: dict[str, Any]) -> dict[str, Any]:
        cio_score = round(
            CIO_SCORE_WEIGHTS["executive_investment_score"] * float(context.get("executive_score") or 0)
            + CIO_SCORE_WEIGHTS["opportunity_score"] * float(context.get("opportunity_score") or 0)
            + CIO_SCORE_WEIGHTS["risk_score"] * float(context.get("risk_score") or 0)
            + CIO_SCORE_WEIGHTS["confidence_score"] * float(context.get("confidence_score") or 0)
            + CIO_SCORE_WEIGHTS["capital_efficiency"] * float(context.get("capital_efficiency") or 0)
            + CIO_SCORE_WEIGHTS["executive_learning_score"] * float(context.get("executive_learning_score") or 0),
            2,
        )

        trust_score = round(
            float(context.get("intelligence_trust") or 0) * 0.6
            + float(context.get("board_consensus") or 0) * 0.4,
            2,
        )

        outlook = self._portfolio_outlook(context, cio_score)

        return {
            "cio_score": cio_score,
            "cio_opinion": opinion.get("cio_opinion"),
            "portfolio_state": list(context.get("investment_state") or []),
            "top_priority": memory.get("top_priority"),
            "top_opportunity": context.get("top_opportunity"),
            "top_risk": context.get("top_risk"),
            "recommended_profile": memory.get("recommended_profile"),
            "confidence": float(context.get("confidence_score") or 0),
            "trust_score": trust_score,
            "portfolio_outlook": outlook,
            "cio_score_components": {
                "executive_investment_score": context.get("executive_score"),
                "opportunity_score": context.get("opportunity_score"),
                "risk_score": context.get("risk_score"),
                "confidence_score": context.get("confidence_score"),
                "capital_efficiency": context.get("capital_efficiency"),
                "executive_learning_score": context.get("executive_learning_score"),
            },
        }

    def _portfolio_outlook(self, context: dict[str, Any], cio_score: float) -> dict[str, float]:
        base = cio_score
        health = float(context.get("portfolio_health") or base)
        durability = float((context.get("confidence_detail") or {}).get("durability_score") or 75)
        decay_30 = max(0.0, min(100.0, base * 0.98 + health * 0.02))
        decay_60 = max(0.0, min(100.0, base * 0.96 + durability * 0.04))
        decay_90 = max(0.0, min(100.0, base * 0.94 + durability * 0.06))
        return {"30d": round(decay_30, 2), "60d": round(decay_60, 2), "90d": round(decay_90, 2)}
