"""Executive score v2 — learning-informed governance score."""
from __future__ import annotations

from typing import Any

from src.apm_v2.config import EXECUTIVE_V2_WEIGHTS


class ExecutiveScoreEngine:
    def evaluate(
        self,
        *,
        effectiveness: dict[str, Any],
        learning: dict[str, Any],
        apm_v1_report: dict[str, Any],
        cace_v17_report: dict[str, Any],
        board: dict[str, Any],
    ) -> dict[str, Any]:
        decision_accuracy = float(effectiveness.get("decision_accuracy") or 0)
        learning_quality = float(learning.get("learning_quality") or 0)
        portfolio_improvement = float(learning.get("portfolio_improvement") or apm_v1_report.get("executive_health") or 0)
        governor_reliability = float(
            ((apm_v1_report.get("context") or {}).get("governor_decisions") or {}).get("strategic_confidence")
            or apm_v1_report.get("executive_score")
            or 70
        )
        confidence_reliability = float(cace_v17_report.get("calibration_score") or 70)

        executive_score = round(
            EXECUTIVE_V2_WEIGHTS["decision_accuracy"] * decision_accuracy
            + EXECUTIVE_V2_WEIGHTS["learning_quality"] * learning_quality
            + EXECUTIVE_V2_WEIGHTS["portfolio_improvement"] * portfolio_improvement
            + EXECUTIVE_V2_WEIGHTS["governor_reliability"] * governor_reliability
            + EXECUTIVE_V2_WEIGHTS["confidence_reliability"] * confidence_reliability,
            2,
        )

        return {
            "executive_score": executive_score,
            "executive_score_v2": executive_score,
            "components": {
                "decision_accuracy": round(decision_accuracy, 2),
                "learning_quality": round(learning_quality, 2),
                "portfolio_improvement": round(portfolio_improvement, 2),
                "governor_reliability": round(governor_reliability, 2),
                "confidence_reliability": round(confidence_reliability, 2),
            },
            "board_consensus": board.get("board_consensus"),
        }
