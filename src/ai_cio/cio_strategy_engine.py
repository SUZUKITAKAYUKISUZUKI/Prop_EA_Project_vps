"""Strategy recommendations for AI CIO."""
from __future__ import annotations

from typing import Any

from src.ai_cio.config import PRIORITY_GROWTH, RECOMMENDATION_CATEGORIES


class CioStrategyEngine:
    def evaluate(self, context: dict[str, Any]) -> list[dict[str, Any]]:
        recs: list[dict[str, Any]] = []
        for item in context.get("top_opportunities") or []:
            strategy = item.get("strategy")
            if not strategy or strategy == "PORTFOLIO":
                continue
            recs.append(
                {
                    "category": "STRATEGY",
                    "priority": PRIORITY_GROWTH,
                    "action": f"PROMOTE_STRATEGY",
                    "strategy": strategy,
                    "description": f"Promote strategy {strategy}",
                    "confidence": float(item.get("score") or context.get("confidence_score") or 0),
                    "source": item.get("source") or "CIL",
                    "requires_approval": True,
                }
            )

        board = context.get("board") or {}
        majority = board.get("majority_recommendation")
        if majority and "PROMOTE" in str(majority).upper():
            recs.append(
                {
                    "category": "STRATEGY",
                    "priority": PRIORITY_GROWTH,
                    "action": str(majority),
                    "description": f"Board consensus: {majority}",
                    "confidence": float(board.get("average_confidence") or context.get("board_consensus") or 0),
                    "source": "APM v2 Board",
                    "requires_approval": True,
                }
            )
        return recs
