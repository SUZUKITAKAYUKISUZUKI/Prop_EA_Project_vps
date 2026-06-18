"""Feedback loop from board consensus to APM recommendations."""
from __future__ import annotations

from typing import Any


class RecommendationFeedbackEngine:
    def evaluate(
        self,
        *,
        board: dict[str, Any],
        apm_v1_report: dict[str, Any],
        learning: dict[str, Any],
    ) -> dict[str, Any]:
        consensus = float(board.get("board_consensus") or 0)
        recommended = board.get("majority_recommendation") or "NO_ACTION"
        return {
            "board_informed_recommendation": recommended,
            "consensus_weight": consensus,
            "feedback_note": (
                f"Board consensus {consensus:.1f} — weight {board.get('agree_count')}/"
                f"{board.get('total_directors')} director agreement."
            ),
            "learning_adjustments": learning.get("improvement_opportunities") or [],
            "prior_apm_action": (apm_v1_report.get("recommendations") or {}).get("recommended_action"),
        }
