"""Confidence scoring from CACE layers."""
from __future__ import annotations

from typing import Any


class ConfidenceEngine:
    def evaluate(self, bundle: dict[str, Any]) -> dict[str, Any]:
        v17 = bundle.get("cace_v17") or {}
        v16 = bundle.get("cace_v16") or {}

        confidence_score = round(
            float(v17.get("confidence") or 0) * 0.35
            + float(v17.get("calibration_score") or 0) * 0.25
            + float(v16.get("consensus_score") or 0) * 0.20
            + float(v16.get("durability_score") or 0) * 0.20,
            2,
        )

        return {
            "confidence_score": confidence_score,
            "portfolio_confidence": v17.get("confidence"),
            "calibration_score": v17.get("calibration_score"),
            "consensus_score": v16.get("consensus_score"),
            "durability_score": v16.get("durability_score"),
            "recommended_action": v16.get("recommended_action") or v17.get("recommended_action"),
        }
