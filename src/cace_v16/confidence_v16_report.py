"""Combined CACE v1.6 intelligence report builder."""
from __future__ import annotations

from typing import Any


class ConfidenceV16Report:
    def build(
        self,
        *,
        profile_id: str,
        cace_v15_report: dict[str, Any],
        decay: dict[str, Any],
        consensus: dict[str, Any],
    ) -> dict[str, Any]:
        forecast = decay.get("forecast") or {}
        return {
            "profile_id": profile_id,
            "confidence": float(cace_v15_report.get("confidence") or decay.get("current_confidence") or 0),
            "confidence_category": cace_v15_report.get("confidence_category"),
            "trend": cace_v15_report.get("trend"),
            "trend_strength": cace_v15_report.get("trend_strength"),
            "regime": cace_v15_report.get("regime"),
            "durability_score": decay.get("durability_score"),
            "durability_category": decay.get("durability_category"),
            "half_life": decay.get("half_life"),
            "forecast": forecast,
            "decay_curve": decay.get("decay_curve"),
            "expected_confidence_30d": forecast.get("30d"),
            "expected_confidence_60d": forecast.get("60d"),
            "expected_confidence_90d": forecast.get("90d"),
            "expected_confidence_180d": forecast.get("180d"),
            "consensus_score": consensus.get("consensus_score"),
            "consensus_category": consensus.get("consensus_category"),
            "agreement_ratio": consensus.get("agreement_ratio"),
            "agree_count": consensus.get("agree_count"),
            "total_modules": consensus.get("total_modules"),
            "recommended_action": consensus.get("recommended_action"),
            "module_agreements": consensus.get("module_agreements"),
            "participant_table": consensus.get("participant_table"),
            "cace_v15": cace_v15_report,
            "decay": decay,
            "consensus": consensus,
        }
