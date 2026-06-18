"""CIO intelligence report builder."""
from __future__ import annotations

from typing import Any


class IntelligenceReport:
    def build(
        self,
        *,
        profile_id: str,
        summary: dict[str, Any],
        opportunity: dict[str, Any],
        risk: dict[str, Any],
        confidence: dict[str, Any],
        capital_efficiency: dict[str, Any],
        bundle: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "profile_id": profile_id,
            **summary,
            "opportunity_report": opportunity,
            "risk_report": risk,
            "confidence_report": confidence,
            "capital_efficiency_report": capital_efficiency,
            "intelligence_trust": summary.get("intelligence_trust"),
            "top_opportunities": opportunity.get("opportunities"),
            "top_risks": risk.get("risks"),
            "layers": {
                "prae": bool(bundle.get("prae_v2")),
                "paae": bool(bundle.get("paae")),
                "pdts": bool(bundle.get("pdts")),
                "slm": bool(bundle.get("slm")),
                "age": bool(bundle.get("age_v4")),
                "cace": bool(bundle.get("cace_v17")),
                "mie": bool(bundle.get("mie")),
                "apm": bool(bundle.get("apm_v1")),
            },
        }
