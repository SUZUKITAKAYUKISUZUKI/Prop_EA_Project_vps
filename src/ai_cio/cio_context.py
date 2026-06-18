"""Build AI CIO context from CIL and APM v2 reports only."""
from __future__ import annotations

from typing import Any


class CioContext:
    def build(self, *, cil_report: dict[str, Any], apm_v2_report: dict[str, Any]) -> dict[str, Any]:
        opportunity = cil_report.get("opportunity_report") or {}
        risk = cil_report.get("risk_report") or {}
        confidence = cil_report.get("confidence_report") or {}
        capital = cil_report.get("capital_efficiency_report") or {}

        return {
            "profile_id": cil_report.get("profile_id"),
            "investment_state": list(cil_report.get("investment_state") or []),
            "executive_score": float(cil_report.get("executive_score") or 0),
            "opportunity_score": float(cil_report.get("opportunity_score") or 0),
            "risk_score": float(cil_report.get("risk_score") or 0),
            "confidence_score": float(cil_report.get("confidence_score") or 0),
            "capital_efficiency": float(cil_report.get("capital_efficiency") or 0),
            "intelligence_trust": float(cil_report.get("intelligence_trust") or 0),
            "portfolio_health": float(cil_report.get("portfolio_health") or 0),
            "top_opportunity": cil_report.get("top_opportunity"),
            "top_risk": cil_report.get("top_risk"),
            "top_opportunities": list(cil_report.get("top_opportunities") or opportunity.get("opportunities") or []),
            "top_risks": list(cil_report.get("top_risks") or risk.get("risks") or []),
            "opportunity_detail": opportunity,
            "risk_detail": risk,
            "confidence_detail": confidence,
            "capital_detail": capital,
            "executive_learning_score": float(apm_v2_report.get("executive_score_v2") or 0),
            "decision_accuracy": float(apm_v2_report.get("decision_accuracy") or 0),
            "board_consensus": float(apm_v2_report.get("board_consensus") or 0),
            "executive_memory": list(apm_v2_report.get("executive_memory") or []),
            "executive_lessons": list(apm_v2_report.get("executive_lessons") or []),
            "decision_outcomes": list(apm_v2_report.get("decision_outcomes") or []),
            "memory_snapshot": dict(apm_v2_report.get("memory_snapshot") or {}),
            "improvement_opportunities": list(apm_v2_report.get("improvement_opportunities") or []),
            "learned_roadmap": list(apm_v2_report.get("learned_roadmap") or []),
            "board": dict(apm_v2_report.get("board") or {}),
        }
