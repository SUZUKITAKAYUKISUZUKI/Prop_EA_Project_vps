"""Internal strategic governor API for AGE v4."""
from __future__ import annotations

from typing import Any

from src.ai_governor_v4.engine import StrategicGovernorEngine


def extract_future_scenarios(report: dict[str, Any]) -> list[dict[str, Any]]:
    return list(report.get("future_scenarios") or [])


def extract_future_rankings(report: dict[str, Any]) -> dict[str, Any]:
    return dict(report.get("future_rankings") or {})


def extract_best_future(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "recommended_action": report.get("recommended_action"),
        "confidence": report.get("confidence"),
        "strategic_score": report.get("strategic_score"),
        "rank_category": report.get("rank_category"),
        "expected_benefit": report.get("expected_benefit"),
        "expected_risk": report.get("expected_risk"),
        "rationale": report.get("rationale"),
        "best_future_metrics": report.get("best_future_metrics"),
    }


def extract_scenario_comparison(report: dict[str, Any]) -> dict[str, Any]:
    return dict(report.get("scenario_comparison") or {})


def build_engine(*, owns_connections: bool = False) -> StrategicGovernorEngine:
    return StrategicGovernorEngine(owns_connections=owns_connections)
