"""Internal helpers for CIO Intelligence Layer v1."""
from __future__ import annotations

from typing import Any


def extract_investment_state(report: dict[str, Any]) -> list[str]:
    return list(report.get("investment_state") or [])


def extract_executive_score(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "executive_score": report.get("executive_score"),
        "components": report.get("executive_components"),
    }


def build_engine(*, owns_connections: bool = False):
    from src.cio_intelligence.engine import CioIntelligenceEngine

    return CioIntelligenceEngine(owns_connections=owns_connections)
