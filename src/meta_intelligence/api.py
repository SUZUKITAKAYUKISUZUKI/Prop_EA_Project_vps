"""Internal helpers for Meta Intelligence Engine v1."""
from __future__ import annotations

from typing import Any


def extract_trust_scores(report: dict[str, Any]) -> dict[str, Any]:
    return dict(report.get("module_trust_scores") or {})


def extract_rankings(report: dict[str, Any]) -> list[dict[str, Any]]:
    return list(report.get("module_rankings") or [])


def extract_drift(report: dict[str, Any]) -> dict[str, Any]:
    return dict(report.get("drift") or {})


def extract_improvements(report: dict[str, Any]) -> list[dict[str, Any]]:
    return list(report.get("self_improvement_notes") or [])


def build_engine(*, owns_connections: bool = False):
    from src.meta_intelligence.engine import MetaIntelligenceEngine

    return MetaIntelligenceEngine(owns_connections=owns_connections)
