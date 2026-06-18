"""Internal CACE v1.5 API helpers."""
from __future__ import annotations

from typing import Any

from src.cace.confidence_intelligence_engine import ConfidenceIntelligenceEngine


def extract_breakdown(report: dict[str, Any]) -> dict[str, float]:
    return dict(report.get("breakdown") or {})


def extract_trend(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "confidence": report.get("confidence"),
        "trend": report.get("trend"),
        "trend_strength": report.get("trend_strength"),
        "trend_direction": report.get("trend_direction"),
        "trend_category": report.get("trend_category"),
        "trend_windows": report.get("trend_windows"),
        "confidence_evolution": report.get("confidence_evolution"),
    }


def extract_regime(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "regime": report.get("regime"),
        "confidence_modifier": report.get("regime_modifier"),
        "raw_confidence": report.get("raw_confidence"),
        "adjusted_confidence": report.get("confidence"),
        "regime_metrics": report.get("regime_metrics"),
        "regime_rationale": report.get("regime_rationale"),
        "regime_appropriate": report.get("regime_appropriate"),
    }


def build_intelligence_engine() -> ConfidenceIntelligenceEngine:
    return ConfidenceIntelligenceEngine()
