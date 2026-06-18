"""Internal CACE API helpers."""
from __future__ import annotations

from typing import Any

from src.cace.engine import CaceEngine


def extract_allocation_confidence(report: dict[str, Any]) -> dict[str, Any]:
    return dict(report.get("allocation_confidence") or {})


def extract_strategy_confidence(report: dict[str, Any]) -> list[dict[str, Any]]:
    return list(report.get("strategy_confidence") or [])


def extract_portfolio_confidence(report: dict[str, Any]) -> dict[str, Any]:
    return dict(report.get("portfolio_confidence") or {})


def build_engine(*, owns_connections: bool = False) -> CaceEngine:
    return CaceEngine(owns_connections=owns_connections)
