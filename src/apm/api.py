"""Internal helpers for APM v1."""
from __future__ import annotations

from typing import Any


def extract_queue(report: dict[str, Any]) -> list[dict[str, Any]]:
    return list(report.get("execution_queue") or report.get("approval_queue") or [])


def extract_roadmap(report: dict[str, Any]) -> list[dict[str, Any]]:
    return list(report.get("roadmap") or [])


def extract_opportunities(report: dict[str, Any]) -> list[dict[str, Any]]:
    return list(report.get("opportunities") or [])


def extract_risk_alerts(report: dict[str, Any]) -> list[dict[str, Any]]:
    return list(report.get("risk_alerts") or [])


def build_engine(*, owns_connections: bool = False):
    from src.apm.engine import ApmEngine

    return ApmEngine(owns_connections=owns_connections)
