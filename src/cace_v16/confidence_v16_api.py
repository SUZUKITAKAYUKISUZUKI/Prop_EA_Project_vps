"""Internal CACE v1.6 API helpers."""
from __future__ import annotations

from typing import Any

from src.cace_v16.confidence_v16_engine import ConfidenceV16Engine


def extract_decay(report: dict[str, Any]) -> dict[str, Any]:
    return dict(report.get("decay") or {})


def extract_consensus(report: dict[str, Any]) -> dict[str, Any]:
    return dict(report.get("consensus") or {})


def build_engine(*, owns_connections: bool = False) -> ConfidenceV16Engine:
    return ConfidenceV16Engine(owns_connections=owns_connections)
