"""Internal helpers for AI CIO v1."""
from __future__ import annotations

from typing import Any


def extract_opinion(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile_id": report.get("profile_id"),
        "cio_opinion": report.get("cio_opinion"),
        "cio_score": report.get("cio_score"),
        "portfolio_state": report.get("portfolio_state"),
        "opinion_rationale": report.get("opinion_rationale"),
        "confidence": report.get("confidence"),
        "trust_score": report.get("trust_score"),
    }


def extract_memory(report: dict[str, Any]) -> dict[str, Any]:
    memory = report.get("cio_memory") or {}
    return {
        "past_successes": memory.get("past_successes"),
        "past_mistakes": memory.get("past_mistakes"),
        "recurring_problems": memory.get("recurring_problems"),
        "executive_lessons": report.get("executive_lessons"),
    }
