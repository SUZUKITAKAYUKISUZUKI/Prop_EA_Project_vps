"""Aggregates intelligence from all Portfolio OS layers."""
from __future__ import annotations

from typing import Any


class IntelligenceAggregator:
    def aggregate(
        self,
        *,
        upstream: dict[str, Any],
        cace_v16_report: dict[str, Any],
        cace_v17_report: dict[str, Any],
        mie_report: dict[str, Any],
        apm_v1_report: dict[str, Any],
        apm_v2_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "profile_id": upstream.get("profile_id"),
            "prae_v2": upstream.get("prae_v2") or {},
            "paae": upstream.get("paae") or {},
            "pdts": upstream.get("pdts") or {},
            "slm": upstream.get("slm") or {},
            "age_v4": upstream.get("age_v4") or {},
            "state_analytics": upstream.get("state_analytics") or {},
            "cace_v16": cace_v16_report,
            "cace_v17": cace_v17_report,
            "mie": mie_report,
            "apm_v1": apm_v1_report,
            "apm_v2": apm_v2_report or {},
        }
