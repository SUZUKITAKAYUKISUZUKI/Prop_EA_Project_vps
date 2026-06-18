"""Incubation stage evaluation."""
from __future__ import annotations

from typing import Any


class IncubationEngine:
    def evaluate(self, metrics: dict[str, Any]) -> dict[str, Any]:
        bt_complete = bool(metrics.get("bt_complete"))
        wft_complete = bool(metrics.get("wft_complete"))
        return {
            "stage": "INCUBATION",
            "bt_complete": bt_complete,
            "wft_complete": wft_complete,
            "ready_for_candidate": bt_complete and not wft_complete,
            "allocation_pct": 0.0,
            "notes": "Research phase — no capital allocation",
        }
