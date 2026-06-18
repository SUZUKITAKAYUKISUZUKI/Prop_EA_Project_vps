"""Daily operations cycle for RC2."""
from __future__ import annotations

from typing import Any


class DailyOperationsEngine:
    def build_required_actions(self, *, ai_cio: dict[str, Any], issues: dict[str, Any]) -> list[str]:
        _, actions = self._extract_actions(ai_cio)
        if issues.get("has_critical"):
            for item in issues.get("open_issues") or []:
                if str(item.get("severity")).upper() in ("CRITICAL", "ALERT"):
                    msg = str(item.get("message") or "")
                    if msg and msg not in actions:
                        actions.insert(0, f"URGENT: {msg}")
        return actions[:15]

    def _extract_actions(self, ai_cio: dict[str, Any]) -> tuple[int, list[str]]:
        from src.live_operations.operational_metrics_engine import OperationalMetricsEngine

        return OperationalMetricsEngine().count_required_actions(ai_cio=ai_cio)
