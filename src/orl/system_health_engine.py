"""System health aggregation for ORL v1."""
from __future__ import annotations

from typing import Any


class SystemHealthEngine:
    def evaluate(self, components: dict[str, Any]) -> dict[str, Any]:
        scores = [
            float(components.get("database_health") or 0),
            float(components.get("api_health") or 0),
            float(components.get("dashboard_health") or 0),
            float(components.get("cache_health") or 0),
            float(components.get("dependency_health") or 0),
            float(components.get("ai_cio_availability") or 0),
        ]
        valid = [s for s in scores if s > 0]
        system_health = round(sum(valid) / len(valid), 2) if valid else 0.0

        issues: list[str] = []
        for key, label in (
            ("database_health", "Database"),
            ("api_health", "API"),
            ("dashboard_health", "Dashboard"),
            ("cache_health", "Cache"),
            ("ai_cio_availability", "AI CIO"),
        ):
            val = float(components.get(key) or 0)
            if val < 70:
                issues.append(f"{label} health below threshold ({val})")

        return {
            "system_health": system_health,
            "component_scores": components,
            "issues": issues,
            "healthy": system_health >= 85,
        }
