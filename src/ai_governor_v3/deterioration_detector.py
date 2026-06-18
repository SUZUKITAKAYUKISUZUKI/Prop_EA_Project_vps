"""Detect forecasted deterioration across governance dimensions."""
from __future__ import annotations

from typing import Any

from src.ai_governor.governor_context import GovernorContext
from src.ai_governor_v3.forecast_config import ForecastConfig, DEFAULT_CONFIG


class DeteriorationDetector:
    def __init__(self, config: ForecastConfig | None = None) -> None:
        self._config = config or DEFAULT_CONFIG

    def detect(self, context: GovernorContext, forecasts: dict[str, Any]) -> list[dict[str, Any]]:
        alerts: list[dict[str, Any]] = []
        threshold = self._config.deterioration_threshold_pct

        alerts.extend(self._health_alerts(forecasts.get("health") or {}, threshold))
        alerts.extend(self._risk_alerts(forecasts.get("risk_budget") or {}, threshold))
        alerts.extend(self._lifecycle_alerts(forecasts.get("lifecycle") or {}, threshold))
        alerts.extend(self._allocation_alerts(forecasts.get("allocation") or {}, threshold))
        alerts.extend(self._profile_alerts(forecasts.get("profile") or {}))

        return sorted(alerts, key=lambda a: _severity_rank(a.get("severity", "LOW")), reverse=True)

    def _health_alerts(self, health: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
        current = float(health.get("current_health") or 0)
        alerts = []
        for key, value in (health.get("future_health") or {}).items():
            if key == "current":
                continue
            projected = float(value)
            if current > 0 and (current - projected) / current * 100.0 >= threshold:
                alerts.append(
                    _alert(
                        "HEALTH_DETERIORATION",
                        "HIGH",
                        {
                            "horizon": key,
                            "current": current,
                            "projected": projected,
                            "degradation_pct": round((current - projected) / current * 100, 1),
                        },
                    )
                )
        return alerts

    def _risk_alerts(self, risk: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
        current = float(risk.get("risk_budget_remaining_pct") or 0)
        alerts = []
        for key, value in (risk.get("risk_budget_forecast") or {}).items():
            if key == "current":
                continue
            projected = float(value)
            if current > 0 and (current - projected) / current * 100.0 >= threshold:
                alerts.append(
                    _alert(
                        "RISK_DETERIORATION",
                        "CRITICAL",
                        {
                            "horizon": key,
                            "current_remaining": current,
                            "projected_remaining": projected,
                        },
                    )
                )
        for raw in risk.get("alerts") or []:
            alerts.append(_alert(str(raw), "CRITICAL", {"source": "risk_budget_forecaster"}))
        return alerts

    def _lifecycle_alerts(self, lifecycle: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
        alerts = []
        for sid, data in (lifecycle.get("strategies") or {}).items():
            cur = data.get("projections", {}).get("current", {})
            cur_fit = float(cur.get("portfolio_fit") or 0)
            for key, proj in data.get("projections", {}).items():
                if key == "current":
                    continue
                fit = float(proj.get("portfolio_fit") or 0)
                if cur_fit > 0 and (cur_fit - fit) / cur_fit * 100.0 >= threshold:
                    alerts.append(
                        _alert(
                            "FIT_DETERIORATION",
                            "MEDIUM",
                            {
                                "strategy": sid,
                                "horizon": key,
                                "current_fit": cur_fit,
                                "projected_fit": fit,
                                "flags": data.get("flags"),
                            },
                        )
                    )
                    break
        return alerts

    def _allocation_alerts(self, allocation: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
        alerts = []
        for warning in allocation.get("warnings") or []:
            alerts.append(_alert("ALLOCATION_DRIFT_WARNING", "MEDIUM", {"warning": warning}))
        if float(allocation.get("max_current_drift_pct") or 0) >= threshold:
            alerts.append(
                _alert(
                    "ALLOCATION_DETERIORATION",
                    "MEDIUM",
                    {"max_drift_pct": allocation.get("max_current_drift_pct")},
                )
            )
        return alerts

    def _profile_alerts(self, profile: dict[str, Any]) -> list[dict[str, Any]]:
        alerts = []
        for risk in profile.get("profile_transition_risk") or []:
            alerts.append(
                _alert(
                    "PROFILE_TRANSITION_RISK",
                    "HIGH",
                    {
                        "current_state": profile.get("current_state"),
                        "forecast": profile.get("state_forecast"),
                    },
                )
            )
        return alerts


def _alert(alert_type: str, severity: str, details: dict[str, Any]) -> dict[str, Any]:
    return {"alert_type": alert_type, "severity": severity, "details_json": details}


def _severity_rank(severity: str) -> int:
    return {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "INFO": 0}.get(severity.upper(), 0)
