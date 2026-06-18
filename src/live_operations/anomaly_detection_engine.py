"""Anomaly detection for RC2 live operations."""
from __future__ import annotations

from typing import Any


class AnomalyDetectionEngine:
    def evaluate(
        self,
        *,
        ops_context: dict[str, Any],
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        anomalies: list[dict[str, Any]] = []
        ai_cio = ops_context.get("ai_cio_report") or {}
        current_conf = float(ai_cio.get("confidence") or 0)

        prior_confidences = [
            float(h.get("confidence") or h.get("cio_confidence") or 0)
            for h in (history or [])
            if h.get("confidence") or h.get("cio_confidence")
        ]
        if prior_confidences:
            avg_prior = sum(prior_confidences) / len(prior_confidences)
            if avg_prior >= 80 and current_conf <= avg_prior - 25:
                anomalies.append(
                    {
                        "type": "confidence_drop",
                        "severity": "ALERT",
                        "from": round(avg_prior, 2),
                        "to": current_conf,
                        "message": f"AI CIO confidence dropped from {avg_prior:.0f} to {current_conf:.0f}",
                    }
                )

        production = ops_context.get("production_report") or {}
        if float(production.get("resilience_score") or 100) < 70:
            anomalies.append(
                {
                    "type": "resilience_drop",
                    "severity": "WARNING",
                    "message": "Production resilience below threshold",
                }
            )

        orl = ops_context.get("orl_report") or {}
        if float(orl.get("readiness_score") or 100) < 75:
            anomalies.append(
                {
                    "type": "orl_readiness_drop",
                    "severity": "WARNING",
                    "message": "ORL readiness below daily operations threshold",
                }
            )

        return {
            "anomalies": anomalies,
            "anomaly_count": len(anomalies),
            "has_critical": any(a.get("severity") == "ALERT" for a in anomalies),
        }
