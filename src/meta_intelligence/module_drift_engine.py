"""Module trust drift detection."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from src.meta_intelligence.config import DRIFT_LOOKBACK_DAYS, DRIFT_WARNING_THRESHOLD, MODULES


class ModuleDriftEngine:
    def evaluate(
        self,
        *,
        current_trust: dict[str, dict[str, Any]],
        trust_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        history = trust_history or []
        alerts: list[dict[str, Any]] = []
        drift_by_module: dict[str, dict[str, Any]] = {}

        for module in MODULES:
            current_score = float((current_trust.get(module) or {}).get("trust_score") or 0)
            previous_score = self._historical_score(module, history)
            if previous_score is None:
                continue
            delta = round(current_score - previous_score, 2)
            drift_by_module[module] = {
                "module": module,
                "previous_score": previous_score,
                "current_score": current_score,
                "delta": delta,
            }
            if delta <= -DRIFT_WARNING_THRESHOLD:
                alerts.append(
                    {
                        "module": module,
                        "alert_code": f"{module}_DRIFT_WARNING",
                        "previous_score": previous_score,
                        "current_score": current_score,
                        "delta": delta,
                        "message": (
                            f"{module}: trust fell from {previous_score:.0f} to {current_score:.0f} "
                            f"over ~{DRIFT_LOOKBACK_DAYS} days."
                        ),
                    }
                )

        return {
            "drift_alerts": alerts,
            "drift_by_module": drift_by_module,
            "alert_count": len(alerts),
        }

    def _historical_score(self, module: str, history: list[dict[str, Any]]) -> float | None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=DRIFT_LOOKBACK_DAYS)
        candidates = []
        for row in history:
            if str(row.get("module") or "").upper() != module:
                continue
            ts = self._parse_ts(row.get("timestamp"))
            if ts and ts <= cutoff:
                candidates.append((ts, float(row.get("trust_score") or 0)))
        if not candidates:
            for row in history:
                if str(row.get("module") or "").upper() == module:
                    return float(row.get("trust_score") or 0)
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _parse_ts(self, value: Any) -> datetime | None:
        if not value:
            return None
        try:
            ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts
        except ValueError:
            return None
