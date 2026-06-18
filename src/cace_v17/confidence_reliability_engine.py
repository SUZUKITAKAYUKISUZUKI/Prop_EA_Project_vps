"""Confidence reliability trends over rolling windows."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from src.cace_v17.calibration_config import RELIABILITY_WINDOWS, RELIABILITY_TRENDS


class ConfidenceReliabilityEngine:
    def evaluate(self, evaluated_records: list[dict[str, Any]]) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        window_scores: dict[str, float] = {}
        window_counts: dict[str, int] = {}

        for days in RELIABILITY_WINDOWS:
            cutoff = now - timedelta(days=days)
            rows = [r for r in evaluated_records if self._parse_ts(r.get("timestamp")) >= cutoff]
            if not rows:
                window_scores[f"last_{days}d"] = 0.0
                window_counts[f"last_{days}d"] = 0
                continue
            values = [
                float(r.get("accuracy_score") or r.get("confidence_accuracy") or 0)
                for r in rows
            ]
            window_scores[f"last_{days}d"] = round(sum(values) / len(values), 2)
            window_counts[f"last_{days}d"] = len(rows)

        trend = self._detect_trend(
            window_scores.get("last_30d", 0.0),
            window_scores.get("last_90d", 0.0),
            window_scores.get("last_180d", 0.0),
        )

        return {
            "window_scores": window_scores,
            "window_counts": window_counts,
            "reliability_trend": trend,
            "reliability_score": window_scores.get("last_90d") or window_scores.get("last_30d") or 0.0,
        }

    def _detect_trend(self, score_30: float, score_90: float, score_180: float) -> str:
        if score_30 == 0.0 and score_90 == 0.0:
            return "STABLE"
        if score_30 > score_90 + 3.0 and score_90 >= score_180:
            return "IMPROVING"
        if score_30 < score_90 - 3.0 and score_90 <= score_180:
            return "DEGRADING"
        return "STABLE"

    def _parse_ts(self, value: Any) -> datetime:
        if not value:
            return datetime.min.replace(tzinfo=timezone.utc)
        try:
            ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts
        except ValueError:
            return datetime.min.replace(tzinfo=timezone.utc)
