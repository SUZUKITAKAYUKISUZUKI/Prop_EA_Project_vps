"""Confidence trend analysis for CACE v1.5."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


class ConfidenceTrendEngine:
    WINDOWS_DAYS = (7, 30, 90)

    def analyze(self, history: list[dict[str, Any]], *, current_confidence: float) -> dict[str, Any]:
        windows: dict[str, dict[str, Any]] = {}
        primary_strength = 0.0
        primary_direction = "FLAT"

        for days in self.WINDOWS_DAYS:
            window_rows = self._rows_in_window(history, days)
            windows[f"{days}d"] = self._window_metrics(window_rows, current_confidence)

        primary = windows.get("30d") or windows.get("7d") or {}
        primary_strength = float(primary.get("trend_strength") or 0)
        primary_direction = str(primary.get("trend_direction") or "FLAT")

        return {
            "confidence": round(current_confidence, 1),
            "trend": self._simplified_trend(primary_direction),
            "trend_direction": primary_direction,
            "trend_strength": round(primary_strength, 1),
            "trend_category": self._trend_category(primary_strength),
            "windows": windows,
            "evolution": self._evolution_label(primary_strength),
        }

    def _rows_in_window(self, history: list[dict[str, Any]], days: int) -> list[dict[str, Any]]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        rows: list[dict[str, Any]] = []
        for row in history:
            ts = self._parse_ts(str(row.get("timestamp") or ""))
            if ts and ts >= cutoff:
                rows.append(row)
        return rows

    def _window_metrics(self, rows: list[dict[str, Any]], current: float) -> dict[str, Any]:
        if not rows:
            return {
                "sample_count": 0,
                "trend_strength": 0.0,
                "trend_direction": "FLAT",
                "trend_category": "FLAT",
            }
        scores = [float(r.get("confidence") or 0) for r in rows]
        oldest = scores[-1] if scores else current
        strength = round(current - oldest, 1)
        direction = self._direction_from_strength(strength)
        return {
            "sample_count": len(scores),
            "start_confidence": round(oldest, 1),
            "end_confidence": round(current, 1),
            "trend_strength": strength,
            "trend_direction": direction,
            "trend_category": self._trend_category(strength),
        }

    def _direction_from_strength(self, strength: float) -> str:
        return self._trend_category(strength)

    def _trend_category(self, strength: float) -> str:
        if strength > 5:
            return "STRONG_UP"
        if strength > 1:
            return "UP"
        if strength >= -1:
            return "FLAT"
        if strength >= -5:
            return "DOWN"
        return "STRONG_DOWN"

    def _simplified_trend(self, direction: str) -> str:
        if direction in {"STRONG_UP", "UP"}:
            return "UP"
        if direction in {"STRONG_DOWN", "DOWN"}:
            return "DOWN"
        return "FLAT"

    def _evolution(self, strength: float) -> str:
        if strength > 1:
            return "IMPROVING"
        if strength < -1:
            return "DETERIORATING"
        return "STABLE"

    def _evolution_label(self, strength: float) -> str:
        return self._evolution(strength)

    def _parse_ts(self, value: str) -> datetime | None:
        if not value:
            return None
        try:
            normalized = value.replace("Z", "+00:00")
            ts = datetime.fromisoformat(normalized)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts
        except ValueError:
            return None
