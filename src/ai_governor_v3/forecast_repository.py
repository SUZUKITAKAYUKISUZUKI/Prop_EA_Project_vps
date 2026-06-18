"""Forecast persistence for AGE v3."""
from __future__ import annotations

import json
from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class ForecastRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns = owns_connection or db is None

    def close(self) -> None:
        if self._owns:
            self._db.close()

    def save_forecast(
        self,
        *,
        forecast_horizon: str,
        health_forecast: dict[str, Any],
        risk_forecast: dict[str, Any],
        recovery_probability: dict[str, Any],
        future_state: list[dict[str, Any]],
        confidence: float,
        recommendation_json: list[dict[str, Any]],
        profile_id: str | None = None,
    ) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO governor_forecasts (
                timestamp, forecast_horizon, health_forecast, risk_forecast,
                recovery_probability, future_state, confidence, recommendation_json, profile_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                forecast_horizon,
                _json(health_forecast),
                _json(risk_forecast),
                _json(recovery_probability),
                _json(future_state),
                confidence,
                _json(recommendation_json),
                profile_id,
            ),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def save_alert(self, alert_type: str, severity: str, details: dict[str, Any]) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO governor_predictive_alerts (timestamp, alert_type, severity, details_json)
            VALUES (?, ?, ?, ?)
            """,
            (utc_now_iso(), alert_type, severity, _json(details)),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def save_alerts(self, alerts: list[dict[str, Any]]) -> list[int]:
        ids = []
        for alert in alerts:
            ids.append(
                self.save_alert(
                    str(alert.get("alert_type") or "UNKNOWN"),
                    str(alert.get("severity") or "MEDIUM"),
                    alert.get("details_json") or {},
                )
            )
        return ids

    def list_forecasts(self, *, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._db.query(
            "SELECT * FROM governor_forecasts ORDER BY timestamp DESC, id DESC LIMIT ?",
            (limit,),
        )
        return [_parse_forecast_row(dict(r)) for r in rows]

    def list_alerts(self, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._db.query(
            "SELECT * FROM governor_predictive_alerts ORDER BY timestamp DESC, id DESC LIMIT ?",
            (limit,),
        )
        return [_parse_alert_row(dict(r)) for r in rows]


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _parse_forecast_row(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("health_forecast", "risk_forecast", "recovery_probability", "future_state", "recommendation_json"):
        if row.get(key):
            try:
                row[key] = json.loads(row[key])
            except (TypeError, json.JSONDecodeError):
                pass
    return row


def _parse_alert_row(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("details_json"):
        try:
            row["details_json"] = json.loads(row["details_json"])
        except (TypeError, json.JSONDecodeError):
            pass
    return row
