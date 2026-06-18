"""Persistence for CACE v1.7 calibration and decision accuracy."""
from __future__ import annotations

import json
from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class CalibrationRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns = owns_connection or db is None

    def close(self) -> None:
        if self._owns:
            self._db.close()

    def save_decision(self, record: dict[str, Any]) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO decision_accuracy_history (
                decision_id, timestamp, profile_id, module, decision_type, confidence,
                predicted_benefit, actual_benefit, predicted_dd, actual_dd,
                prediction_error, accuracy_score, evaluation_date, evaluated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("decision_id"),
                record.get("timestamp") or utc_now_iso(),
                record.get("profile_id"),
                record.get("module"),
                record.get("decision_type"),
                record.get("confidence"),
                record.get("predicted_benefit"),
                record.get("actual_benefit"),
                record.get("predicted_dd"),
                record.get("actual_dd"),
                record.get("prediction_error"),
                record.get("accuracy_score"),
                record.get("evaluation_date"),
                1 if record.get("evaluated") else 0,
            ),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def update_decision(self, record: dict[str, Any]) -> None:
        self._db.portfolio.execute(
            """
            UPDATE decision_accuracy_history
            SET actual_benefit=?, actual_dd=?, prediction_error=?, accuracy_score=?, evaluated=1
            WHERE decision_id=?
            """,
            (
                record.get("actual_benefit"),
                record.get("actual_dd"),
                record.get("prediction_error"),
                record.get("accuracy_score"),
                record.get("decision_id"),
            ),
        )
        self._db.portfolio.commit()

    def load_decisions(self, *, profile_id: str, limit: int = 500) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT * FROM decision_accuracy_history
            WHERE profile_id=?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (profile_id, limit),
        )
        results = []
        for row in rows or []:
            item = dict(row)
            item["evaluated"] = bool(item.get("evaluated"))
            results.append(item)
        return results

    def save_calibration_snapshot(
        self,
        *,
        profile_id: str,
        calibration_score: float,
        calibration_category: str,
        decision_accuracy_score: float,
        reliability_trend: str,
        payload: dict[str, Any],
    ) -> int:
        cur = self._db.portfolio.execute(
            """
            INSERT INTO confidence_calibration_history (
                timestamp, profile_id, calibration_score, calibration_category,
                decision_accuracy_score, reliability_trend, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now_iso(),
                profile_id,
                calibration_score,
                calibration_category,
                decision_accuracy_score,
                reliability_trend,
                json.dumps(payload, ensure_ascii=False),
            ),
        )
        self._db.portfolio.commit()
        return int(cur.lastrowid)

    def save_learning_notes(self, *, profile_id: str, notes: list[str]) -> None:
        for note in notes:
            self._db.portfolio.execute(
                """
                INSERT INTO confidence_learning_notes (timestamp, profile_id, note)
                VALUES (?, ?, ?)
                """,
                (utc_now_iso(), profile_id, note),
            )
        self._db.portfolio.commit()

    def load_latest_calibration(self, *, profile_id: str) -> dict[str, Any] | None:
        row = self._db.query(
            """
            SELECT * FROM confidence_calibration_history
            WHERE profile_id=?
            ORDER BY timestamp DESC, id DESC
            LIMIT 1
            """,
            (profile_id,),
            one=True,
        )
        if not row:
            return None
        item = dict(row)
        if item.get("payload_json"):
            try:
                item["payload_json"] = json.loads(item["payload_json"])
            except (TypeError, json.JSONDecodeError):
                pass
        return item

    def load_learning_notes(self, *, profile_id: str, limit: int = 20) -> list[str]:
        rows = self._db.query(
            """
            SELECT note FROM confidence_learning_notes
            WHERE profile_id=?
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (profile_id, limit),
        )
        return [str(dict(row).get("note") or "") for row in rows or [] if dict(row).get("note")]
