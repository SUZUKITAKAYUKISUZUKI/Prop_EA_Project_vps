"""Persistence for AI CIO v1."""

from __future__ import annotations



import json

from typing import Any



from src.database.db_manager import DatabaseManager, utc_now_iso

from src.repositories.base import create_default_db_manager





def _json_dumps(value: Any) -> str | None:

    if value is None:

        return None

    return json.dumps(value, ensure_ascii=False)





def _json_loads(raw: Any) -> Any:

    if not raw:

        return None

    try:

        return json.loads(raw)

    except (TypeError, json.JSONDecodeError):

        return None





class CioRepository:

    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:

        self._db = db or create_default_db_manager()

        self._owns = owns_connection or db is None



    def close(self) -> None:

        if self._owns:

            self._db.close()



    def save_report(self, *, profile_id: str, report: dict[str, Any]) -> int:

        components = report.get("cio_score_components")

        outcome = report.get("actual_outcome")

        cur = self._db.portfolio.execute(

            """

            INSERT INTO cio_reports (

                timestamp, profile_id, cio_score, cio_opinion,

                cio_score_components_json, actual_outcome_json, payload_json

            ) VALUES (?, ?, ?, ?, ?, ?, ?)

            """,

            (

                utc_now_iso(),

                profile_id,

                report.get("cio_score"),

                report.get("cio_opinion"),

                _json_dumps(components),

                _json_dumps(outcome),

                json.dumps(report, ensure_ascii=False),

            ),

        )

        self._db.portfolio.commit()

        return int(cur.lastrowid)



    def save_opinion(self, *, profile_id: str, opinion: str, cio_score: float) -> int:

        cur = self._db.portfolio.execute(

            """

            INSERT INTO cio_opinions (timestamp, profile_id, cio_opinion, cio_score)

            VALUES (?, ?, ?, ?)

            """,

            (utc_now_iso(), profile_id, opinion, cio_score),

        )

        self._db.portfolio.commit()

        return int(cur.lastrowid)



    def save_recommendations(self, *, profile_id: str, recommendations: list[dict[str, Any]]) -> int:

        count = 0

        ts = utc_now_iso()

        for rec in recommendations:

            self._db.portfolio.execute(

                """

                INSERT INTO cio_recommendations (

                    timestamp, profile_id, category, priority, action,

                    description, confidence, requires_approval, payload_json

                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)

                """,

                (

                    ts,

                    profile_id,

                    rec.get("category"),

                    rec.get("priority"),

                    rec.get("action"),

                    rec.get("description"),

                    rec.get("confidence"),

                    1 if rec.get("requires_approval", True) else 0,

                    json.dumps(rec, ensure_ascii=False),

                ),

            )

            count += 1

        self._db.portfolio.commit()

        return count



    def _hydrate_report_row(self, row: dict[str, Any]) -> dict[str, Any]:

        item = dict(row)

        report: dict[str, Any] | None = None

        if item.get("payload_json"):

            loaded = _json_loads(item["payload_json"])

            if isinstance(loaded, dict):

                report = loaded



        if report is None:

            report = {

                "profile_id": item.get("profile_id"),

                "cio_score": item.get("cio_score"),

                "cio_opinion": item.get("cio_opinion"),

            }



        components = _json_loads(item.get("cio_score_components_json"))

        if components is not None:

            report["cio_score_components"] = components

        elif "cio_score_components" not in report:

            report["cio_score_components"] = None



        outcome = _json_loads(item.get("actual_outcome_json"))

        if outcome is not None:

            report["actual_outcome"] = outcome

        elif "actual_outcome" not in report:

            report["actual_outcome"] = None



        report["report_id"] = item.get("id")

        report["report_timestamp"] = item.get("timestamp")

        return report



    def load_report_by_id(self, *, report_id: int) -> dict[str, Any] | None:

        row = self._db.query(

            "SELECT * FROM cio_reports WHERE id=? LIMIT 1",

            (report_id,),

            one=True,

        )

        if not row:

            return None

        return self._hydrate_report_row(dict(row))



    def load_latest_report(self, *, profile_id: str) -> dict[str, Any] | None:

        row = self._db.query(

            """

            SELECT * FROM cio_reports

            WHERE profile_id=?

            ORDER BY timestamp DESC, id DESC

            LIMIT 1

            """,

            (profile_id,),

            one=True,

        )

        if not row:

            return None

        return self._hydrate_report_row(dict(row))



    def update_actual_outcome(

        self,

        *,

        report_id: int,

        actual_outcome: dict[str, Any],

    ) -> bool:

        row = self._db.query(

            "SELECT id, payload_json FROM cio_reports WHERE id=? LIMIT 1",

            (report_id,),

            one=True,

        )

        if not row:

            return False



        payload: dict[str, Any] = _json_loads(row["payload_json"]) or {}

        payload["actual_outcome"] = actual_outcome

        self._db.portfolio.execute(

            """

            UPDATE cio_reports

            SET actual_outcome_json=?, payload_json=?

            WHERE id=?

            """,

            (

                _json_dumps(actual_outcome),

                json.dumps(payload, ensure_ascii=False),

                report_id,

            ),

        )

        self._db.portfolio.commit()

        return True



    def record_actual_outcome_for_latest(

        self,

        *,

        profile_id: str,

        actual_outcome: dict[str, Any],

    ) -> int | None:

        latest = self.load_latest_report(profile_id=profile_id)

        if not latest or latest.get("report_id") is None:

            return None

        report_id = int(latest["report_id"])

        if not self.update_actual_outcome(report_id=report_id, actual_outcome=actual_outcome):

            return None

        return report_id



    def load_opinion_history(self, *, profile_id: str, limit: int = 30) -> list[dict[str, Any]]:

        rows = self._db.query(

            """

            SELECT * FROM cio_opinions

            WHERE profile_id=?

            ORDER BY timestamp DESC, id DESC

            LIMIT ?

            """,

            (profile_id, limit),

        )

        return [dict(row) for row in rows or []]



    def load_recommendation_history(self, *, profile_id: str, limit: int = 50) -> list[dict[str, Any]]:

        rows = self._db.query(

            """

            SELECT * FROM cio_recommendations

            WHERE profile_id=?

            ORDER BY timestamp DESC, id DESC

            LIMIT ?

            """,

            (profile_id, limit),

        )

        return [dict(row) for row in rows or []]

