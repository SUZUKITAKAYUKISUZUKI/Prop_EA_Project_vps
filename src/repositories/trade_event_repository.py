"""SQLite persistence for live trade_events stream."""
from __future__ import annotations

import json
from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager

LIVE_RUN_DESCRIPTION = "dropbox_live_stream"


class TradeEventRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns_connection = owns_connection or db is None
        self._live_run_id: int | None = None

    def close(self) -> None:
        if self._owns_connection:
            self._db.close()

    def get_or_create_live_run_id(self) -> int:
        if self._live_run_id is not None:
            return self._live_run_id
        row = self._db.query(
            "SELECT run_id FROM runs WHERE run_type='live' AND description=? LIMIT 1",
            (LIVE_RUN_DESCRIPTION,),
            one=True,
        )
        if row:
            self._live_run_id = int(row["run_id"])
            return self._live_run_id
        self._live_run_id = self._db.insert_run(
            "live",
            strategy="portfolio",
            description=LIVE_RUN_DESCRIPTION,
            parameters={"source": "dropbox", "stream": "trade_events"},
        )
        return self._live_run_id

    def insert_feature_from_event(self, event: dict[str, Any]) -> bool:
        features = event.get("features")
        if not isinstance(features, dict):
            features = {
                k: v
                for k, v in event.items()
                if k
                not in {
                    "event_id",
                    "timestamp",
                    "event_type",
                    "trade_id",
                    "strategy",
                    "symbol",
                    "features",
                }
            }
        run_id = self.get_or_create_live_run_id()
        feature_id = self._db.insert_feature(
            run_id,
            features,
            strategy=event.get("strategy"),
            source_key=f"event:{event['event_id']}",
            upsert=True,
        )
        return feature_id > 0

    def count_features(self) -> int:
        run_id = self.get_or_create_live_run_id()
        row = self._db.query("SELECT COUNT(*) AS c FROM features WHERE run_id=?", (run_id,), one=True)
        return int(row["c"]) if row else 0

    def insert_event(self, event: dict[str, Any], *, ignore_duplicates: bool = True) -> bool:
        event_id = str(event["event_id"])
        verb = "INSERT OR IGNORE" if ignore_duplicates else "INSERT"
        payload = {k: v for k, v in event.items() if k not in {"event_id", "timestamp", "event_type", "trade_id", "strategy", "symbol"}}
        cur = self._db.portfolio.execute(
            f"""
            {verb} INTO trade_events (
                event_id, timestamp, event_type, trade_id, strategy, symbol, payload_json, imported_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                str(event.get("timestamp", utc_now_iso())),
                str(event.get("event_type", "UNKNOWN")),
                event.get("trade_id"),
                event.get("strategy"),
                event.get("symbol"),
                json.dumps(payload, ensure_ascii=False, default=str),
                utc_now_iso(),
            ),
        )
        self._db.portfolio.commit()
        inserted = cur.rowcount > 0
        if inserted and event.get("event_type") == "FEATURE_SNAPSHOT":
            self.insert_feature_from_event(event)
        return inserted

    def insert_events_batch(self, events: list[dict[str, Any]], *, ignore_duplicates: bool = True) -> int:
        inserted = 0
        for event in events:
            if self.insert_event(event, ignore_duplicates=ignore_duplicates):
                inserted += 1
        return inserted

    def get_import_offset(self, filename: str) -> int:
        row = self._db.query(
            "SELECT last_offset FROM import_state WHERE filename=?",
            (filename,),
            one=True,
        )
        return int(row["last_offset"]) if row else 0

    def set_import_offset(self, filename: str, last_offset: int) -> None:
        self._db.portfolio.execute(
            """
            INSERT INTO import_state (filename, last_offset, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(filename) DO UPDATE SET
                last_offset=excluded.last_offset,
                updated_at=excluded.updated_at
            """,
            (filename, int(last_offset), utc_now_iso()),
        )
        self._db.portfolio.commit()

    def count_events(self) -> int:
        row = self._db.query("SELECT COUNT(*) AS c FROM trade_events", one=True)
        return int(row["c"]) if row else 0

    def get_recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT event_id, timestamp, event_type, trade_id, strategy, symbol, payload_json, imported_at
            FROM trade_events
            ORDER BY timestamp DESC, event_id DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        return [self._row_to_dict(r) for r in rows]

    def get_events_by_type(self, event_type: str, *, limit: int = 1000) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT event_id, timestamp, event_type, trade_id, strategy, symbol, payload_json, imported_at
            FROM trade_events
            WHERE event_type=?
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (event_type, int(limit)),
        )
        return [self._row_to_dict(r) for r in rows]

    def get_events_between(self, start: str, end: str) -> list[dict[str, Any]]:
        rows = self._db.query(
            """
            SELECT event_id, timestamp, event_type, trade_id, strategy, symbol, payload_json, imported_at
            FROM trade_events
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC
            """,
            (start, end),
        )
        return [self._row_to_dict(r) for r in rows]

    def get_feature_snapshots(self, *, limit: int = 100) -> list[dict[str, Any]]:
        return self.get_events_by_type("FEATURE_SNAPSHOT", limit=limit)

    def _row_to_dict(self, row: Any) -> dict[str, Any]:
        item = dict(row)
        try:
            extra = json.loads(item.pop("payload_json") or "{}")
        except json.JSONDecodeError:
            extra = {}
        item.update(extra)
        return item
