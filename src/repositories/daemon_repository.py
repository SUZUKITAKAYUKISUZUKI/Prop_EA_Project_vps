"""Persistence for import daemon file registry and heartbeat."""
from __future__ import annotations

from typing import Any

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager


class DaemonRepository:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns_connection = owns_connection or db is None
        self._ensure_status_row()

    def close(self) -> None:
        if self._owns_connection:
            self._db.close()

    def _ensure_status_row(self) -> None:
        row = self._db.query("SELECT id FROM daemon_status WHERE id=1", (), one=True)
        if row is None:
            self._db.portfolio.execute(
                "INSERT INTO daemon_status (id, last_seen, processed_files, processed_trades, last_error) "
                "VALUES (1, ?, 0, 0, NULL)",
                (utc_now_iso(),),
            )
            self._db.portfolio.commit()

    def is_file_imported(self, file_hash: str) -> bool:
        row = self._db.query(
            "SELECT file_hash FROM imported_files WHERE file_hash=?",
            (file_hash,),
            one=True,
        )
        return row is not None

    def register_imported_file(self, file_hash: str, filename: str) -> None:
        self._ensure_imported_files_schema()
        existing = self._db.query(
            "SELECT file_hash FROM imported_files WHERE file_hash=?",
            (file_hash,),
            one=True,
        )
        if existing:
            self._db.portfolio.execute(
                "UPDATE imported_files SET filename=?, imported_at=? WHERE file_hash=?",
                (filename, utc_now_iso(), file_hash),
            )
        else:
            self._db.portfolio.execute(
                "INSERT INTO imported_files (file_hash, filename, imported_at, storage_deleted_at) "
                "VALUES (?, ?, ?, NULL)",
                (file_hash, filename, utc_now_iso()),
            )
        self._db.portfolio.commit()

    def _ensure_imported_files_schema(self) -> None:
        rows = self._db.query("PRAGMA table_info(imported_files)", ())
        columns = {str(row["name"]) for row in rows}
        if "storage_deleted_at" not in columns:
            self._db.portfolio.execute(
                "ALTER TABLE imported_files ADD COLUMN storage_deleted_at TEXT"
            )
            self._db.portfolio.commit()

    def is_storage_deleted(self, file_hash: str) -> bool:
        self._ensure_imported_files_schema()
        row = self._db.query(
            "SELECT storage_deleted_at FROM imported_files WHERE file_hash=?",
            (file_hash,),
            one=True,
        )
        return bool(row and row["storage_deleted_at"])

    def mark_storage_deleted(self, file_hash: str) -> None:
        self._ensure_imported_files_schema()
        self._db.portfolio.execute(
            "UPDATE imported_files SET storage_deleted_at=? WHERE file_hash=?",
            (utc_now_iso(), file_hash),
        )
        self._db.portfolio.commit()

    def update_heartbeat(
        self,
        *,
        processed_files: int | None = None,
        processed_trades: int | None = None,
        last_error: str | None = None,
        increment_files: int = 0,
        increment_trades: int = 0,
    ) -> None:
        row = self.get_heartbeat()
        files = row["processed_files"] + increment_files
        trades = row["processed_trades"] + increment_trades
        if processed_files is not None:
            files = processed_files
        if processed_trades is not None:
            trades = processed_trades
        error_value = last_error if last_error is not None else row.get("last_error")
        if last_error == "":
            error_value = None
        self._db.portfolio.execute(
            """
            UPDATE daemon_status
            SET last_seen=?, processed_files=?, processed_trades=?, last_error=?
            WHERE id=1
            """,
            (utc_now_iso(), files, trades, error_value),
        )
        self._db.portfolio.commit()

    def get_heartbeat(self) -> dict[str, Any]:
        row = self._db.query("SELECT * FROM daemon_status WHERE id=1", (), one=True)
        if row is None:
            return {
                "last_seen": "",
                "processed_files": 0,
                "processed_trades": 0,
                "last_error": None,
            }
        return dict(row)
