"""PortfolioOS import daemon — Dropbox JSONL → SQLite with file registry and heartbeat."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from src.importers.dropbox_event_importer import DropboxEventImporter
from src.repositories.daemon_repository import DaemonRepository
from src.runtime.logging_config import DropboxLoggingConfig, load_dropbox_logging_config, require_consumer

from src.runtime.dropbox_paths import resolve_watch_dir

logger = logging.getLogger("portfolioos.import_daemon")

HEARTBEAT_INTERVAL_SEC = int(os.environ.get("DAEMON_HEARTBEAT_SEC", "60"))
DB_RETRY_ATTEMPTS = int(os.environ.get("DAEMON_DB_RETRY_ATTEMPTS", "5"))
DB_RETRY_DELAY_SEC = float(os.environ.get("DAEMON_DB_RETRY_DELAY_SEC", "1.0"))


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class ImportDaemonService:
    def __init__(
        self,
        config: DropboxLoggingConfig | None = None,
        importer: DropboxEventImporter | None = None,
        daemon_repo: DaemonRepository | None = None,
    ) -> None:
        self.config = config or load_dropbox_logging_config()
        require_consumer(self.config, component="ImportDaemonService")
        self.watch_dir = resolve_watch_dir(self.config)
        self.importer = importer or DropboxEventImporter(self.config)
        self.daemon_repo = daemon_repo or DaemonRepository()
        self._owns_repo = daemon_repo is None
        self._last_heartbeat = 0.0
        self._total_files = 0
        self._total_trades = 0

    def close(self) -> None:
        if self._owns_repo:
            self.daemon_repo.close()
        self.importer.repo.close()

    def should_skip_fully_imported_file(self, path: Path) -> bool:
        if not path.exists():
            return True
        filename = path.name
        offset = self.importer.repo.get_import_offset(filename)
        file_size = path.stat().st_size
        if file_size <= 0:
            return False
        if offset < file_size:
            return False
        try:
            file_hash = sha256_file(path)
        except OSError as exc:
            logger.warning("Unable to hash %s: %s", path.name, exc)
            return False
        return self.daemon_repo.is_file_imported(file_hash)

    def import_file(self, path: Path) -> dict[str, Any]:
        if self.should_skip_fully_imported_file(path):
            logger.info("Skip already-imported file (hash) %s", path.name)
            return {
                "parsed": 0,
                "imported": 0,
                "duplicates": 0,
                "skipped_hash": True,
                "filename": path.name,
            }

        last_error: Exception | None = None
        for attempt in range(DB_RETRY_ATTEMPTS):
            try:
                result = self.importer.import_file(path)
                result["skipped_hash"] = False
                result["filename"] = path.name
                self._register_if_complete(path, result)
                return result
            except sqlite3.OperationalError as exc:
                last_error = exc
                if "locked" not in str(exc).lower():
                    raise
                logger.warning("Database locked importing %s (attempt %d/%d)", path.name, attempt + 1, DB_RETRY_ATTEMPTS)
                time.sleep(DB_RETRY_DELAY_SEC * (attempt + 1))
            except (OSError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.error("Corrupt or unreadable file %s: %s", path.name, exc)
                return {
                    "parsed": 0,
                    "imported": 0,
                    "duplicates": 0,
                    "error": str(exc),
                    "filename": path.name,
                }

        if last_error is not None:
            raise last_error
        return {"parsed": 0, "imported": 0, "duplicates": 0, "filename": path.name}

    def _register_if_complete(self, path: Path, result: dict[str, Any]) -> None:
        if not path.exists():
            return
        offset = int(result.get("offset", 0))
        file_size = int(result.get("file_size", path.stat().st_size))
        if offset < file_size:
            return
        try:
            file_hash = sha256_file(path)
            self.daemon_repo.register_imported_file(file_hash, path.name)
        except OSError as exc:
            logger.warning("Could not register hash for %s: %s", path.name, exc)

    def import_new_files(self) -> dict[str, Any]:
        watch = self.watch_dir
        if not watch.exists():
            logger.warning("Watch directory missing: %s", watch)
            return {
                "files": 0,
                "parsed": 0,
                "imported": 0,
                "duplicates": 0,
                "skipped": 0,
                "errors": 1,
                "watch_dir": str(watch),
                "error": "watch_dir_missing",
            }

        summary: dict[str, Any] = {
            "files": 0,
            "parsed": 0,
            "imported": 0,
            "duplicates": 0,
            "skipped": 0,
            "errors": 0,
            "file_results": {},
            "watch_dir": str(watch),
        }

        paths = sorted(watch.glob("events_*.jsonl"))
        paths.extend(sorted(watch.glob("events_*.jsonl.gz")))

        for path in paths:
            try:
                result = self.import_file(path)
            except Exception as exc:
                logger.exception("Import failed for %s", path.name)
                summary["errors"] += 1
                summary["file_results"][path.name] = {"error": str(exc)}
                continue

            summary["files"] += 1
            if result.get("skipped_hash"):
                summary["skipped"] += 1
                continue
            summary["parsed"] += int(result.get("parsed", 0))
            summary["imported"] += int(result.get("imported", 0))
            summary["duplicates"] += int(result.get("duplicates", 0))
            if result.get("error"):
                summary["errors"] += 1
            summary["file_results"][path.name] = result

        return summary

    def maybe_update_heartbeat(self, summary: dict[str, Any] | None = None, *, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_heartbeat < HEARTBEAT_INTERVAL_SEC:
            return
        self._last_heartbeat = now
        if summary:
            self._total_files += int(summary.get("files", 0))
            self._total_trades += int(summary.get("imported", 0))
        last_error = None
        if summary and summary.get("errors"):
            last_error = summary.get("error") or "import_errors"
        try:
            self.daemon_repo.update_heartbeat(
                processed_files=self._total_files,
                processed_trades=self._total_trades,
                last_error=last_error if last_error else "",
            )
        except sqlite3.OperationalError as exc:
            logger.error("Heartbeat update failed: %s", exc)

    def run_import_cycle(self) -> dict[str, Any]:
        summary = self.import_new_files()
        imported = int(summary.get("imported", 0))
        duplicates = int(summary.get("duplicates", 0))
        skipped = int(summary.get("skipped", 0))
        if imported:
            logger.info(
                "Imported %d trades from %d files (%d duplicates, %d hash-skipped)",
                imported,
                summary.get("files", 0),
                duplicates,
                skipped,
            )
        elif summary.get("errors"):
            logger.error("Import cycle completed with errors: %s", summary.get("error", summary))
        self.maybe_update_heartbeat(summary, force=bool(summary.get("errors")))
        return summary


def run_import_cycle(service: ImportDaemonService | None = None) -> dict:
    owned = service is None
    svc = service or ImportDaemonService()
    try:
        return svc.run_import_cycle()
    finally:
        if owned:
            svc.close()
