"""PortfolioOS import daemon — Dropbox JSONL → SQLite with file registry and heartbeat."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from src.importers.dropbox_event_importer import DropboxEventImporter
from src.importers.post_import_cleanup import cleanup_imported_storage
from src.importers.vps_event_pull import VpsEventPuller, staging_dir
from src.importers.vps_m5_exporter import VpsM5Exporter
from src.repositories.daemon_repository import DaemonRepository
from src.runtime.logging_config import DropboxLoggingConfig, load_dropbox_logging_config, require_consumer
from src.runtime.daemon_env import vps_m5_export_enabled, vps_pull_enabled

from src.runtime.dropbox_paths import resolve_watch_dir

logger = logging.getLogger("portfolioos.import_daemon")

HEARTBEAT_INTERVAL_SEC = int(os.environ.get("DAEMON_HEARTBEAT_SEC", "60"))
DB_RETRY_ATTEMPTS = int(os.environ.get("DAEMON_DB_RETRY_ATTEMPTS", "5"))
DB_RETRY_DELAY_SEC = float(os.environ.get("DAEMON_DB_RETRY_DELAY_SEC", "1.0"))


def run_with_sqlite_retry(
    operation,
    *,
    attempts: int | None = None,
    delay_sec: float | None = None,
    logger: logging.Logger | None = None,
):
    """Retry SQLite operations when portfolio_os.db is contended (dashboard, imports)."""
    max_attempts = DB_RETRY_ATTEMPTS if attempts is None else attempts
    delay = DB_RETRY_DELAY_SEC if delay_sec is None else delay_sec
    last_error: sqlite3.OperationalError | None = None
    for attempt in range(max_attempts):
        try:
            return operation()
        except sqlite3.OperationalError as exc:
            last_error = exc
            if "locked" not in str(exc).lower():
                raise
            if logger is not None:
                logger.warning(
                    "Database locked (attempt %d/%d)",
                    attempt + 1,
                    max_attempts,
                )
            time.sleep(delay * (attempt + 1))
    if last_error is not None:
        raise last_error
    return None


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
        self.staging_dir = staging_dir()
        self.importer = importer or DropboxEventImporter(self.config)
        self.daemon_repo = daemon_repo or DaemonRepository()
        self._vps_puller = VpsEventPuller(staging=self.staging_dir)
        self._m5_exporter = VpsM5Exporter()
        self._owns_repo = daemon_repo is None
        self._last_heartbeat = 0.0
        self._total_files = 0
        self._total_trades = 0
        self._cycle_lock = threading.Lock()
        self._empty_watch_cycles = 0
        self._hash_skip_logged: set[str] = set()

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

    def _maybe_cleanup_imported_storage(self, path: Path) -> dict[str, Any] | None:
        filename = path.name
        offset = self.importer.repo.get_import_offset(filename)
        target = path if path.exists() else self.staging_dir / filename
        if not target.exists():
            target = self.watch_dir / filename
        if target.exists():
            file_size = target.stat().st_size
        else:
            file_size = offset
        if offset < file_size or file_size <= 0:
            return None

        file_hash = ""
        if target.exists():
            try:
                file_hash = sha256_file(target)
            except OSError:
                return None
        if not file_hash:
            imported_row = self.daemon_repo._db.query(
                "SELECT file_hash FROM imported_files WHERE filename=? ORDER BY imported_at DESC LIMIT 1",
                (filename,),
                one=True,
            )
            if imported_row:
                file_hash = str(imported_row["file_hash"])
        if not file_hash:
            return None
        if self.daemon_repo.is_storage_deleted(file_hash):
            return None

        try:
            cleanup_path = target if target.exists() else path
            result = cleanup_imported_storage(
                cleanup_path,
                watch_dir=self.watch_dir,
                staging_dir=self.staging_dir,
                config=self.config,
                repo=self.importer.repo,
                consumed_offset=offset,
            )
        except Exception as exc:
            logger.warning("Storage cleanup failed for %s: %s", filename, exc)
            return None

        deleted = any(
            result.get(key)
            for key in ("staging_deleted", "remote_deleted", "local_deleted")
        )
        if deleted:
            run_with_sqlite_retry(
                lambda: self.daemon_repo.mark_storage_deleted(file_hash),
                logger=logger,
            )
            logger.info(
                "Storage cleanup %s: staging=%s remote=%s local=%s",
                filename,
                result.get("staging_deleted"),
                result.get("remote_deleted"),
                result.get("local_deleted"),
            )
        return result

    def import_file(self, path: Path) -> dict[str, Any]:
        if self.should_skip_fully_imported_file(path):
            self._maybe_cleanup_imported_storage(path)
            if path.name not in self._hash_skip_logged:
                logger.info("Skip already-imported file (hash) %s", path.name)
                self._hash_skip_logged.add(path.name)
            else:
                logger.debug("Skip already-imported file (hash) %s", path.name)
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
                logger.warning(
                    "Database locked importing %s (attempt %d/%d)",
                    path.name,
                    attempt + 1,
                    DB_RETRY_ATTEMPTS,
                )
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
            run_with_sqlite_retry(
                lambda: self.daemon_repo.register_imported_file(file_hash, path.name),
                logger=logger,
            )
            self._maybe_cleanup_imported_storage(path)
        except OSError as exc:
            logger.warning("Could not register hash for %s: %s", path.name, exc)

    def _import_source_dirs(self) -> list[Path]:
        dirs = [self.watch_dir]
        if self.staging_dir not in dirs:
            dirs.append(self.staging_dir)
        return dirs

    def _discover_import_paths(self) -> list[Path]:
        extra = [self.staging_dir] if self.staging_dir.resolve() != self.watch_dir.resolve() else []
        return self.importer.discover_files(*extra)

    def _maybe_pull_vps_events(self) -> dict[str, Any] | None:
        if not vps_pull_enabled():
            return None
        watch_has_files = any(self.watch_dir.glob("events_*.jsonl*")) if self.watch_dir.exists() else False
        return self._vps_puller.maybe_pull(force=not watch_has_files)

    def _maybe_export_m5(self) -> dict[str, Any] | None:
        if not vps_m5_export_enabled():
            return None
        return self._m5_exporter.maybe_export()

    def import_new_files(self) -> dict[str, Any]:
        m5_summary = self._maybe_export_m5()
        pull_summary = self._maybe_pull_vps_events()
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
            "staging_dir": str(self.staging_dir),
        }
        if m5_summary is not None:
            summary["m5_export"] = m5_summary
        if pull_summary is not None:
            summary["vps_pull"] = pull_summary

        paths = self._discover_import_paths()

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

        if not paths:
            self._empty_watch_cycles += 1
            if self._empty_watch_cycles == 6:
                logger.warning(
                    "No events_*.jsonl files in watch=%s or staging=%s — "
                    "check Dropbox sync or VPS_EVENTS_PULL / SSH credentials",
                    watch,
                    self.staging_dir,
                )
        else:
            self._empty_watch_cycles = 0

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
            run_with_sqlite_retry(
                lambda: self.daemon_repo.update_heartbeat(
                    processed_files=self._total_files,
                    processed_trades=self._total_trades,
                    last_error=last_error if last_error else "",
                ),
                logger=logger,
            )
        except sqlite3.OperationalError as exc:
            logger.error("Heartbeat update failed after retries: %s", exc)

    def run_import_cycle(self) -> dict[str, Any]:
        with self._cycle_lock:
            return self._run_import_cycle_locked()

    def _run_import_cycle_locked(self) -> dict[str, Any]:
        summary = self.import_new_files()
        imported = int(summary.get("imported", 0))
        duplicates = int(summary.get("duplicates", 0))
        skipped = int(summary.get("skipped", 0))
        if imported:
            logger.info(
                "Imported %d events from %d file(s) (%d duplicates, %d hash-skipped)",
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
