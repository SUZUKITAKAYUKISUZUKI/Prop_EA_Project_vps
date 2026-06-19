"""Dropbox JSONL → SQLite import engine with byte-offset checkpoint recovery."""
from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any

from src.importers.dropbox_cleanup import cleanup_after_import
from src.repositories.trade_event_repository import TradeEventRepository
from src.runtime.dropbox_paths import resolve_watch_dir
from src.runtime.logging_config import DropboxLoggingConfig, load_dropbox_logging_config

logger = logging.getLogger(__name__)


def _open_event_file(path: Path):
    if path.suffix == ".gz" or path.name.endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def parse_jsonl_line(line: str) -> dict[str, Any] | None:
    text = line.strip()
    if not text:
        return None
    try:
        event = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Skipping invalid JSONL line: %s", text[:120])
        return None
    if "event_id" not in event:
        logger.warning("Skipping event without event_id")
        return None
    return event


class DropboxEventImporter:
    def __init__(
        self,
        config: DropboxLoggingConfig | None = None,
        repo: TradeEventRepository | None = None,
    ) -> None:
        self.config = config or load_dropbox_logging_config()
        self.repo = repo or TradeEventRepository()

    def discover_files(self, *extra_dirs: Path) -> list[Path]:
        dirs = [resolve_watch_dir(self.config)]
        for path in extra_dirs:
            if path not in dirs:
                dirs.append(path)
        files: list[Path] = []
        seen: set[str] = set()
        for watch in dirs:
            if not watch.exists():
                continue
            for path in sorted(watch.glob("events_*.jsonl")):
                key = path.name.lower()
                if key not in seen:
                    seen.add(key)
                    files.append(path)
            for path in sorted(watch.glob("events_*.jsonl.gz")):
                key = path.name.lower()
                if key not in seen:
                    seen.add(key)
                    files.append(path)
        return files

    def import_file(self, path: Path) -> dict[str, Any]:
        filename = path.name
        offset = self.repo.get_import_offset(filename)
        if not path.exists():
            return {"parsed": 0, "imported": 0, "duplicates": 0, "offset": 0, "file_size": 0, "cleanup": {"action": "missing", "applied": False}}
        file_size = path.stat().st_size
        if offset > file_size:
            offset = 0

        is_gz = path.suffix == ".gz" or path.name.endswith(".jsonl.gz")
        if is_gz:
            if offset >= file_size:
                cleanup = cleanup_after_import(
                    path,
                    consumed_offset=offset,
                    repo=self.repo,
                    config=self.config,
                    cleanup_cfg=self.config.cleanup,
                    parsed=0,
                    imported=0,
                )
                return {
                    "parsed": 0,
                    "imported": 0,
                    "duplicates": 0,
                    "offset": offset,
                    "file_size": file_size,
                    "cleanup": cleanup,
                }
            offset = 0

        if not is_gz and offset >= file_size and file_size > 0:
            cleanup = cleanup_after_import(
                path,
                consumed_offset=offset,
                repo=self.repo,
                config=self.config,
                cleanup_cfg=self.config.cleanup,
                parsed=0,
                imported=0,
            )
            return {
                "parsed": 0,
                "imported": 0,
                "duplicates": 0,
                "offset": 0 if not path.exists() else offset,
                "file_size": path.stat().st_size if path.exists() else 0,
                "cleanup": cleanup,
            }

        imported = 0
        duplicates = 0
        parsed = 0
        new_offset = offset

        with _open_event_file(path) as fh:
            fh.seek(offset)
            batch: list[dict[str, Any]] = []
            while True:
                line = fh.readline()
                if not line:
                    new_offset = path.stat().st_size if path.name.endswith(".gz") else fh.tell()
                    break
                event = parse_jsonl_line(line)
                if event is None:
                    new_offset = fh.tell()
                    continue
                parsed += 1
                batch.append(event)
                if len(batch) >= self.config.batch_size:
                    inserted = self.repo.insert_events_batch(batch, ignore_duplicates=self.config.dedupe)
                    imported += inserted
                    duplicates += len(batch) - inserted
                    batch.clear()
                    new_offset = fh.tell()
                    self.repo.set_import_offset(filename, new_offset)
            if batch:
                inserted = self.repo.insert_events_batch(batch, ignore_duplicates=self.config.dedupe)
                imported += inserted
                duplicates += len(batch) - inserted
                new_offset = fh.tell()
                self.repo.set_import_offset(filename, new_offset)

        cleanup = cleanup_after_import(
            path,
            consumed_offset=new_offset,
            repo=self.repo,
            config=self.config,
            cleanup_cfg=self.config.cleanup,
            parsed=parsed,
            imported=imported,
        )

        return {
            "parsed": parsed,
            "imported": imported,
            "duplicates": duplicates,
            "offset": new_offset if path.exists() else 0,
            "file_size": path.stat().st_size if path.exists() else 0,
            "cleanup": cleanup,
        }

    def import_all(self, *extra_dirs: Path) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "files": 0,
            "parsed": 0,
            "imported": 0,
            "duplicates": 0,
            "file_results": {},
        }
        for path in self.discover_files(*extra_dirs):
            result = self.import_file(path)
            summary["files"] += 1
            summary["parsed"] += result["parsed"]
            summary["imported"] += result["imported"]
            summary["duplicates"] += result["duplicates"]
            summary["file_results"][path.name] = result
        return summary
