"""Remove or truncate Dropbox JSONL after successful SQLite import."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from src.repositories.trade_event_repository import TradeEventRepository
from src.runtime.logging_config import DropboxCleanupConfig, DropboxLoggingConfig

logger = logging.getLogger(__name__)


def _today_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def is_today_event_file(path: Path) -> bool:
    name = path.name
    if name.startswith("events_") and "_202" in name:
        token = name.replace("events_", "").split(".")[0]
        return token == _today_token()
    return False


def clear_import_state(repo: TradeEventRepository, filename: str) -> None:
    repo._db.portfolio.execute("DELETE FROM import_state WHERE filename=?", (filename,))
    repo._db.portfolio.commit()


def _atomic_write_tail(path: Path, tail: bytes) -> None:
    tmp = path.with_name(path.name + ".import_tmp")
    tmp.write_bytes(tail)
    os.replace(tmp, path)


def truncate_imported_prefix(path: Path, consumed_offset: int) -> bool:
    """Keep only unprocessed tail bytes; reset checkpoint externally."""
    if consumed_offset <= 0:
        return False
    if not path.exists():
        return True
    size_before = path.stat().st_size
    if consumed_offset >= size_before:
        path.unlink(missing_ok=True)
        return True
    raw = path.read_bytes()
    if path.stat().st_size != size_before:
        logger.warning("Skip truncate; file size changed during read: %s", path.name)
        return False
    tail = raw[consumed_offset:]
    _atomic_write_tail(path, tail)
    logger.info("Truncated imported prefix from %s (%d -> %d bytes)", path.name, size_before, len(tail))
    return True


def delete_imported_file(path: Path) -> bool:
    if not path.exists():
        return True
    path.unlink(missing_ok=True)
    logger.info("Deleted imported Dropbox file: %s", path.name)
    return True


def cleanup_after_import(
    path: Path,
    *,
    consumed_offset: int,
    repo: TradeEventRepository,
    config: DropboxLoggingConfig,
    cleanup_cfg: DropboxCleanupConfig,
    parsed: int,
    imported: int = 0,
) -> dict[str, str | bool]:
    """Clear Dropbox storage for bytes already persisted in SQLite."""
    result: dict[str, str | bool] = {"action": "none", "applied": False}
    if not cleanup_cfg.enabled:
        return result
    if not cleanup_cfg.modify_synced_files:
        if consumed_offset > 0 and path.exists() and consumed_offset >= path.stat().st_size:
            return {"action": "checkpoint_only", "applied": False}
        return result
    if not path.exists():
        clear_import_state(repo, path.name)
        return {"action": "already_gone", "applied": True}

    filename = path.name
    file_size = path.stat().st_size
    fully_imported = consumed_offset >= file_size
    if parsed <= 0 and imported <= 0 and not fully_imported:
        return result
    if consumed_offset <= 0:
        return result

    is_gz = filename.endswith(".gz")
    is_today = is_today_event_file(path)

    if is_gz:
        if fully_imported and cleanup_cfg.delete_fully_imported:
            delete_imported_file(path)
            clear_import_state(repo, filename)
            return {"action": "delete_gz", "applied": True}
        return result

    if fully_imported and cleanup_cfg.delete_fully_imported and not is_today:
        delete_imported_file(path)
        clear_import_state(repo, filename)
        return {"action": "delete_archived", "applied": True}

    if fully_imported and cleanup_cfg.delete_fully_imported and is_today:
        delete_imported_file(path)
        clear_import_state(repo, filename)
        return {"action": "delete_today_complete", "applied": True}

    if cleanup_cfg.truncate_today_file and consumed_offset > 0:
        if truncate_imported_prefix(path, consumed_offset):
            repo.set_import_offset(filename, 0)
            return {"action": "truncate_prefix", "applied": True}

    return result
