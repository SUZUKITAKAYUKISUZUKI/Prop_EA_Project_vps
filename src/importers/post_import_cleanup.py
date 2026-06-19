"""Delete imported event files from staging and VPS Dropbox to save storage."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from src.importers.dropbox_cleanup import cleanup_after_import, delete_imported_file, is_today_event_file
from src.importers.vps_event_pull import _remote_file_size, _run_plink, _ssh_cfg, remote_events_dir
from src.repositories.trade_event_repository import TradeEventRepository
from src.runtime.logging_config import DropboxLoggingConfig

logger = logging.getLogger(__name__)


def storage_cleanup_enabled() -> bool:
    return os.environ.get("IMPORT_STORAGE_CLEANUP", "1").strip().lower() in {"1", "true", "yes", "on"}


def remote_delete_enabled() -> bool:
    flag = os.environ.get("VPS_EVENTS_DELETE_AFTER_IMPORT", "").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    if flag in {"1", "true", "yes", "on"}:
        return True
    return os.environ.get("VPS_EVENTS_PULL", "1").strip().lower() in {"1", "true", "yes", "on"}


def staging_cleanup_enabled() -> bool:
    return os.environ.get("IMPORT_CLEANUP_STAGING", "1").strip().lower() in {"1", "true", "yes", "on"}


def should_delete_remote_copy(filename: str) -> bool:
    """Keep today's live .jsonl on VPS while the bridge may still be appending."""
    if filename.endswith(".jsonl.gz"):
        return True
    if filename.endswith(".jsonl"):
        return not is_today_event_file(Path(filename))
    return False


def delete_remote_event_file(filename: str) -> bool:
    cfg = _ssh_cfg()
    if cfg is None:
        raise RuntimeError("VPS SSH credentials missing")
    remote_dir = remote_events_dir()
    remote_path = f"{remote_dir}\\{filename}"
    cmd = f'del /f /q "{remote_path}" 2>nul'
    result = _run_plink(cfg, cmd)
    if result.returncode != 0:
        stderr = (result.stderr or b"").decode("utf-8", errors="replace")[:200]
        raise RuntimeError(f"remote delete failed for {filename}: {stderr or result.returncode}")
    size = _remote_file_size(cfg, filename)
    if size is not None and size > 0:
        raise RuntimeError(f"remote file still present after delete: {filename}")
    logger.info("Deleted VPS Dropbox event file %s", filename)
    return True


def cleanup_imported_storage(
    path: Path,
    *,
    watch_dir: Path,
    staging_dir: Path,
    config: DropboxLoggingConfig,
    repo: TradeEventRepository,
    consumed_offset: int,
) -> dict[str, object]:
    """Remove a fully imported file from staging and/or VPS Dropbox."""
    result: dict[str, object] = {
        "filename": path.name,
        "staging_deleted": False,
        "remote_deleted": False,
        "local_deleted": False,
        "action": "none",
    }
    if not storage_cleanup_enabled():
        result["action"] = "disabled"
        return result

    filename = path.name
    file_exists = path.exists()
    if not file_exists and consumed_offset <= 0:
        result["action"] = "missing"
        return result

    if file_exists:
        file_size = path.stat().st_size
        if consumed_offset < file_size or file_size <= 0:
            result["action"] = "incomplete"
            return result

    resolved_staging = staging_dir.resolve()
    resolved_watch = watch_dir.resolve()
    resolved_path = path.resolve() if file_exists else None

    if file_exists and staging_cleanup_enabled() and resolved_path is not None:
        try:
            if resolved_path.is_relative_to(resolved_staging):
                if delete_imported_file(path):
                    result["staging_deleted"] = True
                    result["action"] = "delete_staging"
        except ValueError:
            pass

    if (
        file_exists
        and resolved_path is not None
        and config.cleanup.modify_synced_files
    ):
        try:
            if resolved_path.is_relative_to(resolved_watch):
                cleanup = cleanup_after_import(
                    path,
                    consumed_offset=consumed_offset,
                    repo=repo,
                    config=config,
                    cleanup_cfg=config.cleanup,
                    parsed=0,
                    imported=0,
                )
                result["local_deleted"] = bool(cleanup.get("applied"))
                result["action"] = cleanup.get("action", result["action"])
        except ValueError:
            pass

    if remote_delete_enabled() and should_delete_remote_copy(filename):
        try:
            if delete_remote_event_file(filename):
                result["remote_deleted"] = True
                if result["action"] == "none":
                    result["action"] = "delete_remote"
        except Exception as exc:
            logger.warning("Remote cleanup failed for %s: %s", filename, exc)
            result["remote_error"] = str(exc)

    return result
