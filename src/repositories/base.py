"""Shared repository utilities and default DB wiring."""
from __future__ import annotations

from pathlib import Path

from src.database.config_loader import load_database_config, resolve_project_paths
from src.database.db_manager import DatabaseManager

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def normalize_source_path(source: str | Path, *, project_root: Path | None = None) -> str:
    root = project_root or PROJECT_ROOT
    path = Path(source)
    if path.is_absolute():
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix()


def create_default_db_manager(*, connect: bool = True) -> DatabaseManager:
    cfg = resolve_project_paths(PROJECT_ROOT, load_database_config(PROJECT_ROOT / "config" / "database.yaml"))
    db = DatabaseManager(
        cfg["portfolio_path"],
        cfg["market_path"],
        journal_mode=cfg["journal_mode"],
        synchronous=cfg["synchronous"],
    )
    if connect:
        db.connect()
    return db
