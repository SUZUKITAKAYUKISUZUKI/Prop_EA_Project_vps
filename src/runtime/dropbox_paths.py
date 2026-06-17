"""Resolve Dropbox event watch directories on local Windows machines."""
from __future__ import annotations

import os
from pathlib import Path

from src.runtime.logging_config import DropboxLoggingConfig, load_dropbox_logging_config

EVENTS_SUBPATH = Path("PortfolioOS") / "events"


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    return Path(raw) if raw else None


def candidate_dropbox_roots() -> list[Path]:
    roots: list[Path] = []
    for value in (
        os.environ.get("DROPBOX_ROOT", "").strip(),
        "C:/Dropbox",
        str(Path.home() / "Dropbox"),
    ):
        if not value:
            continue
        root = Path(value)
        if root.exists() and root not in roots:
            roots.append(root)
    return roots


def candidate_events_dirs() -> list[Path]:
    seen: set[str] = set()
    dirs: list[Path] = []
    for root in candidate_dropbox_roots():
        path = root / EVENTS_SUBPATH
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            dirs.append(path)
    default = Path("C:/Dropbox/PortfolioOS/events")
    key = str(default).lower()
    if key not in seen:
        dirs.append(default)
    return dirs


def resolve_watch_dir(config: DropboxLoggingConfig | None = None) -> Path:
    for env_name in (
        "DROPBOX_WATCH_DIR",
        "DROPBOX_EVENTS_WATCH_DIR",
        "DROPBOX_LIVE_LOGS_DIR",
        "DROPBOX_EVENTS_DIR",
    ):
        env_path = _env_path(env_name)
        if env_path is not None:
            return env_path

    cfg = config or load_dropbox_logging_config()
    if cfg.watch_dir.exists():
        return cfg.watch_dir

    for candidate in candidate_events_dirs():
        if candidate.exists():
            return candidate

    home_events = Path.home() / "Dropbox" / EVENTS_SUBPATH
    if home_events.parent.parent.exists():
        return home_events
    return cfg.watch_dir


def ensure_watch_dir(path: Path | None = None) -> Path:
    watch = path or resolve_watch_dir()
    watch.mkdir(parents=True, exist_ok=True)
    return watch
