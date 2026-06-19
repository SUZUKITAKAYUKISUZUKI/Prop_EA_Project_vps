"""Load local env files for the PortfolioOS import daemon."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_env_file(path: Path, *, override: bool = False) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value


def load_daemon_env(project_root: Path | None = None) -> None:
    """Load import-daemon and optional VPS SSH env into ``os.environ``."""
    root = project_root or PROJECT_ROOT
    _load_env_file(root / "local_import_daemon.env")
    # Consumer role must win over producer default in dropbox_logging.yaml.
    if not os.environ.get("DROPBOX_DATA_FLOW_ROLE", "").strip():
        os.environ["DROPBOX_DATA_FLOW_ROLE"] = "consumer"
    # Optional VPS pull credentials (same file as dashboard SSH tunnel).
    _load_env_file(root / "local_dashboard.env")


def vps_pull_enabled() -> bool:
    flag = os.environ.get("VPS_EVENTS_PULL", "1").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    host = os.environ.get("VPS_HOST", "").strip()
    user = os.environ.get("VPS_SSH_USER", "").strip()
    password = os.environ.get("VPS_SSH_PASSWORD", "").strip()
    return bool(host and user and password)


def vps_m5_export_enabled() -> bool:
    from src.importers.vps_m5_exporter import m5_export_enabled

    return m5_export_enabled()
