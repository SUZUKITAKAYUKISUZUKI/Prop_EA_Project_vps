"""Bridge UI configuration — safe on VPS (no dashboard package required)."""

from __future__ import annotations

import os


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("0", "false", "off", "no", "disabled"):
        return False
    if raw in ("1", "true", "yes", "on", "enabled"):
        return True
    return default


def is_bridge_dashboard_enabled() -> bool:
    """
    Mount HTML dashboards on mt5_bridge. VPS production default: OFF.

    Env: BRIDGE_MOUNT_DASHBOARD=0|1
    """
    return _env_flag("BRIDGE_MOUNT_DASHBOARD", False)


def vps_bridge_url() -> str:
    """Remote bridge URL for local dashboard (LOCAL machine only)."""
    return os.environ.get("VPS_BRIDGE_URL", "http://127.0.0.1:8000").strip().rstrip("/")


def local_dashboard_poll_seconds() -> float:
    raw = os.environ.get("LOCAL_DASHBOARD_POLL_SEC", "3").strip()
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 3.0
