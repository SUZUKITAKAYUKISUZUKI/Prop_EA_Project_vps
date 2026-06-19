"""Pull VPS Dropbox JSONL event files over SSH when local Dropbox sync is empty."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_REMOTE_EVENTS_DIR = r"C:\Dropbox\PortfolioOS\events"
DEFAULT_STAGING_DIRNAME = "incoming_vps_events"
DEFAULT_PULL_INTERVAL_SEC = 60


def _find_binary(name: str, fallbacks: tuple[str, ...]) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for candidate in fallbacks:
        if Path(candidate).is_file():
            return candidate
    return None


def _plink_path() -> str | None:
    return _find_binary(
        "plink",
        (
            r"C:\Program Files\PuTTY\plink.exe",
            r"C:\Program Files (x86)\PuTTY\plink.exe",
        ),
    )


def _pscp_path() -> str | None:
    return _find_binary(
        "pscp",
        (
            r"C:\Program Files\PuTTY\pscp.exe",
            r"C:\Program Files (x86)\PuTTY\pscp.exe",
        ),
    )


def _ssh_cfg() -> dict[str, str] | None:
    host = os.environ.get("VPS_HOST", "").strip()
    user = os.environ.get("VPS_SSH_USER", "").strip()
    password = os.environ.get("VPS_SSH_PASSWORD", "").strip()
    port = os.environ.get("VPS_SSH_PORT", "22").strip() or "22"
    if not host or not user or not password:
        return None
    return {"host": host, "user": user, "password": password, "port": port}


def staging_dir(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parents[2]
    override = os.environ.get("VPS_EVENTS_STAGING_DIR", "").strip()
    if override:
        return Path(override)
    return root / "data" / DEFAULT_STAGING_DIRNAME


def remote_events_dir() -> str:
    return os.environ.get("VPS_REMOTE_EVENTS_DIR", DEFAULT_REMOTE_EVENTS_DIR).strip() or DEFAULT_REMOTE_EVENTS_DIR


def _run_plink(cfg: dict[str, str], remote_cmd: str, *, timeout: int = 90) -> subprocess.CompletedProcess[bytes]:
    plink = _plink_path()
    if not plink:
        raise FileNotFoundError("plink not found")
    return subprocess.run(
        [
            plink,
            "-ssh",
            "-batch",
            "-P",
            cfg["port"],
            "-l",
            cfg["user"],
            "-pw",
            cfg["password"],
            cfg["host"],
            remote_cmd,
        ],
        capture_output=True,
        timeout=timeout,
    )


def list_remote_event_files(cfg: dict[str, str]) -> list[str]:
    remote_dir = remote_events_dir()
    cmd = f'dir /b "{remote_dir}\\events_*.jsonl*" 2>nul'
    result = _run_plink(cfg, cmd)
    if result.returncode != 0 and not result.stdout:
        stderr = (result.stderr or b"").decode("utf-8", errors="replace")[:200]
        raise RuntimeError(f"remote dir listing failed: {stderr or result.returncode}")
    names = [line.strip() for line in result.stdout.decode("utf-8", errors="replace").splitlines() if line.strip()]
    return sorted(names)


def _remote_file_size(cfg: dict[str, str], filename: str) -> int | None:
    remote_dir = remote_events_dir()
    remote_path = f"{remote_dir}\\{filename}"
    ps = (
        "powershell -NoProfile -Command "
        f"\"if (Test-Path '{remote_path}') {{ (Get-Item '{remote_path}').Length }}\""
    )
    result = _run_plink(cfg, ps)
    text = result.stdout.decode("utf-8", errors="replace").strip()
    try:
        return int(text)
    except ValueError:
        return None


def pull_remote_file(cfg: dict[str, str], filename: str, dest_dir: Path) -> Path:
    pscp = _pscp_path()
    if not pscp:
        raise FileNotFoundError("pscp not found")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    remote_dir = remote_events_dir().replace("\\", "/")
    remote_spec = f"{cfg['host']}:{remote_dir}/{filename}"
    result = subprocess.run(
        [
            pscp,
            "-batch",
            "-P",
            cfg["port"],
            "-l",
            cfg["user"],
            "-pw",
            cfg["password"],
            remote_spec,
            str(dest),
        ],
        capture_output=True,
        timeout=180,
    )
    if result.returncode != 0:
        stderr = (result.stderr or b"").decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"pscp failed for {filename}: {stderr or result.returncode}")
    return dest


class VpsEventPuller:
    def __init__(self, *, staging: Path | None = None) -> None:
        self.staging = staging or staging_dir()
        self._last_pull_at = 0.0
        self._last_error = ""

    @property
    def pull_interval_sec(self) -> int:
        return int(os.environ.get("VPS_EVENTS_PULL_INTERVAL_SEC", str(DEFAULT_PULL_INTERVAL_SEC)))

    def maybe_pull(self, *, force: bool = False) -> dict[str, Any]:
        if not os.environ.get("VPS_EVENTS_PULL", "1").strip().lower() in {"1", "true", "yes", "on"}:
            return {"enabled": False, "pulled": [], "skipped": True}
        cfg = _ssh_cfg()
        if cfg is None:
            return {"enabled": False, "pulled": [], "skipped": True, "reason": "missing_vps_credentials"}

        now = time.time()
        if not force and now - self._last_pull_at < self.pull_interval_sec:
            return {"enabled": True, "pulled": [], "skipped": True, "reason": "interval"}

        summary: dict[str, Any] = {
            "enabled": True,
            "pulled": [],
            "skipped": False,
            "staging_dir": str(self.staging),
            "remote_dir": remote_events_dir(),
        }
        try:
            names = list_remote_event_files(cfg)
            summary["remote_files"] = len(names)
            for name in names:
                dest = self.staging / name
                remote_size = _remote_file_size(cfg, name)
                if dest.is_file() and remote_size is not None and dest.stat().st_size >= remote_size:
                    continue
                pull_remote_file(cfg, name, self.staging)
                summary["pulled"].append(name)
                logger.info("Pulled VPS event file %s -> %s", name, dest)
            self._last_pull_at = now
            self._last_error = ""
        except Exception as exc:
            self._last_error = str(exc)
            summary["error"] = self._last_error
            logger.warning("VPS event pull failed: %s", exc)
        return summary
