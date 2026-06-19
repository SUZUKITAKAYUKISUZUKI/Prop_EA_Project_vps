"""Shared SSH helpers for VPS pull/export (plink + pscp)."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def find_binary(name: str, fallbacks: tuple[str, ...]) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for candidate in fallbacks:
        if Path(candidate).is_file():
            return candidate
    return None


def plink_path() -> str | None:
    return find_binary(
        "plink",
        (
            r"C:\Program Files\PuTTY\plink.exe",
            r"C:\Program Files (x86)\PuTTY\plink.exe",
        ),
    )


def pscp_path() -> str | None:
    return find_binary(
        "pscp",
        (
            r"C:\Program Files\PuTTY\pscp.exe",
            r"C:\Program Files (x86)\PuTTY\pscp.exe",
        ),
    )


def ssh_cfg() -> dict[str, str] | None:
    host = os.environ.get("VPS_HOST", "").strip()
    user = os.environ.get("VPS_SSH_USER", "").strip()
    password = os.environ.get("VPS_SSH_PASSWORD", "").strip()
    port = os.environ.get("VPS_SSH_PORT", "22").strip() or "22"
    if not host or not user or not password:
        return None
    return {"host": host, "user": user, "password": password, "port": port}


def run_plink(cfg: dict[str, str], remote_cmd: str, *, timeout: int = 120) -> subprocess.CompletedProcess[bytes]:
    plink = plink_path()
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


def pull_remote_file(cfg: dict[str, str], remote_path: str, dest: Path, *, timeout: int = 180) -> Path:
    pscp = pscp_path()
    if not pscp:
        raise FileNotFoundError("pscp not found")
    dest.parent.mkdir(parents=True, exist_ok=True)
    remote_spec = remote_path.replace("\\", "/")
    if not remote_spec.startswith("/") and ":" in remote_spec:
        remote_spec = f"/{remote_spec.replace(':', '', 1)}"
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
            f"{cfg['host']}:{remote_spec}",
            str(dest),
        ],
        capture_output=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        stderr = (result.stderr or b"").decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"pscp failed for {remote_path}: {stderr or result.returncode}")
    return dest
