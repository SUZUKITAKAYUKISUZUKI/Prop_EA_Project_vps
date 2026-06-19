"""Pull VPS M5 bar exports over SSH and merge into local FT6 CSV files."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from src.importers.m5_csv_merge import merge_ft6_csv
from src.importers.m5_csv_paths import data_dir, export_symbols, resolve_m5_csv_path, staging_m5_dir
from src.importers.vps_ssh import pull_remote_file, run_plink, ssh_cfg

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_SEC = 3600
DEFAULT_REMOTE_PROJECT = r"C:\Prop_EA_Project_vps"
DEFAULT_LOOKBACK_DAYS = 21


def m5_export_enabled() -> bool:
    flag = os.environ.get("VPS_M5_EXPORT", "0").strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        return False
    return ssh_cfg() is not None


def remote_project_dir() -> str:
    return os.environ.get("VPS_REMOTE_PROJECT_DIR", DEFAULT_REMOTE_PROJECT).strip() or DEFAULT_REMOTE_PROJECT


def remote_export_dir() -> str:
    override = os.environ.get("VPS_REMOTE_M5_EXPORT_DIR", "").strip()
    if override:
        return override
    project = remote_project_dir().rstrip("\\/")
    return f"{project}\\data\\m5_exports"


def lookback_days() -> int:
    try:
        return max(1, int(os.environ.get("VPS_M5_LOOKBACK_DAYS", str(DEFAULT_LOOKBACK_DAYS))))
    except ValueError:
        return DEFAULT_LOOKBACK_DAYS


def python_cmd() -> str:
    return os.environ.get("VPS_PYTHON_CMD", "py -3.10").strip() or "py -3.10"


class VpsM5Exporter:
    def __init__(self, *, staging: Path | None = None) -> None:
        self.staging = staging or staging_m5_dir()
        self._last_export_at = 0.0
        self._last_error = ""

    @property
    def export_interval_sec(self) -> int:
        try:
            return max(60, int(os.environ.get("VPS_M5_EXPORT_INTERVAL_SEC", str(DEFAULT_INTERVAL_SEC))))
        except ValueError:
            return DEFAULT_INTERVAL_SEC

    def _run_remote_export(self, cfg: dict[str, str]) -> None:
        project = remote_project_dir()
        export_dir = remote_export_dir()
        symbols = ",".join(export_symbols())
        days = lookback_days()
        script = f"{project}\\scripts\\vps_export_m5_bars.py"
        cmd = (
            f'cd /d "{project}" && {python_cmd()} "{script}" '
            f'--output-dir "{export_dir}" --symbols "{symbols}" --lookback-days {days}'
        )
        timeout = int(os.environ.get("VPS_M5_EXPORT_TIMEOUT_SEC", "300"))
        result = run_plink(cfg, cmd, timeout=timeout)
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or b"").decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"remote M5 export failed: {stderr or result.returncode}")

    def _pull_symbol_file(self, cfg: dict[str, str], symbol: str) -> Path | None:
        remote_path = f"{remote_export_dir()}\\{symbol}_m5.csv"
        dest = self.staging / f"{symbol}_m5.csv"
        try:
            return pull_remote_file(cfg, remote_path, dest)
        except RuntimeError as exc:
            logger.warning("M5 pull failed for %s: %s", symbol, exc)
            return None

    def maybe_export(self, *, force: bool = False) -> dict[str, Any]:
        if not m5_export_enabled():
            return {"enabled": False, "merged": {}, "skipped": True}

        cfg = ssh_cfg()
        if cfg is None:
            return {"enabled": False, "merged": {}, "skipped": True, "reason": "missing_vps_credentials"}

        now = time.time()
        if not force and now - self._last_export_at < self.export_interval_sec:
            return {"enabled": True, "merged": {}, "skipped": True, "reason": "interval"}

        summary: dict[str, Any] = {
            "enabled": True,
            "skipped": False,
            "staging_dir": str(self.staging),
            "remote_export_dir": remote_export_dir(),
            "merged": {},
        }
        try:
            self.staging.mkdir(parents=True, exist_ok=True)
            self._run_remote_export(cfg)
            merged: dict[str, dict[str, object]] = {}
            target_dir = data_dir()
            for symbol in export_symbols():
                incoming = self._pull_symbol_file(cfg, symbol)
                if incoming is None or not incoming.is_file():
                    merged[symbol] = {"merged": 0, "error": "pull_failed"}
                    continue
                target = resolve_m5_csv_path(symbol, base=target_dir)
                merged[symbol] = merge_ft6_csv(target, incoming, symbol=symbol)
            summary["merged"] = merged
            summary["symbols_ok"] = sum(1 for row in merged.values() if not row.get("error"))
            self._last_export_at = now
            self._last_error = ""
            logger.info(
                "VPS M5 export complete: %d/%d symbols merged",
                summary["symbols_ok"],
                len(export_symbols()),
            )
        except Exception as exc:
            self._last_error = str(exc)
            summary["error"] = self._last_error
            logger.warning("VPS M5 export failed: %s", exc)
        return summary
