"""Resolve local M5 FT6 CSV paths for live canonical pairs."""
from __future__ import annotations

import os
from pathlib import Path

from strategies.market_utils import LIVE_CANONICAL_PAIRS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

# Prefer longer history file when both exist (10y over 5y).
_SYMBOL_CSV_CANDIDATES: dict[str, tuple[str, ...]] = {
    "GBPUSD": ("gbpusd_m5_10y.csv", "gbpusd_m5_5y.csv"),
    "EURUSD": ("eurusd_m5_10y.csv", "eurusd_m5_5y.csv"),
    "AUDUSD": ("audusd_m5_10y.csv", "audusd_m5_5y.csv"),
    "NZDUSD": ("nzdusd_m5_10y.csv", "nzdusd_m5_5y.csv"),
    "USDJPY": ("usdjpy_m5_10y.csv", "usdjpy_m5_5y.csv"),
    "AUDJPY": ("audjpy_m5_10y.csv", "audjpy_m5_5y.csv"),
    "AUDNZD": ("audnzd_m5_10y.csv", "audnzd_m5_5y.csv"),
    "EURGBP": ("eurgbp_m5_10y.csv", "eurgbp_m5_5y.csv"),
    "USDCAD": ("usdcad_m5_10y.csv", "usdcad_m5_5y.csv"),
    "XAUUSD": ("xauusd_m5_10y.csv", "xauusd_m5_5y.csv"),
}


def data_dir() -> Path:
    override = os.environ.get("M5_CSV_DATA_DIR", "").strip()
    return Path(override) if override else DATA_DIR


def export_symbols() -> list[str]:
    extra = os.environ.get("M5_EXPORT_SYMBOLS", "").strip()
    if extra:
        return [s.strip().upper() for s in extra.split(",") if s.strip()]
    return list(LIVE_CANONICAL_PAIRS)


def resolve_m5_csv_path(symbol: str, *, base: Path | None = None) -> Path:
    """Return target CSV path for a symbol (existing file or default new name)."""
    root = base or data_dir()
    sym = symbol.upper()
    candidates = _SYMBOL_CSV_CANDIDATES.get(sym, (f"{sym.lower()}_m5_5y.csv",))
    for name in candidates:
        path = root / name
        if path.is_file():
            return path
    return root / candidates[-1]


def staging_m5_dir(project_root: Path | None = None) -> Path:
    root = project_root or PROJECT_ROOT
    override = os.environ.get("M5_EXPORT_STAGING_DIR", "").strip()
    if override:
        return Path(override)
    return root / "data" / "incoming_m5_exports"
