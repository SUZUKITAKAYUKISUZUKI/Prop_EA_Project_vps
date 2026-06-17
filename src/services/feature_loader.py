"""Load Bayes / BT feature logs from SQLite with CSV fallback."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.repositories.feature_repository import FeatureRepository
from src.services.feature_write_service import normalize_logical_path


def _sqlite_read_enabled() -> bool:
    return os.environ.get("FEATURE_READ_SQLITE", "1").strip().lower() not in {"0", "false", "no"}


def load_feature_dataframe(
    path: Path | str,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    strategy: str | None = None,
    allow_csv_fallback: bool = True,
) -> pd.DataFrame:
    logical = normalize_logical_path(path)
    df = pd.DataFrame()

    if _sqlite_read_enabled():
        repo = FeatureRepository()
        try:
            df = repo.get_bayes_dataset(source_path=logical, strategy=strategy)
        finally:
            repo.close()

    if df.empty and allow_csv_fallback:
        csv_path = Path(path)
        if csv_path.is_file():
            df = pd.read_csv(csv_path)

    if df.empty:
        return df

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if start is not None:
            df = df[df["timestamp"] >= start]
        if end is not None:
            df = df[df["timestamp"] <= end]
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df
