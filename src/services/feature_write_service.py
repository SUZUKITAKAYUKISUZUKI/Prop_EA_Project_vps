"""Persist Bayes / BT feature logs to portfolio_os.db."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.database.data_source import normalize_source
from src.database.db_manager import DatabaseManager, utc_now_iso
from src.repositories.base import create_default_db_manager
from src.repositories.feature_repository import FeatureRepository


def _sqlite_write_enabled() -> bool:
    return os.environ.get("FEATURE_WRITE_SQLITE", "1").strip().lower() not in {"0", "false", "no"}


def _export_csv_enabled() -> bool:
    return os.environ.get("FEATURE_EXPORT_CSV", "0").strip().lower() in {"1", "true", "yes"}


def normalize_logical_path(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def _row_source_key(row: dict[str, Any], idx: int, *, stem: str) -> str:
    for key in ("trade_id", "timestamp", "pair", "direction"):
        value = row.get(key)
        if value not in (None, ""):
            return f"{stem}:{value}:{idx}"
    return f"{stem}:row_{idx}"


def _sanitize_row(row: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in row.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, pd.Timestamp):
            value = value.isoformat(sep=" ")
        if pd.isna(value):
            payload[key] = None
        else:
            payload[key] = value
    return payload


class FeatureWriteService:
    def __init__(self, db: DatabaseManager | None = None, *, owns_connection: bool = False) -> None:
        self._db = db or create_default_db_manager()
        self._owns = owns_connection or db is None
        self._features = FeatureRepository(self._db, owns_connection=False)

    def close(self) -> None:
        if self._owns:
            self._db.close()

    def _ensure_feature_run(
        self,
        logical_path: str,
        *,
        strategy: str | None,
        csv_kind: str,
        source: str,
    ) -> int:
        rel = normalize_logical_path(logical_path)
        existing = self._features.resolve_run_id(source_path=rel)
        if existing is not None:
            return existing

        run_id = self._db.insert_run(
            "feature",
            strategy=strategy,
            description=rel,
            parameters={"logical_path": rel, "native": True},
            source=normalize_source(source),
        )
        self._db.register_import(rel, run_id, csv_kind, 0, None)
        return run_id

    def save_feature_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        logical_path: str | Path,
        strategy: str | None = None,
        csv_kind: str = "feature",
        source: str = "BACKTEST",
        upsert: bool = True,
    ) -> int:
        if not rows or not _sqlite_write_enabled():
            return 0

        rel = normalize_logical_path(logical_path)
        run_id = self._ensure_feature_run(rel, strategy=strategy, csv_kind=csv_kind, source=source)
        stem = Path(rel).stem
        written = 0
        start_idx = self._features.count_features(run_id) if upsert else 0
        for offset, row in enumerate(rows):
            idx = start_idx + offset
            payload = _sanitize_row(row)
            source_key = _row_source_key(payload, idx, stem=stem)
            self._db.insert_feature(
                run_id,
                payload,
                strategy=strategy or payload.get("setup_type") or payload.get("strategy"),
                source_key=source_key,
                source=source,
                upsert=upsert,
            )
            written += 1

        total = self._features.count_features(run_id)
        self._db.register_import(rel, run_id, csv_kind, total, None)
        return written

    def sync_feature_rows(
        self,
        records: list[dict[str, Any]],
        *,
        logical_path: str | Path,
        strategy: str | None = None,
        csv_kind: str = "feature",
        source: str = "BACKTEST",
    ) -> int:
        if not records or not _sqlite_write_enabled():
            return 0

        rel = normalize_logical_path(logical_path)
        run_id = self._features.resolve_run_id(source_path=rel)
        if run_id is None:
            return self.save_feature_rows(
                records,
                logical_path=rel,
                strategy=strategy,
                csv_kind=csv_kind,
                source=source,
            )

        existing = self._features.count_features(run_id)
        if len(records) <= existing:
            return 0
        return self.save_feature_rows(
            records[existing:],
            logical_path=rel,
            strategy=strategy,
            csv_kind=csv_kind,
            source=source,
        )

    def save_feature_dataframe(
        self,
        df: pd.DataFrame,
        *,
        logical_path: str | Path,
        strategy: str | None = None,
        csv_kind: str = "feature",
        source: str = "BACKTEST",
        export_csv_path: Path | None = None,
    ) -> int:
        rows = df.to_dict(orient="records")
        count = self.save_feature_rows(
            rows,
            logical_path=logical_path,
            strategy=strategy,
            csv_kind=csv_kind,
            source=source,
        )
        if export_csv_path is not None and _export_csv_enabled():
            export_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(export_csv_path, index=False, encoding="utf-8-sig")
        return count


def persist_dataframe_features(
    df: pd.DataFrame,
    *,
    strategy: str,
    logical_path: str | Path,
    csv_kind: str = "feature",
    source: str = "BACKTEST",
    export_csv_path: Path | None = None,
) -> int:
    service = FeatureWriteService()
    try:
        return service.save_feature_dataframe(
            df,
            logical_path=logical_path,
            strategy=strategy,
            csv_kind=csv_kind,
            source=source,
            export_csv_path=export_csv_path,
        )
    finally:
        service.close()
