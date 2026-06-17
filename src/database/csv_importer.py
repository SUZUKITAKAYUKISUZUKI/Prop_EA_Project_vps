"""CSV auto-detection and import into portfolio_os.db."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.database.db_manager import DatabaseManager, utc_now_iso
from src.database.data_source import infer_source_from_path

logger = logging.getLogger(__name__)

CsvKind = str  # trade | feature | wft | mc | summary | portfolio | risk | skip

TRADE_REQUIRED = {"trade_id", "trade_result"}
TRADE_R_COLS = {"profit_r", "sized_result_r"}


def _norm_cols(cols: list[str]) -> set[str]:
    return {str(c).strip().lower() for c in cols}


def _file_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_csv(path: Path) -> CsvKind:
    """Infer CSV category from filename and header row."""
    name = path.name.lower()
    stem = path.stem.lower()

    try:
        cols = _norm_cols(list(pd.read_csv(path, nrows=0).columns))
    except Exception:
        return "skip"

    if "correlation_matrix" in name or stem == "feature_decay":
        return "skip"
    if "calibration_table" in name or "probability_bucket" in name:
        return "feature" if cols & {"bayes_probability", "trade_r"} else "skip"

    if TRADE_REQUIRED <= cols and cols & TRADE_R_COLS:
        return "trade"

    if "window_id" in cols and (
        "oos_pf" in cols or "oos_profit_r" in cols or "oos_total_r" in cols or "oos_max_dd" in cols
    ):
        return "wft"

    if {"pass_rate", "failure_rate"} <= cols or {"pass_rate", "fail_rate"} <= cols:
        if "window_id" not in cols:
            return "mc"

    if "risk contribution" in name or stem in {"risk_contribution", "marginal_contribution"}:
        return "risk"

    if "allocation" in name or stem.startswith("allocation"):
        return "portfolio"

    if stem in {"strategy_summary", "phase4_scorecard"} or name.endswith("scorecard.csv"):
        return "summary"

    if cols & {"bayes_probability"} and cols & {"trade_r", "profit_r"}:
        return "feature"

    if "feature" in name or "bayes" in name:
        if len(cols) >= 4:
            return "feature"

    if {"pf", "total_r", "totalr"} & {c.replace(" ", "_") for c in cols}:
        if "strategy" in cols or "model_id" in cols:
            return "summary"

    return "skip"


def _first_col(row: pd.Series, *names: str) -> Any:
    for name in names:
        if name in row.index:
            val = row[name]
            if pd.notna(val):
                return val
        lower_map = {str(k).lower(): k for k in row.index}
        for name in names:
            key = lower_map.get(name.lower())
            if key is not None and pd.notna(row[key]):
                return row[key]
    return None


def _as_float(val: Any) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _as_str(val: Any) -> str | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return str(val)


def _infer_strategy(row: pd.Series, default: str | None = None) -> str | None:
    for key in ("setup_type", "strategy", "portfolio_source", "model_id", "name"):
        val = _first_col(row, key)
        if val is not None:
            text = str(val).strip()
            if text and text.upper() not in {"NONE", "NAN"}:
                return text
    trade_id = _as_str(_first_col(row, "trade_id"))
    if trade_id:
        if trade_id.startswith("SMRS"):
            return "SMRS"
        if trade_id.startswith("TX_"):
            return "PORTFOLIO"
    return default


def _parse_allocation(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in str(text).split(","):
        part = part.strip()
        if ":" not in part:
            continue
        name, weight = part.split(":", 1)
        try:
            out[name.strip()] = float(weight.strip())
        except ValueError:
            continue
    return out


def _ensure_run(
    db: DatabaseManager,
    path: Path,
    csv_kind: CsvKind,
    *,
    strategy: str | None = None,
    upsert: bool = True,
) -> int:
    rel = path.as_posix()
    existing = db.get_import_run_id(rel) if upsert else None
    if existing is not None:
        return existing
    return db.insert_run(
        csv_kind,
        strategy=strategy,
        description=rel,
        parameters={"source_file": rel, "checksum": _file_checksum(path)},
        created_at=utc_now_iso(),
        source=infer_source_from_path(path, csv_kind=csv_kind),
    )


def import_trade_csv(db: DatabaseManager, path: Path, *, upsert: bool = True) -> int:
    run_id = _ensure_run(db, path, "trade", strategy="portfolio", upsert=upsert)
    count = 0
    for chunk in pd.read_csv(path, chunksize=50_000, low_memory=False):
        for _, row in chunk.iterrows():
            entry_time = _as_str(_first_col(row, "timestamp", "entry_time", "datetime"))
            holding = _as_float(_first_col(row, "holding_time"))
            exit_time = None
            if entry_time and holding is not None:
                try:
                    exit_time = (
                        pd.to_datetime(entry_time) + pd.to_timedelta(holding, unit="m")
                    ).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    exit_time = None

            strategy = _infer_strategy(row, "UNKNOWN")
            r_val = _first_col(row, "profit_r", "sized_result_r")
            db.insert_trade(
                run_id,
                strategy=strategy,
                symbol=_as_str(_first_col(row, "pair", "symbol")),
                direction=_as_str(_first_col(row, "direction", "side")),
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=_as_float(_first_col(row, "entry_price")),
                exit_price=_as_float(_first_col(row, "exit_price")),
                r_multiple=_as_float(r_val),
                profit=_as_float(_first_col(row, "profit_loss", "profit")),
                result=_as_str(_first_col(row, "trade_result", "result")),
                source_trade_id=_as_str(_first_col(row, "trade_id")),
                upsert=upsert,
            )
            count += 1
    db.register_import(path.as_posix(), run_id, "trade", count, _file_checksum(path))
    return count


def import_feature_csv(db: DatabaseManager, path: Path, *, upsert: bool = True) -> int:
    run_id = _ensure_run(db, path, "feature", upsert=upsert)
    count = 0
    df = pd.read_csv(path, low_memory=False)
    for idx, row in df.iterrows():
        payload = {str(k): (None if pd.isna(v) else v) for k, v in row.items()}
        for k, v in list(payload.items()):
            if hasattr(v, "item"):
                payload[k] = v.item()
        source_key = _as_str(_first_col(row, "trade_id", "timestamp", "pair")) or f"row_{idx}"
        db.insert_feature(
            run_id,
            payload,
            strategy=_infer_strategy(row),
            source_key=f"{path.stem}:{source_key}:{idx}",
            upsert=upsert,
        )
        count += 1
    db.register_import(path.as_posix(), run_id, "feature", count, _file_checksum(path))
    return count


def import_wft_csv(db: DatabaseManager, path: Path, *, upsert: bool = True) -> int:
    run_id = _ensure_run(db, path, "wft", upsert=upsert)
    count = 0
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        window_id = int(_first_col(row, "window_id"))
        oos_r = _as_float(_first_col(row, "oos_profit_r", "oos_total_r", "oos_r"))
        oos_pf = _as_float(_first_col(row, "oos_pf"))
        oos_dd = _as_float(_first_col(row, "oos_max_dd", "oos_max_dd_r"))
        pass_rate = _as_float(_first_col(row, "oos_pass_rate", "pass_rate"))
        pass_flag = None
        if pass_rate is not None:
            pass_flag = 1 if pass_rate >= 100.0 else 0
        db.insert_wft_result(
            run_id,
            window_id,
            oos_pf=oos_pf,
            oos_r=oos_r,
            oos_dd=oos_dd,
            pass_flag=pass_flag,
            upsert=upsert,
        )
        count += 1
    db.register_import(path.as_posix(), run_id, "wft", count, _file_checksum(path))
    return count


def import_mc_csv(db: DatabaseManager, path: Path, *, upsert: bool = True) -> int:
    run_id = _ensure_run(db, path, "mc", upsert=upsert)
    count = 0
    df = pd.read_csv(path)
    for idx, row in df.iterrows():
        label = _as_str(_first_col(row, "model_id", "trials", "name")) or f"row_{idx}"
        pass_rate = _as_float(_first_col(row, "pass_rate"))
        ror = _as_float(_first_col(row, "risk_of_ruin", "failure_rate", "fail_rate"))
        avg_pass_days = _as_float(_first_col(row, "avg_pass_days"))
        max_dd = _as_float(_first_col(row, "worst_dd", "max_dd", "max_dd_r", "dd_p95"))
        db.insert_mc_result(
            run_id,
            pass_rate=pass_rate,
            ror=ror,
            avg_pass_days=avg_pass_days,
            max_dd=max_dd,
            label=str(label),
            upsert=upsert,
        )
        count += 1
    db.register_import(path.as_posix(), run_id, "mc", count, _file_checksum(path))
    return count


def import_summary_csv(db: DatabaseManager, path: Path, *, upsert: bool = True) -> int:
    run_id = _ensure_run(db, path, "summary", upsert=upsert)
    count = 0
    df = pd.read_csv(path)
    for idx, row in df.iterrows():
        label = _as_str(_first_col(row, "model_id", "name", "strategy")) or f"row_{idx}"
        pf = _as_float(_first_col(row, "pf", "mean_pf_oos"))
        wr = _as_float(_first_col(row, "wr"))
        total_r = _as_float(_first_col(row, "total_r", "totalr"))
        max_dd = _as_float(_first_col(row, "max_dd", "max_dd_r", "worst_dd_r_oos"))
        sharpe = _as_float(_first_col(row, "sharpe"))
        recovery = _as_float(_first_col(row, "recovery_factor", "recovery"))
        db.insert_bt_summary(
            run_id,
            pf=pf,
            wr=wr,
            total_r=total_r,
            max_dd=max_dd,
            sharpe=sharpe,
            recovery=recovery,
            label=label,
            upsert=upsert,
        )
        count += 1
    db.register_import(path.as_posix(), run_id, "summary", count, _file_checksum(path))
    return count


def import_portfolio_csv(db: DatabaseManager, path: Path, *, upsert: bool = True) -> int:
    run_id = _ensure_run(db, path, "portfolio", upsert=upsert)
    count = 0
    df = pd.read_csv(path)
    for idx, row in df.iterrows():
        rank_val = _first_col(row, "rank")
        rank = int(rank_val) if rank_val is not None and pd.notna(rank_val) else idx + 1
        alloc_raw = _first_col(row, "allocation")
        allocation = _parse_allocation(alloc_raw) if alloc_raw is not None else {}
        db.insert_portfolio_result(
            run_id,
            allocation_json=allocation or {"raw": _as_str(alloc_raw)},
            pf=_as_float(_first_col(row, "pf")),
            total_r=_as_float(_first_col(row, "total_r", "score")),
            max_dd=_as_float(_first_col(row, "dd", "max_dd")),
            pass_rate=_as_float(_first_col(row, "passrate", "pass_rate")),
            rank=rank,
            upsert=upsert,
        )
        count += 1
    db.register_import(path.as_posix(), run_id, "portfolio", count, _file_checksum(path))
    return count


def import_risk_csv(db: DatabaseManager, path: Path, *, upsert: bool = True) -> int:
    run_id = _ensure_run(db, path, "risk", upsert=upsert)
    count = 0
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        strategy = _as_str(_first_col(row, "strategy")) or "UNKNOWN"
        dd = _as_float(_first_col(row, "dd contribution %", "dd_contribution_pct", "contribution_dd"))
        if dd is None:
            for col in row.index:
                if "dd" in str(col).lower() and "contribution" in str(col).lower():
                    dd = _as_float(row[col])
                    break
        db.insert_risk_attribution(
            run_id,
            strategy=strategy,
            contribution_r=_as_float(_first_col(row, "contribution_r", "totalr", "total_r")),
            contribution_dd=dd,
            contribution_pf=_as_float(_first_col(row, "pf", "contribution_pf")),
            upsert=upsert,
        )
        count += 1
    db.register_import(path.as_posix(), run_id, "risk", count, _file_checksum(path))
    return count


IMPORTERS = {
    "trade": import_trade_csv,
    "feature": import_feature_csv,
    "wft": import_wft_csv,
    "mc": import_mc_csv,
    "summary": import_summary_csv,
    "portfolio": import_portfolio_csv,
    "risk": import_risk_csv,
}


def discover_csv_files(roots: list[Path], skip_globs: list[str] | None = None) -> list[Path]:
    files: list[Path] = []
    skip_globs = skip_globs or []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.csv")):
            rel = path.as_posix()
            if any(path.match(glob) for glob in skip_globs):
                continue
            files.append(path)
    return files


def import_csv_file(db: DatabaseManager, path: Path, *, upsert: bool = True) -> tuple[CsvKind, int]:
    kind = classify_csv(path)
    if kind == "skip":
        return kind, 0
    importer = IMPORTERS[kind]
    try:
        rows = importer(db, path, upsert=upsert)
        logger.info("Imported %s (%s): %d rows", path, kind, rows)
        return kind, rows
    except Exception as exc:
        logger.exception("Failed to import %s (%s): %s", path, kind, exc)
        return kind, -1


def import_all_csvs(
    db: DatabaseManager,
    roots: list[Path],
    *,
    skip_globs: list[str] | None = None,
    upsert: bool = True,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "files_scanned": 0,
        "imported": 0,
        "skipped": 0,
        "failed": 0,
        "rows_by_kind": {},
        "failures": [],
    }
    for path in discover_csv_files(roots, skip_globs):
        summary["files_scanned"] += 1
        kind, rows = import_csv_file(db, path, upsert=upsert)
        if kind == "skip":
            summary["skipped"] += 1
            continue
        if rows < 0:
            summary["failed"] += 1
            summary["failures"].append(path.as_posix())
            continue
        summary["imported"] += 1
        summary["rows_by_kind"][kind] = summary["rows_by_kind"].get(kind, 0) + rows
    return summary
