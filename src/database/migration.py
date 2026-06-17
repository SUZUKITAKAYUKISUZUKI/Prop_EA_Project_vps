"""End-to-end SQLite migration orchestration."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.database.config_loader import load_database_config, resolve_project_paths
from src.database.csv_importer import import_all_csvs
from src.database.db_manager import DatabaseManager
from src.database.market_importer import import_all_market_csvs
from src.database.schema import MARKET_TABLES, PORTFOLIO_TABLES

logger = logging.getLogger(__name__)


def _table_counts(db: DatabaseManager) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in PORTFOLIO_TABLES:
        row = db.query(f"SELECT COUNT(*) AS c FROM {table}", one=True)
        counts[table] = int(row["c"]) if row else 0
    counts["candles"] = 0
    if db.market_path and db.market_path.exists():
        row = db.query("SELECT COUNT(*) AS c FROM candles", market=True, one=True)
        counts["candles"] = int(row["c"]) if row else 0
    return counts


def run_migration(project_root: Path, config_path: Path | None = None) -> dict[str, Any]:
    config_path = config_path or project_root / "config" / "database.yaml"
    raw = load_database_config(config_path)
    cfg = resolve_project_paths(project_root, raw)

    result: dict[str, Any] = {
        "config_path": str(config_path),
        "portfolio_db": str(cfg["portfolio_path"]),
        "market_db": str(cfg["market_path"]),
        "csv_import": {},
        "market_import": {},
        "table_counts": {},
    }

    with DatabaseManager(
        cfg["portfolio_path"],
        cfg["market_path"],
        journal_mode=cfg["journal_mode"],
        synchronous=cfg["synchronous"],
    ) as db:
        logger.info("Schema ready: portfolio=%s market=%s", cfg["portfolio_path"], cfg["market_path"])

        result["csv_import"] = import_all_csvs(
            db,
            cfg["scan_roots"],
            skip_globs=cfg["skip_globs"],
            upsert=cfg["dedupe"],
        )
        result["market_import"] = import_all_market_csvs(
            db,
            cfg["market_roots"],
            chunk_size=cfg["chunk_size"],
        )
        result["table_counts"] = _table_counts(db)

    return result


def format_migration_summary(result: dict[str, Any]) -> str:
    lines = [
        "=== SQLite Migration Summary ===",
        f"Portfolio DB: {result['portfolio_db']}",
        f"Market DB:    {result['market_db']}",
        "",
        "CSV import:",
        f"  scanned : {result['csv_import'].get('files_scanned', 0)}",
        f"  imported: {result['csv_import'].get('imported', 0)}",
        f"  skipped : {result['csv_import'].get('skipped', 0)}",
        f"  failed  : {result['csv_import'].get('failed', 0)}",
    ]
    rows_by_kind = result["csv_import"].get("rows_by_kind", {})
    if rows_by_kind:
        lines.append("  rows by kind:")
        for kind, count in sorted(rows_by_kind.items()):
            lines.append(f"    - {kind}: {count}")

    market = result.get("market_import", {})
    lines.extend(
        [
            "",
            "Market import:",
            f"  files   : {market.get('files', 0)}",
            f"  rows    : {market.get('rows_inserted', 0)}",
            f"  failures: {len(market.get('failures', []))}",
            "",
            "Table counts:",
        ]
    )
    for table, count in sorted(result.get("table_counts", {}).items()):
        lines.append(f"  {table:20s} {count}")
    return "\n".join(lines)
