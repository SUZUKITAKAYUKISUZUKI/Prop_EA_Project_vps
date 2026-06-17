"""Load database.yaml without external YAML dependencies."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

LIST_ITEM_RE = re.compile(r"^(\s*)-\s+(.*)$")
KV_RE = re.compile(r"^(\s*)([\w_]+):\s*(.*)$")


def _parse_scalar(raw: str) -> Any:
    text = raw.strip().strip('"').strip("'")
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.isdigit():
        return int(text)
    return text


def load_database_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    root: dict[str, Any] = {}
    stack: list[tuple[dict[str, Any], int, str | None]] = [(root, -1, None)]

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue

        list_match = LIST_ITEM_RE.match(line)
        if list_match:
            indent = len(list_match.group(1))
            item = _parse_scalar(list_match.group(2))
            while len(stack) > 1 and indent <= stack[-1][1]:
                stack.pop()
            parent, _, last_key = stack[-1]
            if last_key is None:
                continue
            target = parent.get(last_key)
            if isinstance(target, dict) and not target:
                parent[last_key] = []
                target = parent[last_key]
            elif not isinstance(target, list):
                parent[last_key] = []
                target = parent[last_key]
            target.append(item)
            continue

        kv_match = KV_RE.match(line)
        if not kv_match:
            continue

        indent = len(kv_match.group(1))
        key = kv_match.group(2)
        value = kv_match.group(3).strip()

        while len(stack) > 1 and indent <= stack[-1][1]:
            stack.pop()
        current = stack[-1][0]

        if not value:
            current[key] = {}
            stack.append((current[key], indent, key))
            continue

        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                current[key] = []
            else:
                current[key] = [_parse_scalar(part) for part in inner.split(",") if part.strip()]
            stack[-1] = (current, indent, key)
            continue

        current[key] = _parse_scalar(value)
        stack[-1] = (current, indent, key)

    return root


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Generic YAML subset loader (same parser as database.yaml)."""
    return load_database_config(path)


def resolve_project_paths(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    portfolio = config.get("portfolio_os", {})
    market = config.get("market_data", {})
    migration = config.get("migration", {})

    portfolio_path = project_root / portfolio.get("path", "data/portfolio_os.db")
    market_path = project_root / market.get("path", "data/market_data.db")

    scan_roots = [project_root / p for p in migration.get("scan_roots", ["backtest_results"])]
    market_roots = [project_root / p for p in migration.get("market_data_roots", ["data/market_csv"])]

    return {
        "portfolio_path": portfolio_path,
        "market_path": market_path,
        "journal_mode": portfolio.get("journal_mode", "WAL"),
        "synchronous": portfolio.get("synchronous", "NORMAL"),
        "scan_roots": scan_roots,
        "market_roots": market_roots,
        "skip_globs": migration.get("skip_globs", []),
        "chunk_size": int(migration.get("chunk_size", 100_000)),
        "dedupe": bool(config.get("import", {}).get("dedupe", True)),
    }
