#!/usr/bin/env python3
"""Remove PortfolioOS Import Daemon Windows scheduled task."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.windows.task_scheduler import TASK_NAME, uninstall_all


def main() -> int:
    parser = argparse.ArgumentParser(description="Uninstall PortfolioOS Import Daemon scheduled task")
    parser.add_argument("--task-name", default=TASK_NAME)
    args = parser.parse_args()

    ok, message = uninstall_all(args.task_name)
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
