#!/usr/bin/env python3
"""Install PortfolioOS Import Daemon as a Windows scheduled task."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.runtime.dropbox_paths import ensure_watch_dir, resolve_watch_dir
from src.windows.task_scheduler import TASK_NAME, install_task, run_task_now, task_exists


def main() -> int:
    parser = argparse.ArgumentParser(description="Install PortfolioOS Import Daemon scheduled task")
    parser.add_argument("--task-name", default=TASK_NAME)
    parser.add_argument("--username", default=None, help="Windows user (default: current user)")
    parser.add_argument("--no-run", action="store_true", help="Do not start task immediately after install")
    parser.add_argument("--no-force", action="store_true", help="Fail if task already exists")
    parser.add_argument(
        "--method",
        choices=("auto", "task", "startup", "cli", "xml"),
        default="auto",
        help="auto=task then startup fallback; startup=Startup folder (no admin)",
    )
    parser.add_argument(
        "--boot",
        action="store_true",
        help="Also run at system boot (requires administrator PowerShell)",
    )
    parser.add_argument(
        "--create-watch-dir",
        action="store_true",
        help="Create the Dropbox events folder if missing",
    )
    args = parser.parse_args()

    watch = resolve_watch_dir()
    if not watch.exists():
        if args.create_watch_dir:
            ensure_watch_dir(watch)
            print(f"Created watch directory: {watch}")
        else:
            suggested = Path.home() / "Dropbox" / "PortfolioOS" / "events"
            print(f"Watch directory missing: {watch}")
            print(f"Suggested local path: {suggested}")
            print("Fix with one of:")
            print(f'  $env:DROPBOX_EVENTS_DIR = "{suggested}"')
            print("  python src\\windows\\install_task.py --create-watch-dir")
            print("  Wait for Dropbox to sync PortfolioOS from VPS")

    method = args.method
    if method == "task":
        method = "auto"

    ok, message = install_task(
        ROOT,
        task_name=args.task_name,
        username=args.username,
        force=not args.no_force,
        include_boot=args.boot,
        method=method,
    )
    print(message)
    if not ok:
        return 1

    if not args.no_run and task_exists(args.task_name):
        started, run_msg = run_task_now(args.task_name)
        print(run_msg)
        if not started:
            print("Task installed but immediate start failed - it will run at next logon.")
    elif method == "startup" or "Startup entry" in message:
        print("Startup entry installed - daemon starts at next logon.")
        print("To start now: python src\\daemon\\daemon_runner.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
