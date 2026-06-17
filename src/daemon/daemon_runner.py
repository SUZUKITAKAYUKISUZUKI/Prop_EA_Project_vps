"""Never-terminating import daemon runner with watchdog and heartbeat."""
from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.daemon.daemon_logging import setup_daemon_logging
from src.daemon.import_daemon import ImportDaemonService, HEARTBEAT_INTERVAL_SEC
from src.daemon.watchdog_service import WatchdogService
from src.runtime.logging_config import load_dropbox_logging_config

_running = True


def _handle_stop(*_args) -> None:
    global _running
    _running = False


def run_forever(*, use_watchdog: bool | None = None) -> int:
    global _running
    logger, _, _ = setup_daemon_logging()
    config = load_dropbox_logging_config()
    poll_sec = max(1, int(config.poll_interval_seconds))
    enable_watchdog = use_watchdog
    if enable_watchdog is None:
        enable_watchdog = os.environ.get("DAEMON_WATCHDOG", "1").strip().lower() not in {"0", "false", "no"}

    service = ImportDaemonService(config=config)
    watchdog: WatchdogService | None = None

    def trigger_import() -> None:
        try:
            service.run_import_cycle()
        except Exception:
            logger.exception("Watchdog import cycle failed")

    try:
        logger.info("Import daemon starting — watch=%s poll=%ss", service.watch_dir, poll_sec)
        service.daemon_repo.update_heartbeat(last_error="")
        service.maybe_update_heartbeat(force=True)

        if enable_watchdog:
            watchdog = WatchdogService(service.watch_dir, trigger_import)
            watchdog.start()

        signal.signal(signal.SIGINT, _handle_stop)
        signal.signal(signal.SIGTERM, _handle_stop)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, _handle_stop)

        last_heartbeat = 0.0
        while _running:
            try:
                service.run_import_cycle()
            except Exception as exc:
                logger.exception("Import cycle error: %s", exc)
                try:
                    service.daemon_repo.update_heartbeat(last_error=str(exc)[:500])
                except Exception:
                    logger.exception("Failed to record heartbeat error")

            now = time.time()
            if now - last_heartbeat >= HEARTBEAT_INTERVAL_SEC:
                service.maybe_update_heartbeat(force=True)
                last_heartbeat = now

            time.sleep(poll_sec)

        logger.info("Import daemon stopping")
        return 0
    finally:
        if watchdog is not None:
            watchdog.stop()
        service.close()


def main() -> int:
    return run_forever()


if __name__ == "__main__":
    raise SystemExit(main())
