"""Local Dropbox folder watcher — polls and imports new JSONL events into SQLite."""
from __future__ import annotations

import logging
import signal
import time
from typing import Callable

from src.importers.dropbox_event_importer import DropboxEventImporter
from src.runtime.logging_config import DropboxLoggingConfig, load_dropbox_logging_config, require_consumer

logger = logging.getLogger(__name__)


class DropboxImportDaemon:
    def __init__(
        self,
        config: DropboxLoggingConfig | None = None,
        importer: DropboxEventImporter | None = None,
        *,
        on_import: Callable[[dict], None] | None = None,
    ) -> None:
        self.config = config or load_dropbox_logging_config()
        require_consumer(self.config, component="DropboxImportDaemon")
        self.importer = importer or DropboxEventImporter(self.config)
        self.on_import = on_import
        self._running = False

    def run_once(self) -> dict:
        summary = self.importer.import_all()
        if summary["imported"]:
            logger.info(
                "Imported %d events (%d duplicates skipped) from %d files",
                summary["imported"],
                summary["duplicates"],
                summary["files"],
            )
        if self.on_import:
            self.on_import(summary)
        return summary

    def run_forever(self) -> None:
        self._running = True
        logger.info(
            "Dropbox import daemon started — role=consumer watch=%s poll=%ss",
            self.config.watch_dir,
            self.config.poll_interval_seconds,
        )
        while self._running:
            try:
                self.run_once()
            except Exception:
                logger.exception("Import cycle failed — will retry")
            time.sleep(self.config.poll_interval_seconds)

    def stop(self) -> None:
        self._running = False


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    daemon = DropboxImportDaemon()
    signal.signal(signal.SIGINT, lambda *_: daemon.stop())
    signal.signal(signal.SIGTERM, lambda *_: daemon.stop())
    try:
        daemon.run_forever()
    except KeyboardInterrupt:
        daemon.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
