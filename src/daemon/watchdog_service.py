"""Filesystem watchdog for Dropbox live log folder."""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger("portfolioos.import_daemon")

try:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    WATCHDOG_AVAILABLE = False
    FileSystemEventHandler = object  # type: ignore[misc, assignment]
    Observer = None  # type: ignore[misc, assignment]


class _DebouncedImportHandler(FileSystemEventHandler):
    def __init__(self, callback: Callable[[], None], *, debounce_sec: float = 2.0) -> None:
        super().__init__()
        self._callback = callback
        self._debounce_sec = debounce_sec
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def _schedule(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_sec, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self) -> None:
        try:
            self._callback()
        except Exception:
            logger.exception("Watchdog-triggered import failed")

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._schedule()

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._schedule()

    def on_moved(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._schedule()


class WatchdogService:
    def __init__(
        self,
        watch_dir: Path,
        import_callback: Callable[[], None],
        *,
        debounce_sec: float = 2.0,
    ) -> None:
        self.watch_dir = watch_dir
        self.import_callback = import_callback
        self.debounce_sec = debounce_sec
        self._observer: Observer | None = None

    def start(self) -> bool:
        if not WATCHDOG_AVAILABLE:
            logger.warning("watchdog package not installed — polling only")
            return False
        if not self.watch_dir.exists():
            logger.warning("Watchdog skipped — directory missing: %s", self.watch_dir)
            return False
        handler = _DebouncedImportHandler(self.import_callback, debounce_sec=self.debounce_sec)
        self._observer = Observer()
        self._observer.schedule(handler, str(self.watch_dir), recursive=False)
        self._observer.start()
        logger.info("Watchdog started on %s", self.watch_dir)
        return True

    def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
            logger.info("Watchdog stopped")
