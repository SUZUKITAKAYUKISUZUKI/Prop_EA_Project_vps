"""File logging for the PortfolioOS import daemon."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "daemon"

_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_daemon_logging(log_dir: Path | None = None) -> tuple[logging.Logger, Path, Path]:
    directory = log_dir or DEFAULT_LOG_DIR
    directory.mkdir(parents=True, exist_ok=True)
    info_path = directory / "daemon.log"
    error_path = directory / "daemon_error.log"

    logger = logging.getLogger("portfolioos.import_daemon")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    info_handler = RotatingFileHandler(info_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    error_handler = RotatingFileHandler(error_path, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    logger.addHandler(info_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console)
    return logger, info_path, error_path
