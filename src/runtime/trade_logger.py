"""Append-only crash-safe JSONL trade event logger for VPS Dropbox sync."""
from __future__ import annotations

import gzip
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.runtime.producer_archive_cleanup import cleanup_producer_archives
from src.runtime.logging_config import DropboxLoggingConfig, load_dropbox_logging_config, require_producer

logger = logging.getLogger(__name__)

SUPPORTED_EVENT_TYPES = frozenset(
    {
        "TRADE_OPEN",
        "TRADE_CLOSE",
        "SL_MODIFY",
        "TP_MODIFY",
        "PYRAMID_ADD",
        "PET_EXIT",
        "SENTINEL_BLOCK",
        "CHALLENGE_PASS",
        "CHALLENGE_FAIL",
        "TRADE_SIGNAL",
        "FEATURE_SNAPSHOT",
    }
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class TradeEventLogger:
    """Thread-safe append-only JSONL writer with per-line fsync."""

    def __init__(self, config: DropboxLoggingConfig | None = None) -> None:
        self.config = config or load_dropbox_logging_config()
        if self.config.data_flow.is_consumer:
            logger.info(
                "TradeEventLogger disabled on consumer role — live events are VPS → Dropbox only"
            )
        elif self.config.enabled:
            require_producer(self.config, component="TradeEventLogger")
        self._lock = threading.Lock()
        self._current_date: str | None = None
        self._current_path: Path | None = None

    @property
    def enabled(self) -> bool:
        return self.config.write_enabled

    def _event_path_for_date(self, date_str: str) -> Path:
        name = self.config.filename_pattern.format(date=date_str)
        return self.config.output_dir / name

    def _active_path(self) -> Path:
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        if self._current_date != today or self._current_path is None:
            self._current_date = today
            self._current_path = self._event_path_for_date(today)
        return self._current_path

    def _ensure_output_dir(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def _maybe_compress_old_files(self) -> None:
        if not self.config.compress_old_files:
            return
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        for path in self.config.output_dir.glob("events_*.jsonl"):
            if today in path.name:
                continue
            gz = path.with_suffix(path.suffix + ".gz")
            if gz.exists():
                continue
            try:
                with path.open("rb") as src, gzip.open(gz, "wb") as dst:
                    dst.writelines(src)
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Failed to compress %s: %s", path, exc)
        deleted = cleanup_producer_archives(
            self.config.output_dir,
            retention_days=self.config.producer.delete_gz_after_days,
        )
        if deleted:
            logger.info("Producer archive cleanup removed %d file(s)", deleted)

    def _append_line_atomic(self, path: Path, line: str) -> None:
        self._ensure_output_dir()
        data = line if line.endswith("\n") else line + "\n"
        with self._lock:
            with path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())

    def emit(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        if event_type not in SUPPORTED_EVENT_TYPES:
            logger.warning("Unsupported event_type=%s — skipped", event_type)
            return None

        event = {
            "event_id": payload.get("event_id") or str(uuid.uuid4()),
            "timestamp": payload.get("timestamp") or utc_now_iso(),
            "event_type": event_type,
            **{k: v for k, v in payload.items() if k not in {"event_id", "timestamp", "event_type"}},
        }
        path = self._active_path()
        try:
            self._append_line_atomic(path, json.dumps(event, ensure_ascii=False, separators=(",", ":")))
            self._maybe_compress_old_files()
            return event
        except OSError as exc:
            logger.exception("Failed to write trade event to %s: %s", path, exc)
            return None

    def emit_trade_open(
        self,
        *,
        trade_id: str,
        strategy: str,
        symbol: str,
        direction: str,
        entry_price: float,
        sl: float | None = None,
        tp: float | None = None,
        lot: float | None = None,
        risk_r: float | None = None,
        **extra: Any,
    ) -> dict[str, Any] | None:
        return self.emit(
            "TRADE_OPEN",
            {
                "trade_id": trade_id,
                "strategy": strategy,
                "symbol": symbol,
                "direction": direction,
                "entry_price": entry_price,
                "sl": sl,
                "tp": tp,
                "lot": lot,
                "risk_r": risk_r,
                **extra,
            },
        )

    def emit_trade_close(
        self,
        *,
        trade_id: str,
        strategy: str | None = None,
        symbol: str | None = None,
        exit_price: float | None = None,
        profit_r: float | None = None,
        result: str | None = None,
        **extra: Any,
    ) -> dict[str, Any] | None:
        return self.emit(
            "TRADE_CLOSE",
            {
                "trade_id": trade_id,
                "strategy": strategy,
                "symbol": symbol,
                "exit_price": exit_price,
                "profit_r": profit_r,
                "result": result,
                **extra,
            },
        )

    def emit_sl_modify(self, *, trade_id: str, sl: float, **extra: Any) -> dict[str, Any] | None:
        return self.emit("SL_MODIFY", {"trade_id": trade_id, "sl": sl, **extra})

    def emit_tp_modify(self, *, trade_id: str, tp: float, **extra: Any) -> dict[str, Any] | None:
        return self.emit("TP_MODIFY", {"trade_id": trade_id, "tp": tp, **extra})

    def emit_pyramid_add(self, *, trade_id: str, **extra: Any) -> dict[str, Any] | None:
        return self.emit("PYRAMID_ADD", {"trade_id": trade_id, **extra})

    def emit_pet_exit(self, *, trade_id: str | None = None, **extra: Any) -> dict[str, Any] | None:
        return self.emit("PET_EXIT", {"trade_id": trade_id, **extra})

    def emit_sentinel_block(self, *, reason: str, **extra: Any) -> dict[str, Any] | None:
        return self.emit("SENTINEL_BLOCK", {"reason": reason, **extra})

    def emit_challenge_pass(self, **extra: Any) -> dict[str, Any] | None:
        return self.emit("CHALLENGE_PASS", extra)

    def emit_challenge_fail(self, *, reason: str, **extra: Any) -> dict[str, Any] | None:
        return self.emit("CHALLENGE_FAIL", {"reason": reason, **extra})

    def emit_trade_signal(self, *, action: str, **extra: Any) -> dict[str, Any] | None:
        return self.emit("TRADE_SIGNAL", {"action": action, **extra})

    def emit_feature_snapshot(
        self,
        *,
        trade_id: str,
        strategy: str,
        symbol: str | None = None,
        features: dict[str, Any] | None = None,
        **extra: Any,
    ) -> dict[str, Any] | None:
        payload = {
            "trade_id": trade_id,
            "strategy": strategy,
            "symbol": symbol,
            "features": features or {},
            **extra,
        }
        return self.emit("FEATURE_SNAPSHOT", payload)


_default_logger: TradeEventLogger | None = None


def get_trade_logger() -> TradeEventLogger:
    global _default_logger
    if _default_logger is None:
        _default_logger = TradeEventLogger()
    return _default_logger
