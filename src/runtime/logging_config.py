"""Dropbox JSONL logging configuration."""

from __future__ import annotations



import os

from dataclasses import dataclass

from pathlib import Path

from typing import Any, Literal



from src.database.config_loader import load_yaml_config



PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "dropbox_logging.yaml"



DataFlowRole = Literal["producer", "consumer"]





class DataFlowRoleError(RuntimeError):

    """Raised when a component runs on the wrong machine role."""





@dataclass(frozen=True)

class DropboxCleanupConfig:

    enabled: bool = True

    modify_synced_files: bool = False

    delete_fully_imported: bool = True

    truncate_today_file: bool = True





@dataclass(frozen=True)

class ProducerArchiveConfig:

    delete_gz_after_days: int = 0





@dataclass(frozen=True)

class DataFlowConfig:

    role: DataFlowRole = "producer"



    @property

    def is_producer(self) -> bool:

        return self.role == "producer"



    @property

    def is_consumer(self) -> bool:

        return self.role == "consumer"





@dataclass(frozen=True)

class DropboxLoggingConfig:

    enabled: bool

    output_dir: Path

    watch_dir: Path

    rotation: str

    compress_old_files: bool

    filename_pattern: str

    poll_interval_seconds: int

    batch_size: int

    dedupe: bool

    data_flow: DataFlowConfig

    cleanup: DropboxCleanupConfig

    producer: ProducerArchiveConfig



    @property

    def write_enabled(self) -> bool:

        return self.enabled and self.data_flow.is_producer



    @property

    def import_enabled(self) -> bool:

        return self.data_flow.is_consumer





def _parse_role(raw: str | None) -> DataFlowRole:

    env_role = os.environ.get("DROPBOX_DATA_FLOW_ROLE", "").strip().lower()

    role = env_role or (raw or "producer").strip().lower()

    if role not in {"producer", "consumer"}:

        raise ValueError(f"Invalid data_flow.role={role!r}; expected producer or consumer")

    return role  # type: ignore[return-value]





def load_dropbox_logging_config(path: Path | None = None) -> DropboxLoggingConfig:

    cfg_path = path or DEFAULT_CONFIG

    raw = load_yaml_config(cfg_path) if cfg_path.exists() else {}

    env_dir = os.environ.get("DROPBOX_EVENTS_DIR")

    output = Path(env_dir) if env_dir else Path(raw.get("output_dir", "C:/Dropbox/PortfolioOS/events"))

    watch = Path(os.environ.get("DROPBOX_EVENTS_WATCH_DIR", raw.get("watch_dir", str(output))))

    import_cfg = raw.get("import", {}) if isinstance(raw.get("import"), dict) else {}

    cleanup_raw = raw.get("cleanup", {}) if isinstance(raw.get("cleanup"), dict) else {}

    data_flow_raw = raw.get("data_flow", {}) if isinstance(raw.get("data_flow"), dict) else {}

    producer_raw = raw.get("producer", {}) if isinstance(raw.get("producer"), dict) else {}



    role = _parse_role(str(data_flow_raw.get("role", "producer")))

    data_flow = DataFlowConfig(role=role)



    modify_synced_files = cleanup_raw.get("modify_synced_files")

    if modify_synced_files is None:

        modify_synced_files = False



    cleanup = DropboxCleanupConfig(

        enabled=bool(cleanup_raw.get("enabled", True)),

        modify_synced_files=bool(modify_synced_files),

        delete_fully_imported=bool(cleanup_raw.get("delete_fully_imported", True)),

        truncate_today_file=bool(cleanup_raw.get("truncate_today_file", True)),

    )

    producer = ProducerArchiveConfig(

        delete_gz_after_days=int(producer_raw.get("delete_gz_after_days", 0)),

    )

    return DropboxLoggingConfig(

        enabled=bool(raw.get("enabled", True)),

        output_dir=output,

        watch_dir=watch,

        rotation=str(raw.get("rotation", "daily")),

        compress_old_files=bool(raw.get("compress_old_files", True)),

        filename_pattern=str(raw.get("filename_pattern", "events_{date}.jsonl")),

        poll_interval_seconds=int(raw.get("poll_interval_seconds", 10)),

        batch_size=int(import_cfg.get("batch_size", 500)),

        dedupe=bool(import_cfg.get("dedupe", True)),

        data_flow=data_flow,

        cleanup=cleanup,

        producer=producer,

    )





def require_producer(config: DropboxLoggingConfig, *, component: str) -> None:

    if not config.data_flow.is_producer:

        raise DataFlowRoleError(

            f"{component} requires data_flow.role=producer (VPS). "

            f"Current role={config.data_flow.role!r}."

        )





def require_consumer(config: DropboxLoggingConfig, *, component: str) -> None:

    if not config.data_flow.is_consumer:

        raise DataFlowRoleError(

            f"{component} requires data_flow.role=consumer (local PC). "

            f"Current role={config.data_flow.role!r}."

        )


