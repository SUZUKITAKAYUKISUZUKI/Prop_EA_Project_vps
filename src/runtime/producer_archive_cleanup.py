"""VPS-side removal of old gzip event archives (one-way producer cleanup)."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_ARCHIVE_DATE_RE = re.compile(r"events_(\d{8})\.jsonl\.gz$")


def cleanup_producer_archives(output_dir: Path, *, retention_days: int) -> int:
    """Delete old .jsonl.gz on VPS after retention."""
    if retention_days <= 0 or not output_dir.exists():
        return 0
    cutoff = datetime.now(timezone.utc).date() - timedelta(days=retention_days)
    deleted = 0
    for path in sorted(output_dir.glob("events_*.jsonl.gz")):
        match = _ARCHIVE_DATE_RE.match(path.name)
        if not match:
            continue
        try:
            file_date = datetime.strptime(match.group(1), "%Y%m%d").date()
        except ValueError:
            continue
        if file_date >= cutoff:
            continue
        path.unlink(missing_ok=True)
        deleted += 1
        logger.info("Producer archive cleanup removed %s", path.name)
    return deleted
