"""
strategies/cspa_scan_parallel.py — Phase 3 (joblib pair) & Phase 4 (year chunks).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from strategies.archive.cspa_arrays import years_from_datetime_ns

from strategies.bt_scan_parallel import parallel_map_pairs as _parallel_map_pairs
from strategies.bt_scan_parallel import scan_parallel_jobs
from strategies.archive.cspa_scan_engine import scan_parallel_years_enabled

logger = logging.getLogger(__name__)


def year_bar_ranges(datetime_ns: np.ndarray) -> list[tuple[int, int, int]]:
    """
    Calendar-year index ranges on M1 trigger bars.

    Returns list of (year, start_index, end_index_exclusive).
    """
    if len(datetime_ns) == 0:
        return []
    years = years_from_datetime_ns(datetime_ns)
    ranges: list[tuple[int, int, int]] = []
    start = 0
    current_year = int(years[0])
    for i in range(1, len(years)):
        y = int(years[i])
        if y != current_year:
            ranges.append((current_year, start, i))
            start = i
            current_year = y
    ranges.append((current_year, start, len(years)))
    return ranges


def merge_scan_setups(chunks: list[list[Any]]) -> list[Any]:
    merged: list[Any] = []
    for part in chunks:
        merged.extend(part)
    merged.sort(key=lambda s: (s.timestamp, s.bar_index))
    return merged


def parallel_map_pairs(
    worker: Callable[..., Any],
    job_kwargs: list[dict[str, Any]],
    *,
    enabled: bool,
    backend: str = "loky",
) -> list[Any]:
    """Phase 3: run one job per currency pair."""
    return _parallel_map_pairs(
        worker,
        job_kwargs,
        enabled=enabled,
        backend=backend,
        log_prefix="CSPA Phase 3",
    )


def parallel_scan_by_year(
    scan_fn: Callable[..., list[Any]],
    *,
    trigger_dt: np.ndarray,
    enabled: bool | None = None,
    **scan_kwargs: Any,
) -> list[Any]:
    """
    Phase 4: scan each calendar year sequentially merged (parallel scaffold).

    Note: ``last_signal_bar`` cooldown may differ vs single-pass at year boundaries
    when parallel workers run independently — enable only for throughput experiments
    until boundary carry-over is implemented.
    """
    use_parallel = scan_parallel_years_enabled() if enabled is None else enabled
    ranges = year_bar_ranges(trigger_dt)
    if len(ranges) <= 1:
        return scan_fn(**scan_kwargs)

    if not use_parallel:
        out: list[Any] = []
        last_signal = scan_kwargs.get("initial_last_signal_bar", -999)
        for _year, start, end in ranges:
            chunk = scan_fn(
                **{
                    **scan_kwargs,
                    "resume_from_bar": max(scan_kwargs.get("resume_from_bar") or 0, start),
                    "loop_end": end,
                    "initial_last_signal_bar": last_signal,
                    "initial_setups": out,
                }
            )
            out = chunk
            if out:
                last_signal = out[-1].bar_index
        return out

    try:
        from joblib import Parallel, delayed
    except ImportError:
        logger.warning("joblib not installed — sequential year scan fallback")
        return parallel_scan_by_year(scan_fn, enabled=False, **scan_kwargs)

    logger.info("CSPA Phase 4: parallel year chunks (%d years)", len(ranges))
    n_jobs = scan_parallel_jobs()
    year_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(scan_fn)(
            **{
                **scan_kwargs,
                "resume_from_bar": start,
                "loop_end": end,
                "initial_setups": [],
                "initial_last_signal_bar": -999,
            }
        )
        for _year, start, end in ranges
    )
    return merge_scan_setups(year_results)
