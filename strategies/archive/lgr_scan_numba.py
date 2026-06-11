"""Optional Numba kernels for LGR precompute scan (``LGR_SCAN_NUMBA=1``)."""

from __future__ import annotations

import numpy as np

from strategies.scan_numba_util import njit, scan_numba_active

_SESSION_LONDON = 0
_SESSION_NY = 1
_SESSION_ASIA = 2
_SESSION_OTHER = 3


def lgr_scan_numba_enabled() -> bool:
    return scan_numba_active("LGR_SCAN_NUMBA", default=False)


def lgr_scan_numba_active() -> bool:
    return lgr_scan_numba_enabled()


def session_type_to_code(session: str) -> int:
    if session == "LONDON":
        return _SESSION_LONDON
    if session == "NY":
        return _SESSION_NY
    if session == "ASIA":
        return _SESSION_ASIA
    return _SESSION_OTHER


@njit(cache=True)
def _bar_in_session_numba(hour: int, session_code: int) -> bool:
    if session_code == _SESSION_LONDON:
        return 8 <= hour < 17
    if session_code == _SESSION_NY:
        return 13 <= hour < 22
    if session_code == _SESSION_ASIA:
        return hour < 8
    return True


@njit(cache=True)
def prior_daily_extremes_numba(
    high: np.ndarray,
    low: np.ndarray,
    day_norm: np.ndarray,
    idx: int,
) -> tuple:
    if idx < 1:
        return -1.0, -1.0, 0
    current_day = day_norm[idx]
    hi = -1.0e100
    lo = 1.0e100
    found = 0
    for j in range(idx):
        if day_norm[j] != current_day:
            continue
        found = 1
        if high[j] > hi:
            hi = high[j]
        if low[j] < lo:
            lo = low[j]
    if found == 0:
        return -1.0, -1.0, 0
    return hi, lo, 1


@njit(cache=True)
def prior_session_extremes_numba(
    high: np.ndarray,
    low: np.ndarray,
    day_norm: np.ndarray,
    hours: np.ndarray,
    idx: int,
    session_code: int,
) -> tuple:
    if idx < 1:
        return -1.0, -1.0, 0
    current_day = day_norm[idx]
    hi = -1.0e100
    lo = 1.0e100
    found = 0
    for j in range(idx):
        if day_norm[j] != current_day:
            continue
        if not _bar_in_session_numba(int(hours[j]), session_code):
            continue
        found = 1
        if high[j] > hi:
            hi = high[j]
        if low[j] < lo:
            lo = low[j]
    if found == 0:
        for j in range(idx):
            if day_norm[j] != current_day:
                continue
            found = 1
            if high[j] > hi:
                hi = high[j]
            if low[j] < lo:
                lo = low[j]
        if found == 0:
            return -1.0, -1.0, 0
    return hi, lo, 1


__all__ = [
    "lgr_scan_numba_active",
    "lgr_scan_numba_enabled",
    "prior_daily_extremes_numba",
    "prior_session_extremes_numba",
    "session_type_to_code",
]
