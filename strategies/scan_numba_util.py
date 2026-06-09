"""Shared Numba availability and env-flag helpers for BT scan kernels."""

from __future__ import annotations

import os

_NUMBA_OK = False
try:
    from numba import njit  # noqa: F401

    _NUMBA_OK = True
except ImportError:

    def njit(*args, **kwargs):  # type: ignore[misc]
        def decorator(fn):
            return fn

        if args and callable(args[0]):
            return args[0]
        return decorator


def numba_available() -> bool:
    return _NUMBA_OK


def env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def scan_numba_active(env_name: str, *, default: bool = False) -> bool:
    if not env_flag(env_name, default=default):
        return False
    return numba_available()
