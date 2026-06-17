"""Live pyramid runtime configuration (Limit-order execution path)."""

from __future__ import annotations

import os
from typing import Any

from pyramid_manager import _env_flag, resolve_max_pyramid_layers


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return float(raw)


LIVE_PYRAMID_LIMIT_TTL_BARS = int(os.getenv("LIVE_PYRAMID_LIMIT_TTL_BARS", "1"))
LIVE_PYRAMID_USE_MARKET_FALLBACK = _env_flag("LIVE_PYRAMID_USE_MARKET_FALLBACK") is True
LIVE_PYRAMID_TRIGGER_R = _env_float("LIVE_PYRAMID_TRIGGER_R", 1.0)


def live_pyramid_be_after_tp_enabled() -> bool:
    explicit = _env_flag("LIVE_PYRAMID_BE_AFTER_TP")
    if explicit is None:
        return True
    return explicit


def resolve_live_pyramid_trigger_r(setup_type: str | None = None) -> float:
    """Arm breakeven only after TP distance when live TP cap is enabled."""
    base = LIVE_PYRAMID_TRIGGER_R
    if live_pyramid_be_after_tp_enabled():
        from audit.live_tp_cap import is_live_tp_cap_enabled, live_tp_max_r

        if is_live_tp_cap_enabled():
            buffer = _env_float("LIVE_PYRAMID_BE_TP_BUFFER_R", 0.05)
            return max(base, live_tp_max_r(setup_type) + buffer)
    return base


def live_pyramid_env_enabled() -> bool:
    explicit = _env_flag("LIVE_PYRAMID_ENABLED")
    if explicit is not None:
        return explicit
    # VPS 既定: PYRAMID_ENABLED=1 なら Live ブリッジも ON（明示 0 でのみ OFF）
    return _env_flag("PYRAMID_ENABLED") is True


# 後方互換: import 時スナップショット（startup ログ用）。判定は live_pyramid_env_enabled() を使用。
LIVE_PYRAMID_ENABLED = live_pyramid_env_enabled()


def is_live_pyramid_enabled(setup_type: str | None = None) -> bool:
    """LIVE_PYRAMID_ENABLED=1 かつ pyramid_manager.is_pyramid_enabled の両方を満たす場合のみ True。"""
    if not live_pyramid_env_enabled():
        return False
    from pyramid_manager import is_pyramid_enabled

    return is_pyramid_enabled(setup_type)


def live_pyramid_strategy_status() -> list[dict[str, Any]]:
    """L5 ピラミッド状態 + LIVE_PYRAMID_ENABLED を含む監査スナップショット。"""
    from pyramid_manager import get_pyramid_strategy_status

    live_on = live_pyramid_env_enabled()
    return [
        {
            **row,
            "live_pyramid_enabled": live_on and row["effective_enabled"],
        }
        for row in get_pyramid_strategy_status()
    ]


def resolve_live_max_pyramid_layers(setup_type: str | None = None) -> int:
    return resolve_max_pyramid_layers(setup_type)
