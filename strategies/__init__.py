"""Registered trading strategies for the multi-regime platform."""

from __future__ import annotations

from typing import Literal

from strategies.base import StrategyResult
from strategies.base_strategy import BaseStrategy
from strategies.htf_trend_analyzer import HtfTrendResult, TrendDirection, analyze_htf_trend, is_counter_trend
from strategies.london_sweep_failure import LsfcSetup, LondonSweepFailureStrategy
from strategies.cspa import (
    CSPA_PAIR_PRIMARY,
    CSPA_PAIR_SECONDARY,
    CspaSetup,
    CspaStrategy,
    STRATEGY_FULL_NAME as CSPA_FULL_NAME,
)
from strategies.liquidity_grab_reversal import (
    LGR_PAIR_PRIMARY,
    LGR_PAIR_SECONDARY,
    LgrSetup,
    STRATEGY_ABBREV as LGR_ABBREV,
    STRATEGY_FULL_NAME as LGR_FULL_NAME,
    LiquidityGrabReversalStrategy,
)

StrategyModeKey = Literal[
    "lsfc", "als", "fvg", "tref", "vexp", "dtpa", "cspa", "wyckoff", "lgr",
    "continuation", "main", "all", "abc", "abcd",
]

# 実運用 (MT5 Bridge) で発注可能な mode — A～Z letter 割当のみ。
STRATEGY_LETTER_BY_MODE: dict[str, str] = {
    "lsfc": "A",
    "cspa": "B",
}
# C, D — 未割当（旧 FVG / TREF。アーカイブ）
# H — wyckoff (WR): アーカイブ — 新戦略 Liquidity Grab Reversal (LGR) 構築に向けての発展的廃止
# 旧 WS は archive/wyckoff_spring.py — 戦略に優位性が無いことが判明したから
# I — lgr (LGR): BT/WFT のみ（letter 未割当）

STRATEGY_ABBREV_BY_MODE: dict[str, str] = {
    "lsfc": "LSFC",
    "cspa": "CSPA",
    "lgr": LGR_ABBREV,
}

STRATEGY_FULL_NAME_BY_MODE: dict[str, str] = {
    "cspa": CSPA_FULL_NAME,
    "lgr": LGR_FULL_NAME,
}

STRATEGY_PRIORITY_ORDER: tuple[str, ...] = (
    "lsfc", "cspa", "lgr",
)

# (mode_key, strategy class) — archive 外の全実装。BT 用。
STRATEGY_CLASS_REGISTRY: tuple[tuple[str, type[BaseStrategy]], ...] = (
    ("lsfc", LondonSweepFailureStrategy),
    ("cspa", CspaStrategy),
    ("lgr", LiquidityGrabReversalStrategy),
)

MODE_BY_STRATEGY_CLASS: dict[type[BaseStrategy], str] = {
    cls: mode for mode, cls in STRATEGY_CLASS_REGISTRY
}

DEPRECATED_STRATEGY_MODES: frozenset[str] = frozenset()

ARCHIVED_STRATEGY_MODES: frozenset[str] = frozenset(
    {"als", "dtpa", "vexp", "continuation", "tref", "fvg", "wyckoff"}
)

PRODUCTION_STRATEGY_MODE: StrategyModeKey = "lsfc"


def strategy_priority_index(mode: str) -> int:
    try:
        return STRATEGY_PRIORITY_ORDER.index(mode)
    except ValueError:
        return len(STRATEGY_PRIORITY_ORDER)


def is_live_strategy_mode(mode: str) -> bool:
    """True if mode has an A–Z letter and is not archived (eligible for MT5 live)."""
    return mode in STRATEGY_LETTER_BY_MODE and mode not in ARCHIVED_STRATEGY_MODES


def resolve_strategy_mode(strategy: BaseStrategy) -> str:
    """Map a strategy instance to its mode key."""
    for mode, cls in STRATEGY_CLASS_REGISTRY:
        if isinstance(strategy, cls):
            return mode
    raise ValueError(f"Unknown strategy instance: {type(strategy).__name__}")


def expand_strategy_modes(
    strategies: StrategyModeKey | tuple[StrategyModeKey, ...] | str,
) -> tuple[str, ...]:
    """Expand CLI / API strategy selectors into concrete mode keys."""
    if isinstance(strategies, str):
        strategies = (strategies,)
    expanded: list[str] = []
    for item in strategies:
        if item == "main":
            expanded.append("lsfc")
        elif item == "all":
            expanded.append("lsfc")
        elif item in ("abc", "abcd"):
            expanded.extend(["lsfc", "cspa"])
        else:
            expanded.append(item)
    return tuple(expanded)


def get_registered_strategies(
    weights_config: dict[str, int],
    mode_h1: bool,
    *,
    live_only: bool = False,
) -> list[BaseStrategy]:
    """
    Return strategy instances.

    live_only=True: letter 割当 (STRATEGY_LETTER_BY_MODE) のみ — MT5 Bridge 用。
    live_only=False: archive 外の全実装 — BT / WFT 用。
    """
    instances: list[BaseStrategy] = []
    for mode, cls in STRATEGY_CLASS_REGISTRY:
        if mode in ARCHIVED_STRATEGY_MODES:
            continue
        if live_only and not is_live_strategy_mode(mode):
            continue
        instances.append(cls(weights_config=weights_config, mode_h1=mode_h1))
    return instances


def get_live_strategies(
    weights_config: dict[str, int],
    mode_h1: bool,
) -> list[BaseStrategy]:
    """MT5 Bridge / 実運用で発注可能な戦略のみ。"""
    return get_registered_strategies(weights_config, mode_h1, live_only=True)


__all__ = [
    "ARCHIVED_STRATEGY_MODES",
    "BaseStrategy",
    "StrategyResult",
    "DEPRECATED_STRATEGY_MODES",
    "HtfTrendResult",
    "LsfcSetup",
    "PRODUCTION_STRATEGY_MODE",
    "STRATEGY_ABBREV_BY_MODE",
    "STRATEGY_CLASS_REGISTRY",
    "STRATEGY_FULL_NAME_BY_MODE",
    "STRATEGY_LETTER_BY_MODE",
    "STRATEGY_PRIORITY_ORDER",
    "StrategyModeKey",
    "TrendDirection",
    "CSPA_FULL_NAME",
    "CSPA_PAIR_PRIMARY",
    "CSPA_PAIR_SECONDARY",
    "CspaSetup",
    "CspaStrategy",
    "LGR_PAIR_PRIMARY",
    "LGR_PAIR_SECONDARY",
    "LGR_ABBREV",
    "LGR_FULL_NAME",
    "LgrSetup",
    "LiquidityGrabReversalStrategy",
    "LondonSweepFailureStrategy",
    "analyze_htf_trend",
    "expand_strategy_modes",
    "get_live_strategies",
    "get_registered_strategies",
    "is_counter_trend",
    "is_live_strategy_mode",
    "resolve_strategy_mode",
    "strategy_priority_index",
]
