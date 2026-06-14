"""Registered trading strategies for the multi-regime platform."""

from __future__ import annotations

from typing import Literal

from strategies.base import StrategyResult
from strategies.base_strategy import BaseStrategy
from strategies.htf_trend_analyzer import HtfTrendResult, TrendDirection, analyze_htf_trend, is_counter_trend
from strategies.london_sweep_failure import LsfcSetup, LondonSweepFailureStrategy
from strategies.ttm import (
    STRATEGY_ABBREV as TTM_ABBREV,
    STRATEGY_FULL_NAME as TTM_FULL_NAME,
    TTM_PAIR_PRIMARY,
    TtmSetup,
    TtmStrategy,
)
from strategies.dbbs import (
    STRATEGY_ABBREV as DBBS_ABBREV,
    STRATEGY_FULL_NAME as DBBS_FULL_NAME,
    DbbsSetup,
    DbbsStrategy,
)
from strategies.dbbs_common import DBBSG_ABBREV, DBBSG_FULL_NAME, DBBSG_PAIR, DBBS_PAIR_XAU
from strategies.dinapoli import (
    STRATEGY_ABBREV as DINAPOLI_ABBREV,
    STRATEGY_FULL_NAME as DINAPOLI_FULL_NAME,
    DiNapoliSetup,
    DiNapoliStrategy,
)
from strategies.vamr import (
    STRATEGY_ABBREV as VAMR_ABBREV,
    STRATEGY_FULL_NAME as VAMR_FULL_NAME,
    VamrStrategy,
)
StrategyModeKey = Literal[
    "lsfc", "als", "fvg", "tref", "vexp", "dtpa", "cspa", "wyckoff", "lgr", "ttm", "dbbs", "dbbsg", "dinapoli", "vamr", "adre", "adre_v2",
    "continuation", "main", "all", "ab", "ac", "bc", "abc", "abcg", "abcd", "abcdn",
]

# 実運用 (MT5 Bridge) で発注可能な mode — A～Z letter 割当のみ。
STRATEGY_LETTER_BY_MODE: dict[str, str] = {
    "lsfc": "A",
    "dbbs": "B",
    "dinapoli": "C",
    "vamr": "D",
}
STRATEGY_LETTER_BY_SETUP_TYPE: dict[str, str] = {
    "LONDON_SWEEP_FAILURE_CONTINUATION": "A",
    "DBBS": "B",
    "DINAPOLI_STRUCTURE": "C",
    "VAMR": "D",
}
SETUP_TYPE_BY_STRATEGY_LETTER: dict[str, str] = {
    letter: setup_type for setup_type, letter in STRATEGY_LETTER_BY_SETUP_TYPE.items()
}
# B — dbbs (DBBS): Dual Bollinger Band Squeeze + Bear Kill Switch V2（EURUSD/GBPUSD/XAUUSD 本番採用）
# B — cspa (CSPA): アーカイブ — 検証の結果、プロップ用ポートフォリオには向いていない
# C — dinapoli (DN): DiNapoli Structure + DN Prop Gate V1
# D — vamr (VAMR): Volume Area Mean Reversion to POC（AUDNZD/EURGBP/USDCAD）
# D — ttm: 仲値流動性イベント特徴量収集（執行 M1 / 構造 M5 / ATR M15）
# H — wyckoff (WR): アーカイブ — 新戦略 Liquidity Grab Reversal (LGR) 構築に向けての発展的廃止
# I — lgr (LGR): アーカイブ — プロップ向きでない（自己資金口座向けに優秀）
# J — adre (ADRE): アーカイブ — プロップファーム向きではない（2026-06-13）

STRATEGY_ABBREV_BY_MODE: dict[str, str] = {
    "lsfc": "LSFC",
    "dbbs": DBBS_ABBREV,
    "dbbsg": DBBSG_ABBREV,
    "ttm": TTM_ABBREV,
    "dinapoli": DINAPOLI_ABBREV,
    "vamr": VAMR_ABBREV,
}

STRATEGY_FULL_NAME_BY_MODE: dict[str, str] = {
    "dbbs": DBBS_FULL_NAME,
    "dbbsg": DBBSG_FULL_NAME,
    "ttm": TTM_FULL_NAME,
    "dinapoli": DINAPOLI_FULL_NAME,
    "vamr": VAMR_FULL_NAME,
}

STRATEGY_PRIORITY_ORDER: tuple[str, ...] = (
    "lsfc", "dbbs", "dinapoli", "vamr", "ttm",
)

# (mode_key, strategy class) — archive 外の全実装。BT 用。
STRATEGY_CLASS_REGISTRY: tuple[tuple[str, type[BaseStrategy]], ...] = (
    ("lsfc", LondonSweepFailureStrategy),
    ("dbbs", DbbsStrategy),
    ("dbbsg", DbbsStrategy),
    ("ttm", TtmStrategy),
    ("dinapoli", DiNapoliStrategy),
    ("vamr", VamrStrategy),
)

MODE_BY_STRATEGY_CLASS: dict[type[BaseStrategy], str] = {
    cls: mode for mode, cls in STRATEGY_CLASS_REGISTRY
}

DEPRECATED_STRATEGY_MODES: frozenset[str] = frozenset()

ARCHIVED_STRATEGY_MODES: frozenset[str] = frozenset(
    {"als", "dtpa", "vexp", "continuation", "tref", "fvg", "wyckoff", "lgr", "cspa", "adre", "adre_v2"}
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


PORTFOLIO_AC_MODES: frozenset[str] = frozenset({"ac"})
PORTFOLIO_AB_MODES: frozenset[str] = frozenset({"ab"})
PORTFOLIO_BC_MODES: frozenset[str] = frozenset({"bc"})
PORTFOLIO_ABC_MODES: frozenset[str] = frozenset({"abc", "abcd", "abcdn"})
PORTFOLIO_ABCG_MODES: frozenset[str] = frozenset({"abcg"})
MTF_PORTFOLIO_MODES: frozenset[str] = (
    PORTFOLIO_AC_MODES
    | PORTFOLIO_AB_MODES
    | PORTFOLIO_BC_MODES
    | PORTFOLIO_ABC_MODES
    | PORTFOLIO_ABCG_MODES
)


def is_mtf_portfolio_mode(mode: str) -> bool:
    """True for shared-equity portfolio BT/WFT selectors (A+C production portfolio)."""
    return mode in MTF_PORTFOLIO_MODES


def portfolio_includes_dinapoli(mode: str) -> bool:
    """True when portfolio selector expands to a mode including DiNapoli."""
    return mode in (
        PORTFOLIO_ABC_MODES | PORTFOLIO_ABCG_MODES | PORTFOLIO_AC_MODES | PORTFOLIO_BC_MODES
    )


def portfolio_includes_vamr(mode: str) -> bool:
    """True when portfolio selector expands to a mode including VAMR (Strategy D)."""
    return mode in PORTFOLIO_ABC_MODES


def portfolio_includes_dbbsg(mode: str) -> bool:
    """Deprecated alias — DBBSG merged into DBBS (XAUUSD pair)."""
    return portfolio_includes_dbbs(mode)


def portfolio_includes_dbbs(mode: str) -> bool:
    """True when portfolio selector expands to a mode including DBBS."""
    return mode in (PORTFOLIO_AB_MODES | PORTFOLIO_ABC_MODES | PORTFOLIO_BC_MODES)


def portfolio_needs_h4(mode: str) -> bool:
    """True when portfolio BT/WFT requires H4 ATR data."""
    return portfolio_includes_dinapoli(mode)


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
        elif item in PORTFOLIO_ABC_MODES:
            expanded.extend(["lsfc", "dbbs", "dinapoli", "vamr"])
        elif item in PORTFOLIO_ABCG_MODES:
            expanded.extend(["lsfc", "dbbs", "dinapoli"])
        elif item in PORTFOLIO_AB_MODES:
            expanded.extend(["lsfc", "dbbs"])
        elif item in PORTFOLIO_AC_MODES:
            expanded.extend(["lsfc", "dinapoli"])
        elif item in PORTFOLIO_BC_MODES:
            expanded.extend(["dbbs", "dinapoli"])
        elif item == "dbbsg":
            expanded.append("dbbsg")
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
    "STRATEGY_LETTER_BY_SETUP_TYPE",
    "SETUP_TYPE_BY_STRATEGY_LETTER",
    "STRATEGY_PRIORITY_ORDER",
    "StrategyModeKey",
    "TrendDirection",
    "TTM_PAIR_PRIMARY",
    "TTM_ABBREV",
    "TTM_FULL_NAME",
    "TtmSetup",
    "TtmStrategy",
    "DbbsSetup",
    "DbbsStrategy",
    "DBBS_ABBREV",
    "DBBS_FULL_NAME",
    "DBBSG_ABBREV",
    "DBBSG_FULL_NAME",
    "DBBSG_PAIR",
    "DBBS_PAIR_XAU",
    "DiNapoliSetup",
    "DiNapoliStrategy",
    "DINAPOLI_FULL_NAME",
    "LondonSweepFailureStrategy",
    "analyze_htf_trend",
    "expand_strategy_modes",
    "get_live_strategies",
    "get_registered_strategies",
    "is_counter_trend",
    "is_live_strategy_mode",
    "is_mtf_portfolio_mode",
    "MTF_PORTFOLIO_MODES",
    "PORTFOLIO_ABC_MODES",
    "PORTFOLIO_ABCG_MODES",
    "PORTFOLIO_AB_MODES",
    "PORTFOLIO_AC_MODES",
    "PORTFOLIO_BC_MODES",
    "portfolio_includes_dbbs",
    "portfolio_includes_dbbsg",
    "portfolio_includes_dinapoli",
    "portfolio_includes_vamr",
    "portfolio_needs_h4",
    "VamrStrategy",
    "VAMR_ABBREV",
    "VAMR_FULL_NAME",
    "resolve_strategy_mode",
    "strategy_priority_index",
]
