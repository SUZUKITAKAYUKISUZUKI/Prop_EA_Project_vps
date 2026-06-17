"""Shared base-entry duplicate guard for Python-side MT5 execution."""

from __future__ import annotations

from typing import Any

from strategies import SETUP_TYPE_BY_STRATEGY_LETTER
from strategies.market_utils import normalize_pair_name


def extract_strategy_letter(comment: str) -> str:
    token = (comment or "").strip()
    if not token.startswith("PropEA_") or len(token) < 8:
        return ""
    return token[7:8].upper()


def is_pyramid_comment(comment: str) -> bool:
    return (comment or "").startswith("PropEA_PYR_")


def resolve_setup_type(
    *,
    setup_type: str = "",
    strategy_letter: str = "",
) -> str:
    setup_u = (setup_type or "").strip()
    if setup_u:
        return setup_u
    letter = (strategy_letter or "").strip().upper()
    if letter:
        return SETUP_TYPE_BY_STRATEGY_LETTER.get(letter, "")
    return ""


def has_open_base_position_for_strategy(
    positions: list[Any],
    *,
    pair: str,
    magic: int,
    strategy_letter: str = "",
    symbol_matches: Any | None = None,
) -> bool:
    """
    Return True when a non-pyramid base position already exists for pair+magic.

    symbol_matches(symbol, pair) defaults to canonical prefix match.
    """
    canonical = normalize_pair_name(pair) or pair.strip().upper()
    letter_u = (strategy_letter or "").strip().upper()

    def _symbol_matches(symbol: str) -> bool:
        if symbol_matches is not None:
            return bool(symbol_matches(symbol, canonical))
        norm = normalize_pair_name(symbol)
        if norm is not None:
            return norm == canonical
        return str(symbol).upper().startswith(canonical)

    for pos in positions:
        pos_magic = int(getattr(pos, "magic", 0) or 0)
        if pos_magic != int(magic):
            continue
        symbol = str(getattr(pos, "symbol", "") or "")
        if not _symbol_matches(symbol):
            continue
        comment = str(getattr(pos, "comment", "") or "")
        if is_pyramid_comment(comment):
            continue
        if letter_u == "":
            return True
        if extract_strategy_letter(comment) == letter_u:
            return True
    return False


def live_base_entry_blocked_message(
    *,
    pair: str,
    strategy_letter: str = "",
) -> str:
    if strategy_letter:
        return (
            f"Base position already open for {pair} strategy={strategy_letter} "
            "(live_position_guard)"
        )
    return f"Base position already open for {pair} (live_position_guard)"
