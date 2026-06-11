"""Shared market/session utilities for LSFC and legacy continuation strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

PIP_SIZE = 0.0001
JPY_PIP_SIZE = 0.01
LONDON_SESSION_HOUR_START = 15
LONDON_SESSION_HOUR_END = 20
NY_ENTRY_HOUR = 21
LONDON_SESSION_HOURS = range(LONDON_SESSION_HOUR_START, LONDON_SESSION_HOUR_END + 1)

# BT / Live: 第1スロット（gbp_df）に載せるペア
PRIMARY_SLOT_PAIRS = frozenset({"GBPUSD", "AUDUSD", "AUDJPY"})
CORRELATED_PAIR = {
    "GBPUSD": "EURUSD",
    "EURUSD": "GBPUSD",
    "AUDUSD": "NZDUSD",
    "NZDUSD": "AUDUSD",
    "AUDJPY": "USDJPY",
    "USDJPY": "AUDJPY",
}


def pip_size_for_pair(pair: str) -> float:
    """ペアごとの pip サイズ（JPY クロスは 0.01、それ以外は 0.0001）。"""
    return JPY_PIP_SIZE if str(pair).upper().endswith("JPY") else PIP_SIZE


def uses_primary_dataframe(pair: str) -> bool:
    """相関ペアのうち第1 DataFrame スロット（gbp_df）に対応する side。"""
    return pair.upper() in PRIMARY_SLOT_PAIRS


def pair_dataframe_slot(
    pair: str,
    gbp_df: pd.DataFrame,
    eur_df: pd.DataFrame,
    *,
    setup_type: str | None = None,
) -> pd.DataFrame:
    """BT 用: ペアに対応する gbp/eur OHLCV スロットを返す。"""
    del setup_type
    return gbp_df if uses_primary_dataframe(pair) else eur_df


def correlated_pair(pair: str) -> str:
    """相関ペア名を返す（未定義時は入力をそのまま返す）。"""
    return CORRELATED_PAIR.get(pair.upper(), pair.upper())


class HasSweepDistance(Protocol):
    sweep_distance: float


@dataclass(frozen=True)
class SMTFeatures:
    intensity: float
    diff: float
    leader: str


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def positional_index(df: pd.DataFrame, index_label: Any) -> int:
    try:
        loc = df.index.get_loc(index_label)
    except KeyError:
        return int(index_label)
    if isinstance(loc, slice):
        return int(loc.start or 0)
    if isinstance(loc, np.ndarray):
        return int(loc[0])
    return int(loc)


def calc_smt_features(
    gbp_setup: HasSweepDistance | None,
    eur_setup: HasSweepDistance | None,
    pip_size: float = PIP_SIZE,
) -> SMTFeatures:
    gbp_pips = (gbp_setup.sweep_distance / pip_size) if gbp_setup else 0.0
    eur_pips = (eur_setup.sweep_distance / pip_size) if eur_setup else 0.0
    diff = gbp_pips - eur_pips
    intensity = abs(diff)

    def _leader_label(setup: HasSweepDistance | None) -> str:
        pair = getattr(setup, "pair", None) if setup else None
        if pair:
            return str(pair)[:3]
        return "UNK"

    if gbp_pips > eur_pips:
        leader = _leader_label(gbp_setup)
    elif eur_pips > gbp_pips:
        leader = _leader_label(eur_setup)
    else:
        leader = "NONE"
    return SMTFeatures(intensity=intensity, diff=diff, leader=leader)


def calc_smt_intensity(
    gbp_setup: HasSweepDistance | None,
    eur_setup: HasSweepDistance | None,
    pip_size: float = PIP_SIZE,
) -> float:
    return calc_smt_features(gbp_setup, eur_setup, pip_size).intensity
