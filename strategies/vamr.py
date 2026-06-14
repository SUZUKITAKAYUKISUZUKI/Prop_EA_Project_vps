"""
strategies/vamr.py — Volume Area Mean Reversion to POC (VAMR): Strategy D.

Production spec:
- Signal: Pattern 5 (frozen)
- Bayes: frozen model (vamr_bayes_v1.json)
- Sizing: Model B
- L4 Gemini: OFF
- Defenses: Profit Cushion + Twin Brake + DD Throttling ON
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from strategies.base import StrategyResult
from strategies.base_strategy import BaseStrategy
from strategies.market_utils import pip_size_for_pair
from strategies.var_detector import VP_TOUCH_ATR_MULT
from strategies.var_reversal import VarSetup, detect_var_setups
from strategies.vamr_features import (
    add_derived_features,
    primary_pa_type,
    session_type_utc,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_JSON = (
    Path(__file__).resolve().parents[1] / "backtest_results" / "models" / "vamr_bayes_v1.json"
)
PRODUCTION_SIZING_MODEL = "B"

SETUP_TYPE = "VAMR"
STRATEGY_ABBREV = "VAMR"
STRATEGY_FULL_NAME = "Volume Area Mean Reversion to POC"
VAMR_PAIRS = frozenset({"AUDNZD", "EURGBP", "USDCAD"})
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
VAMR_PAIR_FILES: dict[str, tuple[str, str, str]] = {
    "AUDNZD": ("audnzd_h1_10y.csv", "audnzd_m5_10y.csv", "audnzd_h4_10y.csv"),
    "EURGBP": ("eurgbp_h1_10y.csv", "eurgbp_m5_10y.csv", "eurgbp_h4_10y.csv"),
    "USDCAD": ("usdcad_h1_10y.csv", "usdcad_m5_10y.csv", "usdcad_h4_10y.csv"),
}


def is_vamr_pure_bt_mode() -> bool:
    return os.getenv("VAMR_PURE_BT", "0").strip().lower() in ("1", "true", "yes", "on")


def is_vamr_defense_mode() -> bool:
    if is_vamr_pure_bt_mode():
        return False
    flag = os.getenv("VAMR_DEFENSE", "1").strip().lower()
    return flag in ("1", "true", "yes", "on")


def is_vamr_defense_pure_mode() -> bool:
    return is_vamr_pure_bt_mode()


def is_vamr_l4_bypass() -> bool:
    return is_vamr_pure_bt_mode() or is_vamr_defense_mode()


def is_vamr_generic_bayes_bypass() -> bool:
    return is_vamr_pure_bt_mode() or is_vamr_defense_mode()


def configure_vamr_defense_env() -> None:
    if is_vamr_pure_bt_mode():
        return
    os.environ.setdefault("VAMR_DEFENSE", "1")
    os.environ.setdefault("VAMR_GEMINI_AUDIT", "0")
    os.environ.setdefault("LLM_AUDIT_ENABLED", "0")
    os.environ.setdefault("PROFIT_CUSHION_ENABLED", "1")
    os.environ.setdefault("TWIN_BRAKE_ENABLED", "1")
    os.environ.setdefault("DD_THROTTLING_ENABLED", "1")


@lru_cache(maxsize=1)
def get_vamr_bayes_model():
    from strategies.vamr_bayes import load_vamr_bayes_model

    path = Path(os.getenv("VAMR_BAYES_MODEL", str(DEFAULT_MODEL_JSON)))
    if not path.exists():
        raise FileNotFoundError(f"VAMR Bayes model not found: {path}")
    return load_vamr_bayes_model(path)


@lru_cache(maxsize=len(VAMR_PAIRS))
def _load_pair_h1_h4(pair: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from main_platform import load_ohlcv

    pair_u = str(pair).upper()
    h1_name, _, h4_name = VAMR_PAIR_FILES[pair_u]
    h1 = load_ohlcv(DATA_DIR / h1_name)
    h4 = load_ohlcv(DATA_DIR / h4_name)
    for frame in (h1, h4):
        frame["datetime"] = pd.to_datetime(frame["datetime"])
    return h1.reset_index(drop=True), h4.reset_index(drop=True)


def _session_start_ts(ts: pd.Timestamp) -> pd.Timestamp:
    day = pd.Timestamp(ts.date())
    start = day + pd.Timedelta(hours=7)
    if ts < start:
        start -= pd.Timedelta(days=1)
    return start


def enrich_vamr_setup_row(setup: VarSetup) -> dict[str, Any]:
    pair = str(setup.pair).upper()
    h1, h4 = _load_pair_h1_h4(pair)
    bar_i = int(setup.bar_index)
    if bar_i < 0 or bar_i >= len(h1):
        raise IndexError(f"VAMR bar_index out of range for {pair}: {bar_i}")

    atr = float(setup.atr_h1)
    vah = float(setup.vah)
    val = float(setup.val)
    direction = str(setup.direction)
    touch_buf = max(VP_TOUCH_ATR_MULT * atr, pip_size_for_pair(pair) * 2.0)

    h1_high = h1["high"].to_numpy(dtype=np.float64)
    h1_low = h1["low"].to_numpy(dtype=np.float64)
    h1_close = h1["close"].to_numpy(dtype=np.float64)
    h1_vol = h1["volume"].to_numpy(dtype=np.float64)
    h4_close = h4["close"].to_numpy(dtype=np.float64)
    h4_dt = pd.to_datetime(h4["datetime"]).to_numpy()
    h1_dt = pd.to_datetime(h1["datetime"]).to_numpy()

    sess_start = _session_start_ts(pd.Timestamp(setup.timestamp))
    sess_start_i = int(np.searchsorted(h1_dt, np.datetime64(sess_start), side="left"))
    start_i = max(sess_start_i, 0)
    touches = 0
    for j in range(start_i, bar_i + 1):
        if direction == "SELL" and h1_high[j] >= vah - touch_buf:
            touches += 1
        elif direction == "BUY" and h1_low[j] <= val + touch_buf:
            touches += 1

    hi = h1_high[bar_i]
    lo = h1_low[bar_i]
    cl = h1_close[bar_i]
    rng = hi - lo
    if rng > 0.0:
        rejection_strength = (hi - cl) / rng if direction == "SELL" else (cl - lo) / rng
    else:
        rejection_strength = 0.0

    vol_start = max(0, bar_i - 19)
    base = float(np.mean(h1_vol[vol_start : bar_i + 1]))
    volume_ratio_20ma = float(h1_vol[bar_i] / base) if base > 0 else 1.0

    entry_ts = np.datetime64(pd.Timestamp(setup.timestamp))
    h4_i = int(np.searchsorted(h4_dt, entry_ts, side="right")) - 1
    htf_aligned = "unknown"
    if h4_i >= 20:
        ma20 = float(np.mean(h4_close[h4_i - 19 : h4_i + 1]))
        htf_bull = h4_close[h4_i] > ma20
        if direction == "BUY":
            htf_aligned = "aligned" if htf_bull else "not_aligned"
        else:
            htf_aligned = "aligned" if not htf_bull else "not_aligned"

    row = setup.as_dict()
    row.update(
        {
            "timestamp": setup.timestamp,
            "pair": pair,
            "retest_count": touches,
            "rejection_strength": float(rejection_strength),
            "volume_ratio_20ma": volume_ratio_20ma,
            "htf_aligned": htf_aligned,
            "session_type": session_type_utc(setup.timestamp),
            "primary_pa_type": primary_pa_type(setup.pa_types),
            "vp_touch_side": setup.vp_touch_side,
            "atr_percentile_100b": setup.atr_percentile_100b,
        }
    )
    return add_derived_features(pd.DataFrame([row])).iloc[0].to_dict()


def passes_pattern5_row(row: Mapping[str, Any]) -> bool:
    from strategies.vamr_bayes import FROZEN_PATTERN5_THRESHOLDS, PATTERN5_SPEC
    from strategies.vamr_phase2 import apply_pattern_mask

    work = pd.DataFrame([dict(row)])
    work = add_derived_features(work)
    mask = apply_pattern_mask(work, PATTERN5_SPEC, FROZEN_PATTERN5_THRESHOLDS)
    return bool(mask.iloc[0])


def detect_vamr_setups_for_pair(
    *,
    h1_df: pd.DataFrame,
    m5_df: pd.DataFrame,
    pair: str,
    h4_df: pd.DataFrame | None = None,
) -> list[VarSetup]:
    del h4_df
    from strategies.bt_ohlcv import BtOhlcvFrame

    pair_u = str(pair).upper()
    if pair_u not in VAMR_PAIRS:
        return []
    h1_use = h1_df.to_pandas() if isinstance(h1_df, BtOhlcvFrame) else h1_df
    m5_use = m5_df.to_pandas() if isinstance(m5_df, BtOhlcvFrame) else m5_df
    setups = detect_var_setups(h1_use, pair_u, m5_df=m5_use)
    return [s for s in setups if s.tp_target == "POC"]


def detect_vamr_setups(
    df: pd.DataFrame,
    pair_name: str,
    h1_df: pd.DataFrame | None = None,
    **kwargs: Any,
) -> list[VarSetup]:
    exec_df = h1_df if h1_df is not None else df
    return detect_vamr_setups_for_pair(
        h1_df=exec_df,
        m5_df=kwargs.get("m5_df"),
        pair=pair_name,
        h4_df=kwargs.get("h4_df"),
    )


class VamrStrategy(BaseStrategy):
    """VAMR production — Pattern 5 + frozen Bayes + Model B sizing."""

    @property
    def setup_type(self) -> str:
        return SETUP_TYPE

    def detect_setups(
        self,
        df: pd.DataFrame,
        pair_name: str,
        h1_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> list[VarSetup]:
        return detect_vamr_setups(df, pair_name, h1_df=h1_df, **kwargs)

    def analyze_setup(
        self,
        setup: VarSetup,
        gbp_setup: Any | None,
        eur_setup: Any | None,
        h1_gbp: pd.DataFrame,
        h1_eur: pd.DataFrame,
    ) -> StrategyResult:
        del gbp_setup, eur_setup, h1_gbp, h1_eur
        reject = StrategyResult(
            is_setup=False,
            setup_type=SETUP_TYPE,
            direction="FLAT",
            strategy_action="REJECT",
        )
        if not isinstance(setup, VarSetup):
            return reject
        if str(setup.pair).upper() not in VAMR_PAIRS or setup.tp_target != "POC":
            return reject

        try:
            row = enrich_vamr_setup_row(setup)
        except (IndexError, KeyError, FileNotFoundError) as exc:
            logger.debug("VAMR enrich skip %s: %s", setup.pair, exc)
            return reject

        if not passes_pattern5_row(row):
            return StrategyResult(
                is_setup=True,
                setup_type=SETUP_TYPE,
                direction=setup.direction,
                entry_price=setup.entry_price,
                stop_loss=setup.stop_loss,
                take_profit=setup.take_profit,
                candidate_score=0.0,
                raw_features={**row, "reject_reason": "PATTERN5"},
                strategy_action="REJECT",
                base_risk_pct=0.0,
            )

        if is_vamr_pure_bt_mode():
            prob = 1.0
            size_mult = 1.0
        else:
            from strategies.vamr_bayes import predict_win_probability
            from strategies.vamr_sizing import model_b_multiplier

            prob, _ = predict_win_probability(row, model=get_vamr_bayes_model())
            size_mult = model_b_multiplier(prob)

        if size_mult <= 0.0:
            return StrategyResult(
                is_setup=True,
                setup_type=SETUP_TYPE,
                direction=setup.direction,
                entry_price=setup.entry_price,
                stop_loss=setup.stop_loss,
                take_profit=setup.take_profit,
                candidate_score=0.0,
                raw_features={
                    **row,
                    "bayes_probability": prob,
                    "vamr_size_multiplier": 0.0,
                    "reject_reason": "SIZING_SKIP",
                },
                strategy_action="REJECT",
                base_risk_pct=0.0,
            )

        raw = dict(row)
        raw.update(
            {
                "setup_type": SETUP_TYPE,
                "bayes_probability": prob,
                "vamr_size_multiplier": size_mult,
                "decision_source": "ALLOW",
                "sizing_model": PRODUCTION_SIZING_MODEL,
                "smt_intensity": 0.0,
                "smt_diff": 0.0,
                "smt_leader": "NONE",
                "has_bos": False,
                "both_sweep": False,
                "atr_ratio": float(raw.get("atr_vs_session_avg", 1.0) or 1.0),
                "htf_trend_direction": "NEUTRAL",
                "htf_counter_trend": False,
                "htf_lot_multiplier": 1.0,
            }
        )
        return StrategyResult(
            is_setup=True,
            setup_type=SETUP_TYPE,
            direction=setup.direction,
            entry_price=setup.entry_price,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            candidate_score=0.0,
            raw_features=raw,
            strategy_action="ALLOW",
            base_risk_pct=None,
        )


__all__ = [
    "SETUP_TYPE",
    "STRATEGY_ABBREV",
    "STRATEGY_FULL_NAME",
    "VAMR_PAIR_FILES",
    "VAMR_PAIRS",
    "VamrStrategy",
    "configure_vamr_defense_env",
    "detect_vamr_setups",
    "detect_vamr_setups_for_pair",
    "enrich_vamr_setup_row",
    "get_vamr_bayes_model",
    "is_vamr_defense_mode",
    "is_vamr_defense_pure_mode",
    "is_vamr_generic_bayes_bypass",
    "is_vamr_l4_bypass",
    "is_vamr_pure_bt_mode",
    "passes_pattern5_row",
]
