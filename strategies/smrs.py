"""
strategies/smrs.py — Statistical Mean Reversion Scalper (SMRS): Strategy E.

Production spec:
- Signal: Phase S1 frozen baseline (M1)
- Bayes: frozen model (smrs_bayes_v1.json)
- Sizing: Model A
- L4 Gemini: OFF
- Pyramiding: OFF
- Defenses: Profit Cushion + Twin Brake + DD Throttling ON
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from strategies.base import StrategyResult
from strategies.base_strategy import BaseStrategy
from strategies.smrs_bayes import load_smrs_bayes_model, predict_win_probability
from strategies.smrs_production import (
    DEFAULT_BAYES_MODEL_JSON,
    PRODUCTION_MIN_M1_BARS,
    PRODUCTION_PARAMS,
    PRODUCTION_SIZING_MODEL,
    SMRS_BAYES_MODEL_ENV,
    SMRS_DEFENSE_DEFAULT,
    SMRS_DEFENSE_ENV,
    SMRS_PURE_BT_DEFAULT,
    SMRS_PURE_BT_ENV,
    configure_smrs_defense_env,
)
from strategies.smrs_pure import (
    SETUP_TYPE,
    SMRS_PAIRS,
    STRATEGY_ABBREV,
    STRATEGY_DISPLAY_NAME,
    STRATEGY_FULL_NAME,
    STRATEGY_PORTFOLIO_LABEL,
    build_feature_rows,
)
from strategies.smrs_scan_numba import ENVELOPE_DEVS, prep_pair_arrays, simulate_smrs_log

logger = logging.getLogger(__name__)

DEV_INDEX_MAP = {float(ENVELOPE_DEVS[i]): i for i in range(len(ENVELOPE_DEVS))}
MIN_M1_BARS = PRODUCTION_MIN_M1_BARS


def _env_on(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def is_smrs_pure_bt_mode() -> bool:
    return _env_on(SMRS_PURE_BT_ENV, SMRS_PURE_BT_DEFAULT)


def is_smrs_defense_mode() -> bool:
    if is_smrs_pure_bt_mode():
        return False
    return _env_on(SMRS_DEFENSE_ENV, SMRS_DEFENSE_DEFAULT)


def is_smrs_defense_pure_mode() -> bool:
    return is_smrs_pure_bt_mode()


def is_smrs_l4_bypass() -> bool:
    return is_smrs_pure_bt_mode() or is_smrs_defense_mode()


def is_smrs_generic_bayes_bypass() -> bool:
    return is_smrs_pure_bt_mode() or is_smrs_defense_mode()


@lru_cache(maxsize=1)
def get_smrs_bayes_model():
    path = Path(os.getenv(SMRS_BAYES_MODEL_ENV, str(DEFAULT_BAYES_MODEL_JSON)))
    if not path.exists():
        raise FileNotFoundError(f"SMRS Bayes model not found: {path}")
    return load_smrs_bayes_model(path)


@dataclass(frozen=True)
class SmrsSetup:
    timestamp: pd.Timestamp
    pair: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    bar_index: int
    z_score_entry: float
    ma_entry: float
    upper_envelope_entry: float
    lower_envelope_entry: float
    atr_entry: float
    atr_p25_entry: float
    atr_p50_entry: float
    atr_p75_entry: float
    session: str
    setup_type: str = SETUP_TYPE

    @property
    def pair_name(self) -> str:
        return self.pair

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _dev_index(params=PRODUCTION_PARAMS) -> int:
    dev_i = DEV_INDEX_MAP.get(params.envelope_dev)
    if dev_i is None:
        dev_i = int(np.argmin(np.abs(ENVELOPE_DEVS - params.envelope_dev)))
    return dev_i


def _df_to_scan_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    work = df.sort_values("datetime").reset_index(drop=True)
    ts = pd.to_datetime(work["datetime"]).astype("int64").to_numpy(dtype=np.int64)
    open_ = work["open"].to_numpy(dtype=np.float64)
    high = work["high"].to_numpy(dtype=np.float64)
    low = work["low"].to_numpy(dtype=np.float64)
    close = work["close"].to_numpy(dtype=np.float64)
    return ts, open_, high, low, close


def enrich_smrs_setup_row(setup: SmrsSetup) -> dict[str, Any]:
    hour = int(setup.timestamp.hour)
    atr_p50 = float(setup.atr_p50_entry)
    atr_ratio = float(setup.atr_entry / atr_p50) if atr_p50 > 0.0 else 1.0
    row = setup.as_dict()
    row.update(
        {
            "timestamp": setup.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "pair": str(setup.pair).upper(),
            "direction": str(setup.direction).upper(),
            "session": setup.session,
            "hour": hour,
            "z_score_entry": setup.z_score_entry,
            "atr_entry": setup.atr_entry,
            "atr_p50_entry": atr_p50,
            "atr_ratio": atr_ratio,
            "pair_direction": f"{str(setup.pair).upper()}_{str(setup.direction).upper()}",
            "bar_index_entry": setup.bar_index,
        }
    )
    return row


def _row_to_setup(row: dict[str, Any], *, bar_index: int) -> SmrsSetup:
    ts = pd.Timestamp(row["timestamp"])
    tp_raw = row.get("take_profit", "")
    take_profit = float(tp_raw) if tp_raw not in ("", None) else 0.0
    return SmrsSetup(
        timestamp=ts,
        pair=str(row["pair"]).upper(),
        direction=str(row["direction"]).upper(),
        entry_price=float(row["entry_price"]),
        stop_loss=float(row["stop_loss"]),
        take_profit=take_profit,
        bar_index=bar_index,
        z_score_entry=float(row["z_score_entry"]),
        ma_entry=float(row["ma_entry"]),
        upper_envelope_entry=float(row["upper_envelope_entry"]),
        lower_envelope_entry=float(row["lower_envelope_entry"]),
        atr_entry=float(row["atr_entry"]),
        atr_p25_entry=float(row["atr_p25_entry"]),
        atr_p50_entry=float(row["atr_p50_entry"]),
        atr_p75_entry=float(row["atr_p75_entry"]),
        session=str(row["session"]),
    )


def detect_smrs_setups_from_m1(
    m1_df: pd.DataFrame,
    pair: str,
    *,
    params=PRODUCTION_PARAMS,
) -> list[SmrsSetup]:
    pair_u = str(pair).upper()
    if pair_u not in SMRS_PAIRS or m1_df.empty or len(m1_df) < MIN_M1_BARS:
        return []

    ts, open_, high, low, close = _df_to_scan_arrays(m1_df)
    hours, ma, z, atr, atr_p25, atr_p50, atr_p75, upper_all, lower_all = prep_pair_arrays(
        open_, high, low, close, ts
    )
    dev_i = _dev_index(params)
    log = simulate_smrs_log(
        ts,
        high,
        low,
        close,
        hours,
        ma,
        z,
        upper_all[dev_i],
        lower_all[dev_i],
        atr,
        atr_p25,
        atr_p50,
        atr_p75,
        params.z_threshold,
        params.session_code,
        params.atr_code,
        params.exit_code,
        params.max_hold_hours,
    )
    rows = build_feature_rows(pair_u, params, log)
    ts_to_index = {
        pd.Timestamp(t).floor("s"): idx
        for idx, t in enumerate(pd.to_datetime(m1_df["datetime"]))
    }
    setups: list[SmrsSetup] = []
    for row in rows:
        ts_key = pd.Timestamp(row["timestamp"]).floor("s")
        bar_index = ts_to_index.get(ts_key)
        if bar_index is None:
            continue
        setups.append(_row_to_setup(row, bar_index=int(bar_index)))
    return setups


def detect_smrs_setups(
    df: pd.DataFrame,
    pair_name: str,
    h1_df: pd.DataFrame | None = None,
    **kwargs: Any,
) -> list[SmrsSetup]:
    del h1_df, kwargs
    return detect_smrs_setups_from_m1(df, pair_name)


class SmrsStrategy(BaseStrategy):
    """SMRS production — frozen M1 signal + Bayes + Model A sizing."""

    @property
    def setup_type(self) -> str:
        return SETUP_TYPE

    def detect_setups(
        self,
        df: pd.DataFrame,
        pair_name: str,
        h1_df: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> list[SmrsSetup]:
        return detect_smrs_setups(df, pair_name, h1_df=h1_df, **kwargs)

    def analyze_setup(
        self,
        setup: SmrsSetup,
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
        if not isinstance(setup, SmrsSetup):
            return reject
        if str(setup.pair).upper() not in SMRS_PAIRS:
            return reject

        try:
            row = enrich_smrs_setup_row(setup)
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            logger.debug("SMRS enrich skip %s: %s", setup.pair, exc)
            return reject

        if is_smrs_pure_bt_mode():
            prob = 1.0
            size_mult = 1.0
        else:
            from strategies.smrs_sizing import model_a_multiplier

            prob, _ = predict_win_probability(row, model=get_smrs_bayes_model())
            size_mult = model_a_multiplier(prob)

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
                    "smrs_size_multiplier": 0.0,
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
                "smrs_size_multiplier": size_mult,
                "decision_source": "ALLOW",
                "sizing_model": PRODUCTION_SIZING_MODEL,
                "gemini_audit": "OFF",
                "llm_audit": "OFF",
                "smt_intensity": 0.0,
                "smt_diff": 0.0,
                "smt_leader": "NONE",
                "has_bos": False,
                "both_sweep": False,
                "atr_ratio": float(raw.get("atr_ratio", 1.0) or 1.0),
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
    "STRATEGY_DISPLAY_NAME",
    "STRATEGY_FULL_NAME",
    "STRATEGY_PORTFOLIO_LABEL",
    "SMRS_PAIRS",
    "SmrsSetup",
    "SmrsStrategy",
    "configure_smrs_defense_env",
    "detect_smrs_setups",
    "detect_smrs_setups_from_m1",
    "enrich_smrs_setup_row",
    "get_smrs_bayes_model",
    "is_smrs_defense_mode",
    "is_smrs_defense_pure_mode",
    "is_smrs_generic_bayes_bypass",
    "is_smrs_l4_bypass",
    "is_smrs_pure_bt_mode",
]
