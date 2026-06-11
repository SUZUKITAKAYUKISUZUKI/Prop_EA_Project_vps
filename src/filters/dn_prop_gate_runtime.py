"""Runtime DN Prop Gate V1 scoring from pipeline / feature-log context."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.filters.dn_bayes_ev_v2 import apply_dn_ev_v2_to_row, get_default_dn_ev_v2_model
from src.filters.dn_prop_gate_v1 import evaluate_dn_prop_gate_sizing, is_dn_prop_gate_enabled
from strategies.dinapoli import DiNapoliSetup
from strategies.dinapoli_feature_log import DnFeatureMarketContext, build_dn_entry_features


def score_dn_prop_gate_from_setup(
    *,
    setup: DiNapoliSetup,
    trade_id: str,
    decision_source: str,
    llm_confidence: int,
    llm_reason: str,
    minutes_to_news: int,
    m15_df: pd.DataFrame,
    h1_df: pd.DataFrame,
    h4_df: pd.DataFrame | None = None,
    run_id: str = "runtime",
    wft_window: int = -1,
    is_oos: int = 0,
) -> dict[str, Any]:
    row = build_dn_entry_features(
        setup=setup,
        trade_id=trade_id,
        run_id=run_id,
        decision_source=decision_source,
        llm_confidence=llm_confidence,
        llm_reason=llm_reason,
        minutes_to_news=minutes_to_news,
        ctx=DnFeatureMarketContext(exec_m15=m15_df, h1=h1_df, h4=h4_df),
        wft_window=wft_window,
        is_oos=is_oos,
    )
    apply_dn_ev_v2_to_row(row, model=get_default_dn_ev_v2_model())
    row["ev_bucket"] = evaluate_dn_prop_gate_sizing(row)["tier"]
    gate = evaluate_dn_prop_gate_sizing(row)
    row.update(
        {
            "dn_prop_gate_tier": gate["tier"],
            "dn_prop_gate_lot_multiplier": gate["lot_multiplier"],
            "dn_prop_gate_pattern": gate["pattern"],
            "dn_prop_gate_base_risk_pct": gate["base_risk_pct"],
        }
    )
    return row


def prop_gate_enabled() -> bool:
    return is_dn_prop_gate_enabled()
