"""
Portfolio Equity Trail — chronological replay on backtest trade logs.

Account-level PET is applied after merging strategy trade CSVs (ABCD + SMRS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from audit.risk_manager import STARTING_EQUITY
from core.portfolio_equity_trail import (
    PetRuntimeState,
    PortfolioEquityTrail,
    evaluate_pet,
    is_pet_enabled,
    load_pet_config,
    resolve_pet_mode,
)
from prop_audit_reporter import _apply_trade_equity


@dataclass(frozen=True)
class PetReplayStats:
    input_trades: int
    kept_trades: int
    blocked_trades: int
    pet_breach_events: int
    endgame_lot_scaled: int
    final_equity: float


def _executed_mask(df: pd.DataFrame) -> pd.Series:
    return df["trade_result"].isin(["WIN", "LOSS"])


def _lot_for_buffer(row: pd.Series) -> float:
    for col in ("final_lot_size", "lot_size"):
        if col in row.index and pd.notna(row[col]):
            val = float(row[col])
            if val > 0.0:
                return val
    lf = float(row.get("lot_factor", 0.01) or 0.01)
    return max(lf * 0.01, 0.01)


def replay_pet_on_trade_log(
    df: pd.DataFrame,
    *,
    profile: str = "challenge",
    config: dict[str, Any] | None = None,
    starting_equity: float = STARTING_EQUITY,
    log_events: bool = False,
) -> tuple[pd.DataFrame, PetReplayStats]:
    """
    Replay executed trades in timestamp order with PET entry gating (SOFT default).

    Blocked trades are removed from the output (account-level skip).
    Endgame risk multiplier scales lot_factor on allowed trades.
    """
    if not is_pet_enabled(profile):
        executed = int(_executed_mask(df).sum())
        return df.copy(), PetReplayStats(
            input_trades=executed,
            kept_trades=executed,
            blocked_trades=0,
            pet_breach_events=0,
            endgame_lot_scaled=0,
            final_equity=starting_equity,
        )

    cfg = config or load_pet_config()
    engine = PortfolioEquityTrail(config=cfg, mode=resolve_pet_mode())
    if not log_events:
        engine.log_event = lambda *args, **kwargs: None  # type: ignore[method-assign]

    work = df.sort_values("timestamp").copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    executed_idx = work.index[_executed_mask(work)].tolist()

    pet_state = PetRuntimeState.create(starting_equity)
    equity = starting_equity
    phase_start = starting_equity
    daily_start = starting_equity
    server_day = ""

    kept: list[int] = []
    blocked = 0
    breaches = 0
    endgame_scaled = 0

    for idx in executed_idx:
        row = work.loc[idx]
        ts = pd.Timestamp(row["timestamp"])
        day = ts.strftime("%Y-%m-%d")
        if day != server_day:
            server_day = day
            daily_start = equity

        open_positions = [
            {
                "ticket": str(row.get("trade_id", idx)),
                "setup_type": str(row.get("setup_type", "")),
                "lot_size": _lot_for_buffer(row),
                "bayes_probability": float(row.get("bayes_probability", 0.5) or 0.5),
                "expected_r": float(row.get("profit_r", 1.0) or 1.0),
                "risk_contribution": _lot_for_buffer(row),
            }
        ]

        decision = engine.evaluate(
            pet_state,
            current_equity=equity,
            balance=equity,
            server_day=server_day,
            open_positions=open_positions,
            phase_start_equity=phase_start,
            day_start_equity=daily_start,
        )

        if decision.disable_new_entries:
            blocked += 1
            if decision.breached:
                breaches += 1
            continue

        lot_factor = float(row.get("lot_factor", 1.0) or 1.0)
        if decision.endgame.active and decision.risk_multiplier < 1.0:
            lot_factor = round(lot_factor * decision.risk_multiplier, 6)
            endgame_scaled += 1
            work.at[idx, "lot_factor"] = lot_factor
            if "pet_endgame_lot_scaled" not in work.columns:
                work["pet_endgame_lot_scaled"] = 0
            work.at[idx, "pet_endgame_lot_scaled"] = 1

        profit_r = float(row.get("profit_r", 0.0) or 0.0)
        equity_before = equity
        equity = _apply_trade_equity(equity, profit_r, lot_factor, profile, phase_start)
        work.at[idx, "equity_before_trade"] = round(equity_before, 2)
        work.at[idx, "equity_after_trade"] = round(equity, 2)
        kept.append(idx)

    not_executed = work.index[~_executed_mask(work)].tolist()
    out_idx = sorted(set(kept + not_executed))
    out = work.loc[out_idx].sort_values("timestamp").reset_index(drop=True)
    if "pet_applied" not in out.columns:
        out["pet_applied"] = 1

    stats = PetReplayStats(
        input_trades=len(executed_idx),
        kept_trades=len(kept),
        blocked_trades=blocked,
        pet_breach_events=breaches,
        endgame_lot_scaled=endgame_scaled,
        final_equity=round(equity, 2),
    )
    return out, stats
