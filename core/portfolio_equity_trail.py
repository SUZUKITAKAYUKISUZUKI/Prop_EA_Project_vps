"""
Portfolio Equity Trail (PET) v1 — Phase 5.2 account-level profit protection.

Independent from trade-level trailing. Protects accumulated portfolio gains in R-space
with Fintokei execution-cost buffers.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from audit.broker_costs import FINTOKEI_MIN_NET_PROFIT_USD
from audit.risk_manager import STARTING_EQUITY, challenge_profit_progress_pct
from core.pet_endgame_mode import PetEndgameDecision, evaluate_pet_endgame
from core.pet_position_ranker import select_lowest_ranked
from core.pet_stage_manager import (
    compute_locked_profit_r,
    compute_protected_equity,
    effective_lock_fraction,
    execution_buffer_usd,
    load_pet_stages,
    resolve_stage,
)
from core.prop_profiles import PropProfile, get_profile, load_prop_profiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PetMode = Literal["SOFT", "MEDIUM", "HARD"]

PET_EVENT_COLUMNS = [
    "timestamp",
    "peak_equity",
    "protected_equity",
    "current_equity",
    "peak_gain_r",
    "locked_profit_r",
    "stage",
    "mode",
    "action",
    "breached",
    "challenge_progress_pct",
    "endgame_active",
]


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("0", "false", "off", "no", "disabled"):
        return False
    if raw in ("1", "true", "yes", "on", "enabled"):
        return True
    return default


def _resolve_profile(profile: str | None = None) -> str:
    from audit.risk_manager import normalize_profile

    if profile:
        return normalize_profile(profile)
    env = os.environ.get("PROP_FIRM_PROFILE", "").strip().lower()
    if env in ("challenge", "funded"):
        return env
    return "challenge"


def default_pet_enabled_for_profile(profile: str) -> bool:
    """Challenge: PET OFF by default. Funded: PET ON by default."""
    cfg = load_pet_config()
    defaults = cfg.get("profile_defaults") or {}
    prof = _resolve_profile(profile)
    if prof == "funded":
        return bool(defaults.get("funded", True))
    return bool(defaults.get("challenge", False))


def is_pet_enabled(profile: str | None = None) -> bool:
    """
    Return whether PET is active for the given prop profile.

    Priority: explicit PET_ENABLED env > profile default (challenge OFF / funded ON)
    > configs/pet_config.json ``enabled`` flag.
    """
    raw = os.environ.get("PET_ENABLED", "").strip().lower()
    if raw in ("0", "false", "off", "no", "disabled"):
        return False
    if raw in ("1", "true", "yes", "on", "enabled"):
        return True
    if not default_pet_enabled_for_profile(_resolve_profile(profile)):
        return False
    cfg = load_pet_config()
    return bool(cfg.get("enabled", True))


def load_pet_config(path: Path | str | None = None) -> dict[str, Any]:
    if path is None:
        env_path = os.environ.get("PET_CONFIG_PATH", "").strip()
        path = Path(env_path) if env_path else PROJECT_ROOT / "configs" / "pet_config.json"
    else:
        path = Path(path)
    if not path.exists():
        return {"enabled": False}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_r_unit_usd(day_start_equity: float, config: dict[str, Any]) -> float:
    explicit = float(config.get("r_unit_usd") or 0.0)
    if explicit > 0.0:
        return explicit
    pct = float(config.get("base_risk_pct_for_r_unit", 0.01))
    base = day_start_equity if day_start_equity > 0 else STARTING_EQUITY
    return round(base * pct, 4)


def equity_gain_r(current_equity: float, day_start_equity: float, r_unit_usd: float) -> float:
    if r_unit_usd <= 0.0:
        return 0.0
    return round((current_equity - day_start_equity) / r_unit_usd, 4)


@dataclass
class PetRuntimeState:
    day_start_equity: float = STARTING_EQUITY
    peak_equity: float = STARTING_EQUITY
    daily_peak_equity: float = STARTING_EQUITY
    peak_gain_r: float = 0.0
    protected_equity: float = STARTING_EQUITY
    locked_profit_r: float = 0.0
    stage: int = 0
    stage_name: str = "OFF"
    active: bool = False
    breached: bool = False
    disable_new_entries: bool = False
    trading_halted_for_day: bool = False
    server_day: str = ""
    last_action: str = "INIT"

    @classmethod
    def create(cls, equity: float = STARTING_EQUITY) -> PetRuntimeState:
        return cls(
            day_start_equity=equity,
            peak_equity=equity,
            daily_peak_equity=equity,
            protected_equity=equity,
            server_day="",
        )

    def reset_daily(self, equity: float, server_day: str) -> None:
        self.day_start_equity = equity
        self.peak_equity = equity
        self.daily_peak_equity = equity
        self.peak_gain_r = 0.0
        self.protected_equity = equity
        self.locked_profit_r = 0.0
        self.stage = 0
        self.stage_name = "OFF"
        self.active = False
        self.breached = False
        self.disable_new_entries = False
        self.trading_halted_for_day = False
        self.server_day = server_day
        self.last_action = "DAILY_RESET"


@dataclass(frozen=True)
class PetDecision:
    active: bool
    stage: int
    stage_name: str
    current_equity: float
    peak_equity: float
    protected_equity: float
    locked_profit_r: float
    equity_gain_r: float
    peak_gain_r: float
    disable_new_entries: bool
    breached: bool
    close_all: bool
    close_tickets: tuple[str, ...]
    endgame: PetEndgameDecision
    bayes_threshold_min: float
    risk_multiplier: float
    mode: PetMode
    action: str
    message: str
    execution_buffer_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "stage": self.stage,
            "stage_name": self.stage_name,
            "current_equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "protected_equity": self.protected_equity,
            "locked_profit_r": self.locked_profit_r,
            "equity_gain_r": self.equity_gain_r,
            "peak_gain_r": self.peak_gain_r,
            "disable_new_entries": self.disable_new_entries,
            "breached": self.breached,
            "close_all": self.close_all,
            "close_tickets": list(self.close_tickets),
            "endgame_active": self.endgame.active,
            "bayes_threshold_min": self.bayes_threshold_min,
            "risk_multiplier": self.risk_multiplier,
            "mode": self.mode,
            "action": self.action,
            "message": self.message,
            "execution_buffer_usd": self.execution_buffer_usd,
        }


@dataclass
class PortfolioEquityTrail:
    config: dict[str, Any] = field(default_factory=load_pet_config)
    profile: PropProfile = field(default_factory=lambda: get_profile("Fintokei_100K"))
    mode: PetMode = "SOFT"
    log_path: Path = field(default_factory=lambda: PROJECT_ROOT / "logs" / "pet_events.csv")

    def __post_init__(self) -> None:
        log_cfg = self.config.get("logging") or {}
        self.log_path = Path(log_cfg.get("event_log_path", self.log_path))

    def _cost_cfg(self) -> dict[str, Any]:
        costs = self.config.get("execution_costs") or {}
        per_lot = float(costs.get("execution_buffer_per_lot_usd", 8.0))
        if per_lot <= 0.0:
            per_lot = (
                float(costs.get("commission_per_lot_round_trip_usd", 6.0))
                + float(costs.get("avg_spread_cost_usd", 1.0))
                + float(costs.get("slippage_cost_usd", 1.0))
            )
        costs = dict(costs)
        costs["execution_buffer_per_lot_usd"] = per_lot
        return costs

    def _total_lot(self, open_positions: list[dict[str, Any]] | None) -> float:
        if not open_positions:
            return float(self.config.get("default_reference_lot", 0.01))
        total = 0.0
        for pos in open_positions:
            total += float(pos.get("lot_size", pos.get("volume", 0.0)) or 0.0)
        return max(total, float(self.config.get("default_reference_lot", 0.01)))

    def evaluate(
        self,
        state: PetRuntimeState,
        *,
        current_equity: float,
        balance: float,
        server_day: str,
        open_positions: list[dict[str, Any]] | None = None,
        challenge_progress_pct: float | None = None,
        phase_start_equity: float | None = None,
        day_start_equity: float | None = None,
    ) -> PetDecision:
        del balance
        if server_day:
            if state.server_day and server_day != state.server_day:
                state.reset_daily(current_equity, server_day)
            elif not state.server_day:
                state.server_day = server_day
                if day_start_equity is not None and day_start_equity > 0.0:
                    state.day_start_equity = day_start_equity
                state.peak_equity = max(state.peak_equity, current_equity)
                state.daily_peak_equity = max(state.daily_peak_equity, current_equity)

        r_unit = resolve_r_unit_usd(state.day_start_equity, self.config)
        pet_start_r = float(self.config.get("pet_start_r", 3.0))
        stages = load_pet_stages(self.config)
        costs = self._cost_cfg()
        min_net = float(self.config.get("min_net_profit_usd", FINTOKEI_MIN_NET_PROFIT_USD))
        total_lot = self._total_lot(open_positions)
        exec_buffer = execution_buffer_usd(
            total_lot,
            per_lot_usd=float(costs["execution_buffer_per_lot_usd"]),
            min_net_profit_usd=min_net,
        )

        if current_equity > state.peak_equity:
            state.peak_equity = current_equity
        if current_equity > state.daily_peak_equity:
            state.daily_peak_equity = current_equity

        gain_r = equity_gain_r(current_equity, state.day_start_equity, r_unit)
        peak_gain_r = equity_gain_r(state.peak_equity, state.day_start_equity, r_unit)
        state.peak_gain_r = peak_gain_r

        phase_start = phase_start_equity or state.day_start_equity
        if challenge_progress_pct is None:
            challenge_progress_pct = challenge_profit_progress_pct(phase_start, current_equity)

        target_profit = float(self.profile.target_profit)
        peak_profit_pct = challenge_profit_progress_pct(phase_start, state.peak_equity)
        target_progress_pct = (
            peak_profit_pct / target_profit * 100.0 if target_profit > 0.0 else 0.0
        )

        endgame = evaluate_pet_endgame(
            challenge_progress_pct,
            target_profit_pct=target_profit,
            config=self.config,
        )
        challenge_cfg = self.config.get("challenge_mode") or {}
        stage = resolve_stage(peak_gain_r, stages)
        lock_fraction = effective_lock_fraction(
            stage,
            challenge_progress_pct=target_progress_pct,
            challenge_trigger_pct=float(challenge_cfg.get("progress_trigger_pct", 50.0)),
            challenge_lock_bonus=float(challenge_cfg.get("lock_fraction_bonus", 0.15)),
            endgame_lock_multiplier=endgame.lock_multiplier,
            endgame_active=endgame.active,
        )
        locked_profit_r = compute_locked_profit_r(peak_gain_r, lock_fraction)

        if bool(self.config.get("enable_daily_pet", True)):
            daily_peak_gain_r = equity_gain_r(state.daily_peak_equity, state.day_start_equity, r_unit)
            daily_stage = resolve_stage(daily_peak_gain_r, stages)
            daily_lock_fraction = effective_lock_fraction(
                daily_stage,
                challenge_progress_pct=target_progress_pct,
                challenge_trigger_pct=float(challenge_cfg.get("progress_trigger_pct", 50.0)),
                challenge_lock_bonus=float(challenge_cfg.get("lock_fraction_bonus", 0.15)),
                endgame_lock_multiplier=endgame.lock_multiplier,
                endgame_active=endgame.active,
            )
            daily_locked_r = compute_locked_profit_r(daily_peak_gain_r, daily_lock_fraction)
            locked_profit_r = max(locked_profit_r, daily_locked_r)

        protected = compute_protected_equity(
            state.day_start_equity,
            locked_profit_r,
            r_unit,
            exec_buffer,
        )
        active = peak_gain_r >= pet_start_r and locked_profit_r > 0.0
        breached = active and current_equity < protected

        state.stage = stage.stage
        state.stage_name = stage.name
        state.active = active
        state.locked_profit_r = locked_profit_r
        state.protected_equity = protected
        state.breached = breached

        close_all = False
        close_tickets: list[str] = []
        disable_entries = state.trading_halted_for_day
        action = "NONE"
        message = f"PET {stage.name} gain={gain_r:.2f}R peak={peak_gain_r:.2f}R"

        if not active:
            message = f"PET OFF gain={gain_r:.2f}R (< {pet_start_r:.1f}R start)"
        elif breached:
            disable_entries = True
            state.trading_halted_for_day = bool(self.config.get("enable_daily_pet", True))
            if self.mode == "HARD":
                close_all = True
                action = "CLOSE_ALL"
                message = f"PET HARD breach equity={current_equity:.2f} < protected={protected:.2f}"
            elif self.mode == "MEDIUM":
                close_tickets = select_lowest_ranked(open_positions or [], count=1)
                action = "CLOSE_LOWEST_RANKED"
                message = (
                    f"PET MEDIUM breach — close lowest ranked | "
                    f"equity={current_equity:.2f} protected={protected:.2f}"
                )
            else:
                action = "DISABLE_NEW_ENTRIES"
                message = (
                    f"PET SOFT breach — entries blocked | "
                    f"equity={current_equity:.2f} protected={protected:.2f}"
                )
        elif active:
            action = "PROTECT_ACTIVE"
            message = (
                f"PET {stage.name} locked={locked_profit_r:.2f}R "
                f"protected={protected:.2f}"
            )

        state.disable_new_entries = disable_entries
        state.last_action = action
        state.breached = breached

        decision = PetDecision(
            active=active,
            stage=stage.stage,
            stage_name=stage.name,
            current_equity=current_equity,
            peak_equity=state.peak_equity,
            protected_equity=protected,
            locked_profit_r=locked_profit_r,
            equity_gain_r=gain_r,
            peak_gain_r=peak_gain_r,
            disable_new_entries=disable_entries,
            breached=breached,
            close_all=close_all,
            close_tickets=tuple(close_tickets),
            endgame=endgame,
            bayes_threshold_min=endgame.bayes_threshold_delta if endgame.active else 0.0,
            risk_multiplier=endgame.risk_multiplier if endgame.active else 1.0,
            mode=self.mode,
            action=action,
            message=message,
            execution_buffer_usd=exec_buffer,
        )
        self.log_event(state, decision)
        return decision

    def log_event(self, state: PetRuntimeState, decision: PetDecision) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.log_path.exists()
        row = {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "peak_equity": decision.peak_equity,
            "protected_equity": decision.protected_equity,
            "current_equity": decision.current_equity,
            "peak_gain_r": decision.peak_gain_r,
            "locked_profit_r": decision.locked_profit_r,
            "stage": decision.stage_name,
            "mode": decision.mode,
            "action": decision.action,
            "breached": int(decision.breached),
            "challenge_progress_pct": round(
                challenge_profit_progress_pct(state.day_start_equity, decision.current_equity),
                4,
            ),
            "endgame_active": int(decision.endgame.active),
        }
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=PET_EVENT_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def evaluate_pet(
    state: PetRuntimeState,
    *,
    current_equity: float,
    balance: float,
    server_day: str,
    open_positions: list[dict[str, Any]] | None = None,
    challenge_progress_pct: float | None = None,
    phase_start_equity: float | None = None,
    day_start_equity: float | None = None,
    config: dict[str, Any] | None = None,
    mode: PetMode | None = None,
) -> PetDecision:
    engine = PortfolioEquityTrail(config=config or load_pet_config())
    engine.mode = mode if mode is not None else resolve_pet_mode()
    return engine.evaluate(
        state,
        current_equity=current_equity,
        balance=balance,
        server_day=server_day,
        open_positions=open_positions,
        challenge_progress_pct=challenge_progress_pct,
        phase_start_equity=phase_start_equity,
        day_start_equity=day_start_equity,
    )


def pet_hold_signal(message: str, *, tags: tuple[str, ...] = ("PET",)) -> dict[str, Any]:
    return {
        "action": "HOLD",
        "lot_size": 0.0,
        "risk_budget": 0.0,
        "sl": 0.0,
        "tp": 0.0,
        "entry": 0.0,
        "message": message,
        "decision_source": "PET",
        "tags": list(tags),
    }


def resolve_pet_mode() -> PetMode:
    raw = os.environ.get("PET_MODE", "").strip().upper()
    if raw in ("SOFT", "MEDIUM", "HARD"):
        return raw  # type: ignore[return-value]
    return str(load_pet_config().get("default_mode", "SOFT")).upper()  # type: ignore[return-value]


def apply_pet_to_trade_signal(
    signal: dict[str, Any],
    decision: PetDecision | None,
) -> dict[str, Any]:
    """Merge PET decision into an MT5 trade_signal dict."""
    if decision is None:
        return signal
    out = dict(signal)
    out["pet"] = decision.to_dict()
    if decision.endgame.active and out.get("lot_size"):
        out["lot_size"] = round(float(out["lot_size"]) * decision.risk_multiplier, 4)
        rb = out.get("risk_budget")
        if rb:
            out["risk_budget"] = round(float(rb) * decision.risk_multiplier, 4)
    if decision.close_tickets:
        out["pet_close_tickets"] = list(decision.close_tickets)
    if decision.close_all:
        return pet_panic_signal(decision.message, tags=("PET", "PANIC_CLOSE"))
    if decision.disable_new_entries and out.get("action") in ("BUY", "SELL"):
        blocked = pet_hold_signal(decision.message, tags=("PET", decision.action))
        blocked["pet"] = decision.to_dict()
        if decision.close_tickets:
            blocked["pet_close_tickets"] = list(decision.close_tickets)
        return blocked
    return out


def pet_panic_signal(message: str, *, tags: tuple[str, ...] = ("PET", "PANIC_CLOSE")) -> dict[str, Any]:
    return {
        "action": "PANIC_CLOSE",
        "lot_size": 0.0,
        "risk_budget": 0.0,
        "sl": 0.0,
        "tp": 0.0,
        "entry": 0.0,
        "message": message,
        "decision_source": "PET",
        "tags": list(tags),
    }


@dataclass(frozen=True)
class PetMonteCarloResult:
    trials: int
    pass_rate: float
    fail_rate: float
    risk_of_ruin: float
    avg_pass_days: float
    worst_dd: float


def _simulate_challenge_day(
    trades: np.ndarray,
    *,
    profile: PropProfile,
    pet_on: bool,
    pet_engine: PortfolioEquityTrail,
    rng: np.random.Generator,
) -> dict[str, Any]:
    equity = STARTING_EQUITY
    phase_start = STARTING_EQUITY
    peak = equity
    day_start = equity
    state = PetRuntimeState.create(equity)
    target = phase_start * (1.0 + profile.target_profit / 100.0)
    max_dd = 0.0
    halted = False

    for day in range(90):
        state.reset_daily(equity, f"2024-01-{day + 1:02d}")
        day_start = equity
        halted = False
        n_trades = int(rng.integers(1, 5))
        for _ in range(n_trades):
            if pet_on:
                decision = pet_engine.evaluate(
                    state,
                    current_equity=equity,
                    balance=equity,
                    server_day=state.server_day,
                    challenge_progress_pct=challenge_profit_progress_pct(phase_start, equity),
                    phase_start_equity=phase_start,
                )
                if decision.disable_new_entries or decision.breached:
                    halted = True
                    break
            if halted:
                break
            r = float(rng.choice(trades))
            equity *= 1.0 + 0.01 * r
            peak = max(peak, equity)
            dd = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
            if dd >= profile.total_dd_limit:
                return {"outcome": "fail", "pass_days": None, "max_dd": max_dd}
            daily_dd = (day_start - min(day_start, equity)) / day_start * 100.0
            if daily_dd >= profile.daily_dd_limit:
                return {"outcome": "fail", "pass_days": None, "max_dd": max_dd}
            if equity >= target:
                return {"outcome": "pass", "pass_days": float(day + 1), "max_dd": max_dd}
    return {"outcome": "timeout", "pass_days": None, "max_dd": max_dd}


def run_pet_monte_carlo_validation(
    trades: pd.DataFrame,
    *,
    trials: int = 1000,
    config: dict[str, Any] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    cfg = config or load_pet_config()
    profiles = load_prop_profiles()
    profile = profiles.get("Fintokei_100K") or get_profile("Fintokei_100K")
    r_vals = pd.to_numeric(
        trades.get("R", trades.get("profit_r", pd.Series([0.5, -0.8]))),
        errors="coerce",
    ).dropna()
    if r_vals.empty:
        r_vals = pd.Series([0.5, -0.8, 1.0, -1.0])
    arr = r_vals.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    pet_engine = PortfolioEquityTrail(config=cfg)

    def _run(pet_on: bool) -> PetMonteCarloResult:
        outcomes = []
        pass_days = []
        dds = []
        for _ in range(trials):
            result = _simulate_challenge_day(
                arr,
                profile=profile,
                pet_on=pet_on,
                pet_engine=pet_engine,
                rng=rng,
            )
            outcomes.append(result["outcome"])
            if result["pass_days"] is not None:
                pass_days.append(result["pass_days"])
            dds.append(result["max_dd"])
        passes = outcomes.count("pass")
        fails = outcomes.count("fail")
        return PetMonteCarloResult(
            trials=trials,
            pass_rate=round(passes / trials * 100.0, 2),
            fail_rate=round(fails / trials * 100.0, 2),
            risk_of_ruin=round(fails / trials * 100.0, 2),
            avg_pass_days=round(float(np.mean(pass_days)) if pass_days else 0.0, 2),
            worst_dd=round(float(max(dds)) if dds else 0.0, 2),
        )

    off = _run(False)
    on = _run(True)
    return {
        "pet_off": off.__dict__,
        "pet_on": on.__dict__,
        "success": (
            on.pass_rate >= off.pass_rate
            and on.risk_of_ruin <= off.risk_of_ruin
            and on.worst_dd <= off.worst_dd
        ),
    }
