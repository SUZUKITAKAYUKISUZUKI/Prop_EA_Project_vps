"""Objective profiles — state-specific scoring for Prop Firm Objective Optimizer."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.account_state_engine.account_state_engine import AccountState


class ObjectiveKind(str, Enum):
    CHALLENGE = "challenge"
    FUNDED = "funded"
    LIVE = "live"
    RECOVERY = "recovery"


@dataclass(frozen=True)
class ObjectiveProfile:
    kind: ObjectiveKind
    label: str
    maximize: tuple[str, ...]
    minimize: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "label": self.label,
            "maximize": list(self.maximize),
            "minimize": list(self.minimize),
        }


OBJECTIVE_PROFILES: dict[ObjectiveKind, ObjectiveProfile] = {
    ObjectiveKind.CHALLENGE: ObjectiveProfile(
        kind=ObjectiveKind.CHALLENGE,
        label="FASTEST PASS",
        maximize=("PassRate", "TotalR", "SpeedToTarget"),
        minimize=("PassDays", "RiskOfRuin"),
    ),
    ObjectiveKind.FUNDED: ObjectiveProfile(
        kind=ObjectiveKind.FUNDED,
        label="STABILITY FIRST",
        maximize=("PF", "Sharpe", "RecoveryFactor", "StabilityScore"),
        minimize=("MaxDD", "UlcerIndex"),
    ),
    ObjectiveKind.LIVE: ObjectiveProfile(
        kind=ObjectiveKind.LIVE,
        label="CAPITAL GROWTH",
        maximize=("PF", "Sharpe", "TotalR", "StabilityScore"),
        minimize=("MaxDD", "RiskOfRuin"),
    ),
    ObjectiveKind.RECOVERY: ObjectiveProfile(
        kind=ObjectiveKind.RECOVERY,
        label="CAPITAL PRESERVATION",
        maximize=("CapitalPreservation",),
        minimize=("Drawdown",),
    ),
}


@dataclass(frozen=True)
class ObjectiveMetrics:
    pass_rate: float = 0.0
    speed_to_target: float = 0.0
    total_r: float = 0.0
    dd: float = 0.0
    pass_days: float = 30.0
    risk_of_ruin: float = 0.0
    pf: float = 0.0
    sharpe: float = 0.0
    stability: float = 0.0
    recovery_factor: float = 0.0
    max_dd: float = 0.0
    ulcer_index: float = 0.0
    capital_preservation: float = 0.0


def objective_for_state(state: AccountState | str) -> ObjectiveProfile:
    key = state.value if isinstance(state, AccountState) else str(state).lower()
    mapping = {
        "challenge": ObjectiveKind.CHALLENGE,
        "funded": ObjectiveKind.FUNDED,
        "live": ObjectiveKind.LIVE,
        "recovery": ObjectiveKind.RECOVERY,
    }
    return OBJECTIVE_PROFILES[mapping.get(key, ObjectiveKind.CHALLENGE)]


def recommended_objective_label(state: AccountState | str) -> str:
    return objective_for_state(state).label


def _norm_pass_rate(v: float) -> float:
    return min(1.0, max(0.0, v / 100.0 if v > 1.0 else v))


def _norm_speed(v: float) -> float:
    if v <= 0:
        return 1.0
    return min(1.0, 30.0 / v)


def _norm_r(v: float) -> float:
    return min(1.0, max(0.0, v / 100.0))


def _norm_dd(v: float) -> float:
    return min(1.0, max(0.0, abs(v) / 100.0))


def compute_objective_score(
    state: AccountState | str,
    metrics: ObjectiveMetrics,
) -> float:
    """State-specific objective score used by PFOO allocation search."""
    kind = objective_for_state(state).kind

    if kind == ObjectiveKind.CHALLENGE:
        pass_rate = _norm_pass_rate(metrics.pass_rate)
        speed = _norm_speed(metrics.pass_days)
        total_r = _norm_r(metrics.total_r)
        dd = _norm_dd(metrics.dd)
        return round(0.40 * pass_rate + 0.30 * speed + 0.20 * total_r - 0.10 * dd, 4)

    if kind == ObjectiveKind.FUNDED:
        pf = min(3.0, max(0.0, metrics.pf)) / 3.0
        sharpe = min(2.0, max(0.0, metrics.sharpe)) / 2.0
        stability = min(1.0, max(0.0, metrics.stability))
        recovery = min(3.0, max(0.0, metrics.recovery_factor)) / 3.0
        return round(0.30 * pf + 0.25 * sharpe + 0.25 * stability + 0.20 * recovery, 4)

    if kind == ObjectiveKind.LIVE:
        pf = min(3.0, max(0.0, metrics.pf)) / 3.0
        sharpe = min(2.0, max(0.0, metrics.sharpe)) / 2.0
        total_r = _norm_r(metrics.total_r)
        stability = min(1.0, max(0.0, metrics.stability))
        dd_penalty = _norm_dd(metrics.max_dd) * 0.15
        return round(0.35 * pf + 0.25 * sharpe + 0.25 * total_r + 0.15 * stability - dd_penalty, 4)

    cap = min(1.0, max(0.0, metrics.capital_preservation))
    dd_penalty = _norm_dd(metrics.dd)
    return round(cap - dd_penalty, 4)


def metrics_from_pfoo_context(
    *,
    pass_probability: float,
    expected_pass_days: float,
    total_dd_used_pct: float,
    total_dd_limit: float,
    utility: float = 0.0,
    pf: float = 0.0,
    sharpe: float = 0.0,
    total_r: float = 0.0,
) -> ObjectiveMetrics:
    headroom = max(0.0, total_dd_limit - total_dd_used_pct)
    cap = headroom / max(total_dd_limit, 1e-9)
    stability = cap
    recovery_factor = pf * cap if pf else cap
    return ObjectiveMetrics(
        pass_rate=pass_probability,
        speed_to_target=_norm_speed(expected_pass_days),
        total_r=total_r,
        dd=total_dd_used_pct,
        pass_days=expected_pass_days,
        risk_of_ruin=max(0.0, 100.0 - pass_probability) / 100.0,
        pf=pf,
        sharpe=sharpe,
        stability=stability,
        recovery_factor=recovery_factor,
        max_dd=total_dd_used_pct,
        ulcer_index=total_dd_used_pct / max(total_dd_limit, 1e-9),
        capital_preservation=cap,
    )
