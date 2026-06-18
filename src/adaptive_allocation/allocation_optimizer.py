"""Strategy quality scoring and weight adjustment for PAAE."""
from __future__ import annotations

from typing import Any

from src.adaptive_allocation.allocation_constraints import AllocationConstraints, enforce_constraints, normalize_weights
from src.adaptive_allocation.allocation_policy import base_weights_for_state
from src.database.profile_migrations import DASHBOARD_STRATEGY_CODES


class AllocationOptimizer:
    """Compute quality scores and propose weight adjustments."""

    def build_strategy_inputs(self, prae_v2: dict[str, Any]) -> dict[str, dict[str, float]]:
        profit_map: dict[str, float] = {}
        for row in prae_v2.get("weight_adjusted_contribution") or []:
            profit_map[str(row["strategy"])] = float(row.get("contribution_pct") or 0.0)

        dd_map = dict((prae_v2.get("dd_attribution") or {}).get("strategy_contribution") or {})
        dd_map = {str(k): float(v) for k, v in dd_map.items()}

        recovery_map: dict[str, float] = {code: 0.0 for code in DASHBOARD_STRATEGY_CODES}
        for event in prae_v2.get("recovery_analysis") or []:
            for strat, pct in (event.get("strategy_contribution") or {}).items():
                recovery_map[str(strat)] = recovery_map.get(str(strat), 0.0) + float(pct)

        risk_map: dict[str, float] = {}
        for row in prae_v2.get("strategy_risk") or []:
            risk_map[str(row["strategy"])] = float(row.get("risk_score") or 0.0)

        inputs: dict[str, dict[str, float]] = {}
        for code in DASHBOARD_STRATEGY_CODES:
            inputs[code] = {
                "profit_contribution": profit_map.get(code, 0.0),
                "dd_contribution": dd_map.get(code, 0.0),
                "recovery_contribution": recovery_map.get(code, 0.0),
                "risk_score": risk_map.get(code, 0.0),
            }
        return inputs

    def compute_quality_scores(
        self,
        strategy_inputs: dict[str, dict[str, float]],
        portfolio_fit_scores: dict[str, float] | None = None,
    ) -> dict[str, float]:
        raw: dict[str, float] = {}
        for code, metrics in strategy_inputs.items():
            raw[code] = (
                float(metrics.get("profit_contribution", 0.0))
                - float(metrics.get("dd_contribution", 0.0))
                - float(metrics.get("recovery_contribution", 0.0))
            )
        if not raw:
            base = {code: 50.0 for code in DASHBOARD_STRATEGY_CODES}
        else:
            min_v = min(raw.values())
            max_v = max(raw.values())
            span = max(max_v - min_v, 1e-9)
            base = {code: round((val - min_v) / span * 100.0, 1) for code, val in raw.items()}

        if not portfolio_fit_scores:
            return base

        combined: dict[str, float] = {}
        for code in DASHBOARD_STRATEGY_CODES:
            existing = float(base.get(code, 50.0))
            fit = float(portfolio_fit_scores.get(code, 50.0))
            combined[code] = round(existing * 0.75 + fit * 0.25, 1)
        return combined

    def adjust_weights(
        self,
        current: dict[str, float],
        quality_scores: dict[str, float],
        *,
        account_state: str,
        constraints: AllocationConstraints | None = None,
        lifecycle_stages: dict[str, str] | None = None,
        core_min_weights: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], set[str], dict[str, str]]:
        c = constraints or AllocationConstraints()
        base = base_weights_for_state(account_state)
        start = current if sum(current.values()) > 0 else base

        proposed = dict(start)
        reasons: dict[str, str] = {}
        disabled: set[str] = set()
        core_min = core_min_weights or {}

        for code, quality in quality_scores.items():
            stage = (lifecycle_stages or {}).get(code, "PRODUCTION").upper()
            if stage in {"INCUBATION", "RETIRED"}:
                proposed[code] = 0.0
                disabled.add(code)
                reasons[code] = f"lifecycle_{stage.lower()}"
                continue
            if stage in {"CANDIDATE", "RECOVERY"}:
                proposed[code] = 0.05
                reasons[code] = f"lifecycle_{stage.lower()}_fixed_5pct"
                continue

            delta = 0.0
            reason = "maintain"
            if quality > 80:
                delta = 0.10
                reason = "quality_gt_80_increase"
            elif quality >= 60:
                delta = 0.0
                reason = "quality_60_80_maintain"
            elif quality >= 40:
                delta = -0.10
                reason = "quality_lt_60_decrease"
            elif quality >= 20:
                delta = -0.20
                reason = "quality_lt_40_decrease"
            else:
                delta = -proposed.get(code, 0.0)
                disabled.add(code)
                reason = "quality_lt_20_disable_candidate"
            proposed[code] = round(max(0.0, proposed.get(code, base.get(code, 0.0)) + delta), 4)
            if stage == "CORE":
                proposed[code] = max(proposed[code], core_min.get(code, 0.10))
                reason = "core_paae_min_10pct"
            reasons[code] = reason

        final = enforce_constraints(start, proposed, c, disabled=disabled)
        final = normalize_weights({k: v for k, v in final.items() if k not in disabled or v > 0})
        if disabled:
            final = normalize_weights({k: v for k, v in final.items() if k not in disabled})
        return final, disabled, reasons
