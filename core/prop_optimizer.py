"""Phase 5 — Prop Firm Objective Optimizer (PFOO) orchestrator."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from audit.risk_manager import STARTING_EQUITY, challenge_profit_progress_pct
from core.endgame_mode import EndgameDecision, evaluate_endgame_mode
from core.pass_probability import (
    AccountSnapshot,
    ChallengeState,
    PassProbabilityResult,
    estimate_expected_pass_days,
    estimate_pass_probability,
)
from core.progress_risk import progress_risk_multiplier
from core.prop_profiles import PropProfile, get_profile, load_pfoo_config
from core.recovery_mode import RecoveryDecision, evaluate_recovery_mode
from core.risk_budget_optimizer import RiskBudgetResult, optimize_risk_budget
from core.utility_engine import ObjectiveMode, UtilityResult, compute_utility
from prae.config import load_config as load_prae_config
from prae.loaders import apply_strategy_labels, discover_strategies, load_portfolio_trades
from prae.metrics import apply_allocation_weights

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OperationalStatus = Literal["NORMAL", "RECOVERY", "ENDGAME"]


@dataclass
class MonteCarloValidation:
    trials: int
    pass_rate: float
    fail_rate: float
    avg_pass_days: float
    worst_dd: float
    dd_p95: float
    expected_utility: float


@dataclass
class PFOOResult:
    profile: PropProfile
    mode: ObjectiveMode
    operational_status: OperationalStatus
    account: AccountSnapshot
    challenge: ChallengeState
    progress_risk_multiplier: float
    recovery: RecoveryDecision
    endgame: EndgameDecision
    pass_probability: PassProbabilityResult
    expected_pass_days: float
    utility: UtilityResult
    risk_budget: RiskBudgetResult
    monte_carlo: dict[int, MonteCarloValidation]
    strategies: tuple[str, ...]
    recommended_weights: dict[str, float]
    prae_context: dict[str, Any]
    report_path: Path
    artifact_dir: Path


def _load_prae_context(artifact_dir: Path) -> dict[str, Any]:
    from src.repositories.base import normalize_source_path
    from src.repositories.portfolio_repository import PortfolioRepository

    ctx: dict[str, Any] = {}
    portfolio_repo = PortfolioRepository()

    risk_rel = normalize_source_path(artifact_dir / "risk_contribution.csv")
    risk_rows = portfolio_repo.get_risk_attribution(source_path=risk_rel)
    if not risk_rows:
        risk_rows = portfolio_repo.get_risk_attribution(source_path="backtest_results/prae_v1/risk_contribution.csv")
    if risk_rows:
        ctx["risk_contribution"] = {
            row["strategy"]: row.get("contribution_dd")
            for row in risk_rows
            if row.get("contribution_dd") is not None
        }

    marginal_rel = normalize_source_path(artifact_dir / "marginal_contribution.csv")
    marginal_rows = portfolio_repo.get_risk_attribution(source_path=marginal_rel)
    if marginal_rows:
        ctx["marginal_ranking"] = marginal_rows
        ctx["weakest_strategies"] = tuple(row["strategy"] for row in marginal_rows[:3])
    alloc_path = artifact_dir / "recommended_allocation.json"
    if alloc_path.exists():
        ctx["prae_allocation"] = json.loads(alloc_path.read_text(encoding="utf-8"))
    from src.services.portfolio_service import PortfolioService

    portfolio_svc = PortfolioService(portfolio_repo=portfolio_repo)
    ctx["correlation"] = portfolio_svc.correlation_matrix(
        source_path="backtest_results/main_abcd_3y.csv"
    )
    return ctx


def _build_challenge_state(
    account: AccountSnapshot,
    profile: PropProfile,
    *,
    days_elapsed: int = 0,
) -> ChallengeState:
    progress = challenge_profit_progress_pct(account.phase_start_equity, account.equity)
    peak = max(account.peak_equity, account.equity)
    total_dd_used = (peak - account.equity) / peak * 100.0 if peak > 0 else 0.0
    daily_dd_used = max(0.0, profile.daily_dd_limit * 0.1)
    return ChallengeState(
        days_elapsed=days_elapsed,
        profit_progress_percent=round(progress, 2),
        daily_dd_used_percent=round(daily_dd_used, 2),
        total_dd_used_percent=round(total_dd_used, 2),
    )


def _resolve_account_state(
    account: AccountSnapshot,
    profile: PropProfile,
    challenge: ChallengeState,
) -> str:
    from src.account_state_engine.account_state_engine import AccountStateEngine, AccountStateInput

    starting = float(getattr(account, "phase_start_equity", None) or STARTING_EQUITY)
    target_balance = starting * (1.0 + profile.target_profit / 100.0)
    challenge_passed = challenge.profit_progress_percent >= profile.target_profit
    profile_key = str(getattr(profile, "profile_key", "") or "challenge")
    account_type = "live" if profile_key == "live" else "prop"

    engine = AccountStateEngine()
    result = engine.evaluate(
        AccountStateInput(
            current_balance=float(account.equity),
            initial_balance=starting,
            target_balance=target_balance,
            max_total_dd=float(profile.total_dd_limit),
            current_dd=float(challenge.total_dd_used_percent),
            account_type=account_type,
            challenge_passed=challenge_passed,
        )
    )
    return result.state.value


def _effective_global_risk_mult(
    progress_mult: float,
    recovery: RecoveryDecision,
    endgame: EndgameDecision,
) -> float:
    mult = progress_mult
    if recovery.active:
        mult *= recovery.risk_multiplier
    if endgame.active:
        mult *= endgame.risk_multiplier
    return round(mult, 4)


def _run_mc_validation(
    trades: pd.DataFrame,
    *,
    profile: PropProfile,
    account: AccountSnapshot,
    challenge: ChallengeState,
    weights: dict[str, float],
    mode: ObjectiveMode,
    config: dict[str, Any],
    trials: int,
    global_risk_mult: float,
    fast: bool = False,
    horizon_trades: int | None = None,
) -> MonteCarloValidation:
    weighted = apply_allocation_weights(trades, weights)
    active = weighted[weighted["allocation_weight"] > 0]
    pass_res = estimate_pass_probability(
        active,
        profile=profile,
        account=account,
        challenge=challenge,
        trials=trials,
        global_risk_mult=global_risk_mult,
        horizon_trades=horizon_trades,
        fast=fast,
    )
    days_res = estimate_expected_pass_days(
        active,
        profile=profile,
        account=account,
        challenge=challenge,
        trials=max(30, trials // 10),
        global_risk_mult=global_risk_mult,
    )
    util = compute_utility(
        pass_probability=pass_res.pass_probability,
        expected_pass_days=days_res.expected_pass_days,
        total_dd_used_pct=challenge.total_dd_used_percent,
        total_dd_limit=profile.total_dd_limit,
        mode=mode,
        config=config,
    )
    dd_proxy = challenge.total_dd_used_percent + pass_res.fail_probability * 0.05
    return MonteCarloValidation(
        trials=trials,
        pass_rate=pass_res.pass_probability,
        fail_rate=pass_res.fail_probability,
        avg_pass_days=days_res.expected_pass_days,
        worst_dd=round(dd_proxy, 2),
        dd_p95=round(min(profile.total_dd_limit, dd_proxy * 1.2), 2),
        expected_utility=util.utility,
    )


def _operational_status(recovery: RecoveryDecision, endgame: EndgameDecision) -> OperationalStatus:
    if endgame.active:
        return "ENDGAME"
    if recovery.active:
        return "RECOVERY"
    return "NORMAL"


def _write_report(path: Path, result: PFOOResult) -> None:
    lines = [
        "# PFOO — Prop Firm Objective Optimizer Report",
        "",
        f"Profile: **{result.profile.name}** | Mode: **{result.mode}** | Status: **{result.operational_status}**",
        "",
        "## Challenge State",
        "",
        f"- Progress: **{result.challenge.profit_progress_percent}%** / {result.profile.target_profit}%",
        f"- Total DD used: **{result.challenge.total_dd_used_percent}%** / {result.profile.total_dd_limit}%",
        f"- Days elapsed: **{result.challenge.days_elapsed}**",
        "",
        "## Pass Probability",
        "",
        f"- **{result.pass_probability.pass_probability}%** (fail {result.pass_probability.fail_probability}%)",
        f"- Expected pass days: **{result.expected_pass_days}**",
        f"- Utility ({result.mode}): **{result.utility.utility}**",
        "",
        "## Progress-Aware Risk",
        "",
        f"- Risk multiplier: **{result.progress_risk_multiplier}**",
        f"- Global effective multiplier: **{_effective_global_risk_mult(result.progress_risk_multiplier, result.recovery, result.endgame)}**",
        "",
        "## Recommended Risk Budget",
        "",
        "```json",
        json.dumps(result.recommended_weights, indent=2),
        "```",
        "",
        "## Monte Carlo Validation",
        "",
        "| Trials | Pass Rate | Fail Rate | Avg Pass Days | Worst DD | 95% DD | Utility |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for trials, mc in sorted(result.monte_carlo.items()):
        lines.append(
            f"| {trials} | {mc.pass_rate:.2f} | {mc.fail_rate:.2f} | {mc.avg_pass_days:.1f} | "
            f"{mc.worst_dd:.2f} | {mc.dd_p95:.2f} | {mc.expected_utility:.4f} |"
        )
    lines.extend(["", "## Top Risk Budget Candidates", ""])
    if not result.risk_budget.top_candidates.empty:
        lines.append(result.risk_budget.top_candidates.to_markdown(index=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class PropOptimizer:
    """Final decision layer — prop-firm objective optimization."""

    def __init__(
        self,
        *,
        config: dict[str, Any] | None = None,
        profile_name: str | None = None,
        mode: ObjectiveMode = "BALANCED",
    ) -> None:
        self.config = config or load_pfoo_config()
        resolved_name = profile_name
        if resolved_name is None:
            try:
                from src.services.profile_service import ProfileService

                svc = ProfileService()
                try:
                    resolved_name = svc.load_active_profile().profile_id
                finally:
                    svc.close()
            except Exception:
                resolved_name = self.config.get("default_profile", "Fintokei_100K")
        self.profile = get_profile(resolved_name)
        self.mode = mode.upper()  # type: ignore[assignment]

    def evaluate(
        self,
        trades: pd.DataFrame,
        *,
        account: AccountSnapshot | None = None,
        challenge: ChallengeState | None = None,
        prae_context: dict[str, Any] | None = None,
        full_mc: bool = False,
    ) -> PFOOResult:
        label_map = dict(self.config.get("strategy_labels") or {})
        prae_cfg = load_prae_config()
        for k, v in (prae_cfg.get("strategy_labels") or {}).items():
            label_map.setdefault(k, v)
        trades = apply_strategy_labels(trades, label_map)
        strategies = discover_strategies(trades)

        acct = account or AccountSnapshot(
            equity=STARTING_EQUITY,
            balance=STARTING_EQUITY,
            peak_equity=STARTING_EQUITY,
            phase_start_equity=STARTING_EQUITY,
        )
        ch = challenge or _build_challenge_state(acct, self.profile)
        account_state = _resolve_account_state(acct, self.profile, ch)

        prae_ctx = prae_context or _load_prae_context(
            PROJECT_ROOT / self.config.get("prae_artifact_dir", "backtest_results/prae_v1")
        )
        weakest = tuple(prae_ctx.get("weakest_strategies") or ())

        progress_mult = progress_risk_multiplier(
            ch.profit_progress_percent,
            self.config.get("progress_risk_bands"),
        )
        recovery = evaluate_recovery_mode(
            ch,
            total_dd_limit=self.profile.total_dd_limit,
            weakest_strategies=weakest,
            config=self.config,
        )
        endgame = evaluate_endgame_mode(
            ch,
            target_profit_pct=self.profile.target_profit,
            config=self.config,
        )
        global_mult = _effective_global_risk_mult(progress_mult, recovery, endgame)

        base_weights = prae_ctx.get("prae_allocation")
        risk_budget = optimize_risk_budget(
            trades,
            strategies,
            profile=self.profile,
            account=acct,
            challenge=ch,
            mode=self.mode,
            config=self.config,
            prae_context=prae_ctx,
            base_weights=base_weights,
            account_state=account_state,
        )

        weighted = apply_allocation_weights(trades, risk_budget.weights)
        active = weighted[weighted["allocation_weight"] > 0]
        pass_res = estimate_pass_probability(
            active,
            profile=self.profile,
            account=acct,
            challenge=ch,
            trials=min(200, int((self.config.get("monte_carlo_trials") or [200])[0])),
            global_risk_mult=global_mult,
        )
        days_res = estimate_expected_pass_days(
            active,
            profile=self.profile,
            account=acct,
            challenge=ch,
            global_risk_mult=global_mult,
        )
        util = compute_utility(
            pass_probability=pass_res.pass_probability,
            expected_pass_days=days_res.expected_pass_days,
            total_dd_used_pct=ch.total_dd_used_percent,
            total_dd_limit=self.profile.total_dd_limit,
            mode=self.mode,
            config=self.config,
        )

        mc_results: dict[int, MonteCarloValidation] = {}
        for trial_count in self.config.get("monte_carlo_trials") or [1000]:
            mc_results[int(trial_count)] = _run_mc_validation(
                trades,
                profile=self.profile,
                account=acct,
                challenge=ch,
                weights=risk_budget.weights,
                mode=self.mode,
                config=self.config,
                trials=int(trial_count),
                global_risk_mult=global_mult,
                fast=not full_mc,
                horizon_trades=300 if not full_mc else None,
            )

        return PFOOResult(
            profile=self.profile,
            mode=self.mode,
            operational_status=_operational_status(recovery, endgame),
            account=acct,
            challenge=ch,
            progress_risk_multiplier=progress_mult,
            recovery=recovery,
            endgame=endgame,
            pass_probability=pass_res,
            expected_pass_days=days_res.expected_pass_days,
            utility=util,
            risk_budget=risk_budget,
            monte_carlo=mc_results,
            strategies=strategies,
            recommended_weights=risk_budget.weights,
            prae_context=prae_ctx,
            report_path=PROJECT_ROOT / "reports" / "pfoo_report.md",
            artifact_dir=PROJECT_ROOT / "backtest_results" / "pfoo_v1",
        )


def run_pfoo(
    *,
    input_paths: list[Path] | None = None,
    profile_name: str | None = None,
    mode: ObjectiveMode = "BALANCED",
    account: AccountSnapshot | None = None,
    challenge: ChallengeState | None = None,
    config_path: Path | None = None,
    reports_dir: Path | None = None,
    artifact_dir: Path | None = None,
    full_mc: bool = False,
) -> PFOOResult:
    config = load_pfoo_config(config_path)
    if full_mc:
        config = {**config, "monte_carlo_trials": [1000, 5000, 10000]}
    if input_paths is None:
        input_paths = [PROJECT_ROOT / p for p in config.get("default_inputs", [])]

    trades = load_portfolio_trades(input_paths)
    print(f"PFOO: {len(trades):,} trades loaded - optimizing...")
    optimizer = PropOptimizer(
        config=config,
        profile_name=profile_name,
        mode=mode,
    )
    result = optimizer.evaluate(trades, account=account, challenge=challenge, full_mc=full_mc)
    print("PFOO: optimization complete - writing artifacts...")

    artifact_dir = artifact_dir or result.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = reports_dir or PROJECT_ROOT / "reports"
    report_path = reports_dir / "pfoo_report.md"

    (artifact_dir / "recommended_risk_budget.json").write_text(
        json.dumps(result.recommended_weights, indent=2),
        encoding="utf-8",
    )

    analytics = None
    try:
        from src.services.analytics_write_service import AnalyticsWriteService

        analytics = AnalyticsWriteService()
        analytics.save_pfoo_artifacts(artifact_dir=str(artifact_dir), result=result)
        print(f"PFOO artifacts saved to SQLite (run registered under {artifact_dir})")
    finally:
        if analytics is not None:
            analytics.close()

    if os.environ.get("ANALYTICS_EXPORT_CSV", "0").strip().lower() in {"1", "true", "yes"}:
        result.risk_budget.top_candidates.to_csv(artifact_dir / "risk_budget_top10.csv", index=False)
        mc_summary = pd.DataFrame(
            [
                {
                    "trials": t,
                    "pass_rate": mc.pass_rate,
                    "fail_rate": mc.fail_rate,
                    "avg_pass_days": mc.avg_pass_days,
                    "worst_dd": mc.worst_dd,
                    "dd_p95": mc.dd_p95,
                    "expected_utility": mc.expected_utility,
                }
                for t, mc in sorted(result.monte_carlo.items())
            ]
        )
        mc_summary.to_csv(artifact_dir / "monte_carlo_validation.csv", index=False)

    result.report_path = report_path
    result.artifact_dir = artifact_dir
    _write_report(report_path, result)
    return result
