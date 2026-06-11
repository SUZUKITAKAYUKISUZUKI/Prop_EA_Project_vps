"""
optimize_lgr_prop.py — LGR Prop-Focused Optuna (EV sizing + defense only)

ARCHIVED (2026-06-10): LGR はプロップ向きでないため archive/lgr/ へ移動。
参照用。`python -m archive.lgr.optimize_lgr_prop` または archive から実行。

エントリー条件は固定。WFT OOS の DD 超過率を最優先し、
score = PF * Sharpe * (1 - dd_exceed_rate) を最大化する。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from optuna.trial import TrialState

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: no cover

from backtest_runner import (
    DEFAULT_EUR_FILE,
    DEFAULT_EUR_M15_FILE,
    DEFAULT_GBP_FILE,
    DEFAULT_GBP_M15_FILE,
    compute_backtest_metrics,
)
from archive.lgr.lgr_bayes_gate import profit_factor
from archive.lgr.lgr_prop_controls import (
    DISQUALIFY_SCORE,
    LgrPropTrialParams,
    clear_lgr_prop_trial_env,
    compute_prop_score,
    configure_lgr_prop_baseline_env,
    lgr_prop_trial_env,
)
from optuna_runtime import enable_optuna_runtime
from strategies.archive.lgr_scan_hot import configure_lgr_production_detector
from strategies.archive.liquidity_grab_reversal import LiquidityGrabReversalStrategy
from walkforward_runner import (
    WalkForwardAggregator,
    WalkForwardAnalyzer,
    WalkForwardRunner,
    compute_sharpe_from_records,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LGR_RESULTS = PROJECT_ROOT / "backtest_results" / "archive" / "lgr"
DEFAULT_TRIALS_CSV = LGR_RESULTS / "lgr_prop_optuna_trials.csv"
DEFAULT_TOP20_MD = LGR_RESULTS / "LGR_PROP_OPTUNA_TOP20.md"
DEFAULT_BEST_JSON = LGR_RESULTS / "lgr_prop_optuna_best.json"
DEFAULT_WFT_REPORT = LGR_RESULTS / "wft_lgr_prop_optuna"

logger = logging.getLogger("optimize_lgr_prop")


def suggest_lgr_prop_params(trial: optuna.Trial) -> LgrPropTrialParams:
    top5_risk = trial.suggest_float("top5_risk", 1.00, 2.00)
    top20_risk = trial.suggest_float("top20_risk", 1.00, min(1.50, top5_risk))
    mid_risk = trial.suggest_float("mid_risk", 0.50, min(1.25, top20_risk))
    bottom_risk = trial.suggest_float("bottom_risk", 0.10, min(0.75, mid_risk))
    top_pct = trial.suggest_int("top_pct", 3, 15)
    top20_pct = trial.suggest_int("top20_pct", top_pct + 1, 40)
    daily_stop_r = trial.suggest_float("daily_stop_r", -5.0, -2.0)
    max_positions = trial.suggest_int("max_positions", 1, 5)
    session_open_min = trial.suggest_int("session_open_min", 0, 60, step=15)
    session_open_max = trial.suggest_int("session_open_max", 30, 120, step=15)
    if session_open_max <= session_open_min:
        session_open_max = min(120, session_open_min + 15)
    params = LgrPropTrialParams(
        top5_risk=top5_risk,
        top20_risk=top20_risk,
        mid_risk=mid_risk,
        bottom_risk=bottom_risk,
        top_pct=top_pct,
        top20_pct=top20_pct,
        daily_stop_r=daily_stop_r,
        max_positions=max_positions,
        session_open_min=session_open_min,
        session_open_max=session_open_max,
    )
    if top5_risk < top20_risk:
        top20_risk = top5_risk
    if top20_risk < mid_risk:
        mid_risk = top20_risk
    if mid_risk < bottom_risk:
        bottom_risk = mid_risk
    return LgrPropTrialParams(
        top5_risk=top5_risk,
        top20_risk=top20_risk,
        mid_risk=mid_risk,
        bottom_risk=bottom_risk,
        top_pct=top_pct,
        top20_pct=top20_pct,
        daily_stop_r=daily_stop_r,
        max_positions=max_positions,
        session_open_min=session_open_min,
        session_open_max=session_open_max,
    )


def run_lgr_prop_wft(
    params: LgrPropTrialParams,
    *,
    data_source: dict[str, Path],
    is_months: int,
    oos_months: int,
    step_months: int,
    report_dir: Path,
    wft_fresh: bool,
    max_windows: int | None,
) -> dict[str, Any]:
    configure_lgr_prop_baseline_env()
    configure_lgr_production_detector()
    from archive.lgr.lgr_ev_position_sizing import initialize_lgr_ev_sizing

    initialize_lgr_ev_sizing(retrain=False)

    with lgr_prop_trial_env(params):
        runner = WalkForwardRunner(
            LiquidityGrabReversalStrategy,
            data_source=data_source,
            strategy_mode="lgr",
            config={
                "is_months": is_months,
                "oos_months": oos_months,
                "step_months": step_months,
            },
            report_dir=report_dir,
            use_llm=False,
            mock_llm=True,
            mock_mode="strategy_edge",
            resume=False,
            fresh=wft_fresh,
            lgr_ev_sizing=True,
            lgr_defense_bt=True,
            lgr_features_output=LGR_RESULTS / "logs" / "lgr_features.csv",
        )
        results = runner.run(
            is_months=is_months,
            oos_months=oos_months,
            step_months=step_months,
            max_windows=max_windows,
        )

    windows = results.windows

    stability = WalkForwardAnalyzer.compute_stability(windows)
    dd_exceed_rate = float(stability["oos_dd_exceed_rate_pct"]) / 100.0
    combined = WalkForwardAggregator.chain_oos_records(windows)
    metrics = compute_backtest_metrics(combined)
    executed = [r for r in combined if r.get("trade_result") in ("WIN", "LOSS")]
    pf = 0.0
    if executed:
        pf = float(profit_factor(pd.Series([float(r["profit_r"]) for r in executed])))
    sharpe = compute_sharpe_from_records(combined)
    score = compute_prop_score(pf, sharpe, dd_exceed_rate)

    return {
        "score": score,
        "pf": pf,
        "sharpe": sharpe,
        "total_r": metrics.total_profit_r,
        "max_dd_pct": metrics.max_total_dd_pct,
        "max_daily_dd_pct": metrics.max_daily_dd_pct,
        "dd_exceed_rate": dd_exceed_rate,
        "dd_exceed_rate_pct": stability["oos_dd_exceed_rate_pct"],
        "mean_oos_profit_r": stability["mean_oos_profit_r"],
        "mean_oos_max_dd_pct": stability["mean_oos_max_dd_pct"],
        "executed_trades": metrics.executed_trades,
        "windows": len(windows),
    }


def record_trial_attrs(trial: optuna.Trial, params: LgrPropTrialParams, stats: dict[str, Any]) -> None:
    for key, value in params.as_dict().items():
        trial.set_user_attr(key, value)
    for key in (
        "pf",
        "sharpe",
        "total_r",
        "max_dd_pct",
        "max_daily_dd_pct",
        "dd_exceed_rate",
        "dd_exceed_rate_pct",
        "mean_oos_profit_r",
        "mean_oos_max_dd_pct",
        "executed_trades",
        "windows",
    ):
        trial.set_user_attr(key, stats.get(key))


def append_trial_csv(path: Path, trial: optuna.trial.FrozenTrial, *, score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial",
        "state",
        "score",
        "pf",
        "sharpe",
        "total_r",
        "max_dd_pct",
        "dd_exceed_rate_pct",
        "top5_risk",
        "top20_risk",
        "mid_risk",
        "bottom_risk",
        "top_pct",
        "top20_pct",
        "daily_stop_r",
        "max_positions",
        "session_open_min",
        "session_open_max",
        "elapsed_sec",
    ]
    state_name = "RUNNING"
    if hasattr(trial, "state") and trial.state is not None:
        state_name = trial.state.name
    row = {
        "trial": trial.number,
        "state": state_name,
        "score": score,
        "pf": trial.user_attrs.get("pf"),
        "sharpe": trial.user_attrs.get("sharpe"),
        "total_r": trial.user_attrs.get("total_r"),
        "max_dd_pct": trial.user_attrs.get("max_dd_pct"),
        "dd_exceed_rate_pct": trial.user_attrs.get("dd_exceed_rate_pct"),
        "top5_risk": trial.params.get("top5_risk"),
        "top20_risk": trial.params.get("top20_risk"),
        "mid_risk": trial.params.get("mid_risk"),
        "bottom_risk": trial.params.get("bottom_risk"),
        "top_pct": trial.params.get("top_pct"),
        "top20_pct": trial.params.get("top20_pct"),
        "daily_stop_r": trial.params.get("daily_stop_r"),
        "max_positions": trial.params.get("max_positions"),
        "session_open_min": trial.params.get("session_open_min"),
        "session_open_max": trial.params.get("session_open_max"),
        "elapsed_sec": trial.user_attrs.get("elapsed_sec"),
    }
    write_header = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _valid_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [
        t
        for t in study.trials
        if t.state == TrialState.COMPLETE
        and t.value is not None
        and float(t.value) > DISQUALIFY_SCORE + 1
    ]


def _trial_metric(trial: optuna.trial.FrozenTrial, key: str, default: float = 0.0) -> float:
    value = trial.user_attrs.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_top20_markdown(study: optuna.Study) -> str:
    valid = _valid_trials(study)
    if not valid:
        return "# LGR Prop Optuna - no valid trials\n"

    top = sorted(valid, key=lambda t: float(t.value or DISQUALIFY_SCORE), reverse=True)[:20]
    lines = [
        "# LGR Prop Optuna - Top 20 Trials",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "| Rank | Trial | Score | PF | Sharpe | TotalR | MaxDD% | DD exceed% | top5 | top20 | mid | bottom | top% | top20% | stopR | maxPos | sessMin | sessMax |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, trial in enumerate(top, start=1):
        lines.append(
            "| {rank} | {n} | {score:.4f} | {pf:.3f} | {sh:.3f} | {tr:+.1f} | {dd:.2f} | {dex:.1f} | "
            "{t5:.2f} | {t20:.2f} | {mid:.2f} | {bot:.2f} | {tp} | {t2p} | {stop:.1f} | {mp} | {smin} | {smax} |".format(
                rank=rank,
                n=trial.number,
                score=float(trial.value or 0.0),
                pf=_trial_metric(trial, "pf"),
                sh=_trial_metric(trial, "sharpe"),
                tr=_trial_metric(trial, "total_r"),
                dd=_trial_metric(trial, "max_dd_pct"),
                dex=_trial_metric(trial, "dd_exceed_rate_pct"),
                t5=trial.params.get("top5_risk", 0),
                t20=trial.params.get("top20_risk", 0),
                mid=trial.params.get("mid_risk", 0),
                bot=trial.params.get("bottom_risk", 0),
                tp=trial.params.get("top_pct", 0),
                t2p=trial.params.get("top20_pct", 0),
                stop=trial.params.get("daily_stop_r", 0),
                mp=trial.params.get("max_positions", 0),
                smin=trial.params.get("session_open_min", 0),
                smax=trial.params.get("session_open_max", 0),
            )
        )

    best_pf = max(valid, key=lambda t: _trial_metric(t, "pf"))
    best_sh = max(valid, key=lambda t: _trial_metric(t, "sharpe"))
    best_dd = min(valid, key=lambda t: _trial_metric(t, "max_dd_pct", default=999.0))
    best_prop = max(valid, key=lambda t: float(t.value or DISQUALIFY_SCORE))

    def _env_block(title: str, trial: optuna.trial.FrozenTrial) -> list[str]:
        params = LgrPropTrialParams(
            top5_risk=float(trial.params["top5_risk"]),
            top20_risk=float(trial.params["top20_risk"]),
            mid_risk=float(trial.params["mid_risk"]),
            bottom_risk=float(trial.params["bottom_risk"]),
            top_pct=int(trial.params["top_pct"]),
            top20_pct=int(trial.params["top20_pct"]),
            daily_stop_r=float(trial.params["daily_stop_r"]),
            max_positions=int(trial.params["max_positions"]),
            session_open_min=int(trial.params["session_open_min"]),
            session_open_max=int(trial.params["session_open_max"]),
        )
        return [
            f"## {title}",
            "",
            f"- Trial #{trial.number} | score={float(trial.value or 0):.4f}",
            f"- PF={_trial_metric(trial, 'pf'):.3f} | Sharpe={_trial_metric(trial, 'sharpe'):.3f} | "
            f"TotalR={_trial_metric(trial, 'total_r'):+.1f} | MaxDD={_trial_metric(trial, 'max_dd_pct'):.2f}% | "
            f"DD exceed={_trial_metric(trial, 'dd_exceed_rate_pct'):.1f}%",
            "",
            "```powershell",
            *[f'$env:{key}="{value}"' for key, value in params.as_env().items()],
            "```",
            "",
        ]

    lines.extend(_env_block("Best Prop Score", best_prop))
    lines.extend(_env_block("Best PF", best_pf))
    lines.extend(_env_block("Best Sharpe", best_sh))
    lines.extend(_env_block("Lowest DD", best_dd))
    lines.extend(
        [
            "## Recommended Production Setting",
            "",
            "Prop score balance winner (PF x Sharpe x (1 - dd_exceed_rate)).",
            "",
        ]
    )
    lines.extend(_env_block("Recommended", best_prop)[2:])
    return "\n".join(lines)


def build_objective(args: argparse.Namespace) -> Any:
    data_source = {
        "m15_gbp": args.gbp_m15,
        "m15_eur": args.eur_m15,
        "h1_gbp": args.gbp_h1,
        "h1_eur": args.eur_h1,
    }

    def objective(trial: optuna.Trial) -> float:
        params = suggest_lgr_prop_params(trial)
        reject = params.validate()
        if reject:
            trial.set_user_attr("disqualified", True)
            trial.set_user_attr("disqualify_reason", reject)
            return DISQUALIFY_SCORE

        t0 = time.perf_counter()
        report_dir = args.wft_report_dir / f"trial_{trial.number:04d}"
        try:
            stats = run_lgr_prop_wft(
                params,
                data_source=data_source,
                is_months=args.is_months,
                oos_months=args.oos_months,
                step_months=args.step_months,
                report_dir=report_dir,
                wft_fresh=True,
                max_windows=args.max_windows,
            )
        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            logger.exception("Trial %d failed", trial.number)
            return DISQUALIFY_SCORE

        elapsed = time.perf_counter() - t0
        trial.set_user_attr("elapsed_sec", round(elapsed, 2))
        record_trial_attrs(trial, params, stats)
        trial.set_user_attr("disqualified", stats["dd_exceed_rate"] > 0.10)

        logger.info(
            "Trial %d | score=%.4f pf=%.3f sharpe=%.3f dd_exceed=%.1f%% totalR=%+.1f (%.0fs)",
            trial.number,
            stats["score"],
            stats["pf"],
            stats["sharpe"],
            stats["dd_exceed_rate_pct"],
            stats["total_r"],
            elapsed,
        )
        append_trial_csv(args.trials_csv, trial, score=float(stats["score"]))
        return float(stats["score"])

    return objective


class LgrPropOptunaProgress:
    def __init__(self, n_trials: int, enabled: bool = True) -> None:
        self.n_trials = n_trials
        self.enabled = bool(enabled and tqdm is not None)
        self._bar: Any = None

    def __enter__(self) -> LgrPropOptunaProgress:
        if self.enabled:
            self._bar = tqdm(
                total=self.n_trials,
                unit="trial",
                desc="Optuna[LGR-Prop]",
                dynamic_ncols=True,
                file=sys.stderr,
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._bar is not None:
            self._bar.close()

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if self._bar is None:
            return
        try:
            best = f"{float(study.best_value):.4f}"
        except ValueError:
            best = "n/a"
        last = trial.value if trial.value is not None else DISQUALIFY_SCORE
        self._bar.set_postfix({"best": best, "last": f"{float(last):.4f}"}, refresh=False)
        self._bar.update(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LGR Prop-Focused Optuna (WFT objective)")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--study-name", type=str, default="lgr_prop_v1")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--is-months", type=int, default=12)
    parser.add_argument("--oos-months", type=int, default=3)
    parser.add_argument("--step-months", type=int, default=3)
    parser.add_argument("--max-windows", type=int, default=None, help="Debug: limit WFT windows")
    parser.add_argument("--smoke", action="store_true", help="Short run: 3 trials, 2 windows")
    parser.add_argument("--gbp-m15", type=Path, default=DEFAULT_GBP_M15_FILE)
    parser.add_argument("--eur-m15", type=Path, default=DEFAULT_EUR_M15_FILE)
    parser.add_argument("--gbp-h1", type=Path, default=DEFAULT_GBP_FILE)
    parser.add_argument("--eur-h1", type=Path, default=DEFAULT_EUR_FILE)
    parser.add_argument("--trials-csv", type=Path, default=DEFAULT_TRIALS_CSV)
    parser.add_argument("--top20-md", type=Path, default=DEFAULT_TOP20_MD)
    parser.add_argument("--best-json", type=Path, default=DEFAULT_BEST_JSON)
    parser.add_argument("--wft-report-dir", type=Path, default=DEFAULT_WFT_REPORT)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.n_trials = min(args.n_trials, 3)
        args.max_windows = args.max_windows or 2
        args.is_months = min(args.is_months, 6)
        args.oos_months = min(args.oos_months, 2)
        args.step_months = min(args.step_months, 2)
    return args


def main() -> int:
    args = parse_args()
    enable_optuna_runtime()
    configure_lgr_prop_baseline_env()
    configure_lgr_production_detector()
    os.environ["BACKTEST_MODE"] = "1"

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.getLogger("backtest_runner").setLevel(logging.WARNING)
    logging.getLogger("walkforward_runner").setLevel(logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.WARNING)

    for path in (args.gbp_m15, args.eur_m15, args.gbp_h1, args.eur_h1):
        if not path.exists():
            logger.error("Missing data file: %s", path)
            return 1

    if args.trials_csv.is_file() and args.smoke:
        args.trials_csv.unlink()

    print("=" * 72)
    print(" LGR Prop Optuna - EV sizing + defense (no entry tuning)")
    print("=" * 72)
    print(f"  Trials        : {args.n_trials}")
    print(f"  WFT           : IS={args.is_months}m OOS={args.oos_months}m step={args.step_months}m")
    if args.max_windows:
        print(f"  Max windows   : {args.max_windows}")
    print(f"  Objective     : PF * Sharpe * (1 - dd_exceed_rate)")
    print(f"  Disqualify    : dd_exceed_rate > 10%")
    print(f"  Trials CSV    : {args.trials_csv}")
    print("=" * 72 + "\n")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
        direction="maximize",
        sampler=sampler,
    )

    objective = build_objective(args)
    with LgrPropOptunaProgress(args.n_trials, enabled=not args.no_progress) as progress:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            callbacks=[progress],
            catch=(Exception,),
        )

    clear_lgr_prop_trial_env()

    valid = _valid_trials(study)
    if not valid:
        print("\n[WARN] No valid trials completed.")
        return 1

    best = max(valid, key=lambda t: float(t.value or DISQUALIFY_SCORE))
    top20_md = format_top20_markdown(study)
    args.top20_md.parent.mkdir(parents=True, exist_ok=True)
    args.top20_md.write_text(top20_md, encoding="utf-8")

    best_params = LgrPropTrialParams(
        top5_risk=float(best.params["top5_risk"]),
        top20_risk=float(best.params["top20_risk"]),
        mid_risk=float(best.params["mid_risk"]),
        bottom_risk=float(best.params["bottom_risk"]),
        top_pct=int(best.params["top_pct"]),
        top20_pct=int(best.params["top20_pct"]),
        daily_stop_r=float(best.params["daily_stop_r"]),
        max_positions=int(best.params["max_positions"]),
        session_open_min=int(best.params["session_open_min"]),
        session_open_max=int(best.params["session_open_max"]),
    )
    best_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "optimizer": "optuna",
        "study_name": args.study_name,
        "trial_number": best.number,
        "score": float(best.value or 0.0),
        "params": best.params,
        "metrics": dict(best.user_attrs),
        "recommended_env": best_params.as_env(),
    }
    args.best_json.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(top20_md)
    print(f"\nWrote {args.top20_md}")
    print(f"Wrote {args.best_json}")
    print(f"Trial log: {args.trials_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
