"""Statistical Mean Reversion Scalper (SMRS) — Strategy E production configuration."""

from __future__ import annotations

from pathlib import Path

from strategies.smrs_pure import (
    PURE_BASELINE,
    SMRS_GEMINI_AUDIT,
    SMRS_LLM_AUDIT,
    SMRS_PAIRS,
    SETUP_TYPE,
    STRATEGY_ABBREV,
    STRATEGY_FULL_NAME,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Strategy letter / setup type (portfolio E)
STRATEGY_LETTER = "E"
SETUP_TYPE_SMRS = SETUP_TYPE

# Production sizing — Phase 4 Model A (Conservative Tier)
PRODUCTION_SIZING_MODEL = "A"
SIZING_TABLE: tuple[tuple[str, str], ...] = (
    ("p < 0.70", "SKIP"),
    ("0.70 – 0.80", "0.50R"),
    ("0.80 – 0.90", "1.00R"),
    ("p ≥ 0.90", "1.50R"),
)

# Frozen entry parameters (Phase S1 baseline)
PRODUCTION_PARAMS = PURE_BASELINE

# Bayesian model artifact (Phase 3 full-period reference)
DEFAULT_BAYES_MODEL_JSON = PROJECT_ROOT / "backtest_results" / "models" / "smrs_bayes_v1.json"
DEFAULT_FEATURES_CSV = PROJECT_ROOT / "backtest_results" / "logs" / "smrs_features_pure_3y.csv"
DEFAULT_BAYES_OOS_CSV = PROJECT_ROOT / "backtest_results" / "phase3_smrs" / "smrs_phase3_bayes_results.csv"

# Canonical portfolio outputs (A+B+C+D+E)
DEFAULT_ABCD_CSV = PROJECT_ROOT / "backtest_results" / "main_abcd_3y.csv"
DEFAULT_PORTFOLIO_CSV = PROJECT_ROOT / "backtest_results" / "main_abcde_3y.csv"
DEFAULT_WFT_SUMMARY_MD = PROJECT_ROOT / "backtest_results" / "wft_summary_abcde.md"
DEFAULT_PORTFOLIO_SUMMARY_MD = PROJECT_ROOT / "reports" / "main_abcde_3y_portfolio_summary.md"

__all__ = [
    "DEFAULT_ABCD_CSV",
    "DEFAULT_BAYES_MODEL_JSON",
    "DEFAULT_BAYES_OOS_CSV",
    "DEFAULT_FEATURES_CSV",
    "DEFAULT_PORTFOLIO_CSV",
    "DEFAULT_PORTFOLIO_SUMMARY_MD",
    "DEFAULT_WFT_SUMMARY_MD",
    "PRODUCTION_PARAMS",
    "PRODUCTION_SIZING_MODEL",
    "SETUP_TYPE_SMRS",
    "SIZING_TABLE",
    "SMRS_GEMINI_AUDIT",
    "SMRS_LLM_AUDIT",
    "SMRS_PAIRS",
    "STRATEGY_ABBREV",
    "STRATEGY_FULL_NAME",
    "STRATEGY_LETTER",
]
