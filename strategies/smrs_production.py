"""Statistical Mean Reversion Scalper (SMRS) — Strategy E production configuration.

Single source of truth for live / VPS defaults (Strategy E).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

# --- Live pipeline defaults (frozen production spec) ---
PRODUCTION_TIMEFRAME = "M1"
PRODUCTION_MIN_M1_BARS = 120
PRODUCTION_BAYES_ENABLED = True
PRODUCTION_GEMINI_AUDIT = False
PRODUCTION_LLM_AUDIT = False
PRODUCTION_PYRAMIDING = False
PRODUCTION_L2_MIN_SCORE = 0
PRODUCTION_DEFENSE_ENABLED = True

# Env var names + default string values (.env / bridge startup)
SMRS_DEFENSE_ENV = "SMRS_DEFENSE"
SMRS_DEFENSE_DEFAULT = "1"
SMRS_PURE_BT_ENV = "SMRS_PURE_BT"
SMRS_PURE_BT_DEFAULT = "0"
SMRS_GEMINI_AUDIT_ENV = "SMRS_GEMINI_AUDIT"
SMRS_GEMINI_AUDIT_DEFAULT = "0"
SMRS_LLM_AUDIT_ENV = "SMRS_LLM_AUDIT"
SMRS_LLM_AUDIT_DEFAULT = "0"
SMRS_L2_MIN_SCORE_ENV = "SMRS_L2_MIN_SCORE"
SMRS_L2_MIN_SCORE_DEFAULT = "0"
SMRS_BAYES_MODEL_ENV = "SMRS_BAYES_MODEL"
PYRAMID_SMRS_ENV = "PYRAMID_SMRS"
PYRAMID_SMRS_DEFAULT = "0"

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


@dataclass(frozen=True)
class SmrsProductionSpec:
    """Human-readable production default snapshot (Strategy E)."""

    letter: str
    setup_type: str
    full_name: str
    abbrev: str
    pairs: tuple[str, ...]
    timeframe: str
    sizing_model: str
    bayes_enabled: bool
    gemini_audit: bool
    llm_audit: bool
    pyramiding: bool
    l2_min_score: int
    defense_layers: bool
    profit_cushion: bool
    twin_brake: bool
    dd_throttling: bool
    envelope_dev: float
    z_threshold: float
    session_filter: str
    exit_logic: str
    max_hold_hours: int

    @classmethod
    def from_production(cls) -> SmrsProductionSpec:
        params = PRODUCTION_PARAMS
        return cls(
            letter=STRATEGY_LETTER,
            setup_type=SETUP_TYPE_SMRS,
            full_name=STRATEGY_FULL_NAME,
            abbrev=STRATEGY_ABBREV,
            pairs=SMRS_PAIRS,
            timeframe=PRODUCTION_TIMEFRAME,
            sizing_model=PRODUCTION_SIZING_MODEL,
            bayes_enabled=PRODUCTION_BAYES_ENABLED,
            gemini_audit=PRODUCTION_GEMINI_AUDIT,
            llm_audit=PRODUCTION_LLM_AUDIT,
            pyramiding=PRODUCTION_PYRAMIDING,
            l2_min_score=PRODUCTION_L2_MIN_SCORE,
            defense_layers=PRODUCTION_DEFENSE_ENABLED,
            profit_cushion=True,
            twin_brake=True,
            dd_throttling=True,
            envelope_dev=params.envelope_dev,
            z_threshold=params.z_threshold,
            session_filter=params.session_filter,
            exit_logic=params.exit_logic,
            max_hold_hours=params.max_hold_hours,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "letter": self.letter,
            "setup_type": self.setup_type,
            "full_name": self.full_name,
            "abbrev": self.abbrev,
            "pairs": list(self.pairs),
            "timeframe": self.timeframe,
            "sizing_model": self.sizing_model,
            "bayes_enabled": self.bayes_enabled,
            "gemini_audit": self.gemini_audit,
            "llm_audit": self.llm_audit,
            "pyramiding": self.pyramiding,
            "l2_min_score": self.l2_min_score,
            "defense_layers": self.defense_layers,
            "profit_cushion": self.profit_cushion,
            "twin_brake": self.twin_brake,
            "dd_throttling": self.dd_throttling,
            "entry": {
                "envelope_dev": self.envelope_dev,
                "z_threshold": self.z_threshold,
                "session_filter": self.session_filter,
                "exit_logic": self.exit_logic,
                "max_hold_hours": self.max_hold_hours,
            },
            "sizing_table": [f"{label}: {value}" for label, value in SIZING_TABLE],
        }


PRODUCTION_SPEC = SmrsProductionSpec.from_production()


def _env_on(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def configure_smrs_defense_env() -> None:
    """Apply Strategy E production defaults to the process environment (idempotent)."""
    if _env_on(SMRS_PURE_BT_ENV, SMRS_PURE_BT_DEFAULT):
        return
    os.environ.setdefault(SMRS_DEFENSE_ENV, SMRS_DEFENSE_DEFAULT)
    os.environ.setdefault(SMRS_GEMINI_AUDIT_ENV, SMRS_GEMINI_AUDIT_DEFAULT)
    os.environ.setdefault(SMRS_LLM_AUDIT_ENV, SMRS_LLM_AUDIT_DEFAULT)
    os.environ.setdefault(SMRS_L2_MIN_SCORE_ENV, SMRS_L2_MIN_SCORE_DEFAULT)
    os.environ.setdefault(PYRAMID_SMRS_ENV, PYRAMID_SMRS_DEFAULT)
    os.environ.setdefault("LLM_AUDIT_ENABLED", "0")
    os.environ.setdefault("PROFIT_CUSHION_ENABLED", "1")
    os.environ.setdefault("TWIN_BRAKE_ENABLED", "1")
    os.environ.setdefault("DD_THROTTLING_ENABLED", "1")


__all__ = [
    "DEFAULT_ABCD_CSV",
    "DEFAULT_BAYES_MODEL_JSON",
    "DEFAULT_BAYES_OOS_CSV",
    "DEFAULT_FEATURES_CSV",
    "DEFAULT_PORTFOLIO_CSV",
    "DEFAULT_PORTFOLIO_SUMMARY_MD",
    "DEFAULT_WFT_SUMMARY_MD",
    "PRODUCTION_BAYES_ENABLED",
    "PRODUCTION_DEFENSE_ENABLED",
    "PRODUCTION_GEMINI_AUDIT",
    "PRODUCTION_L2_MIN_SCORE",
    "PRODUCTION_LLM_AUDIT",
    "PRODUCTION_MIN_M1_BARS",
    "PRODUCTION_PARAMS",
    "PRODUCTION_PYRAMIDING",
    "PRODUCTION_SIZING_MODEL",
    "PRODUCTION_SPEC",
    "PRODUCTION_TIMEFRAME",
    "PYRAMID_SMRS_DEFAULT",
    "PYRAMID_SMRS_ENV",
    "SETUP_TYPE_SMRS",
    "SIZING_TABLE",
    "SMRS_BAYES_MODEL_ENV",
    "SMRS_DEFENSE_DEFAULT",
    "SMRS_DEFENSE_ENV",
    "SMRS_GEMINI_AUDIT",
    "SMRS_GEMINI_AUDIT_DEFAULT",
    "SMRS_GEMINI_AUDIT_ENV",
    "SMRS_L2_MIN_SCORE_DEFAULT",
    "SMRS_L2_MIN_SCORE_ENV",
    "SMRS_LLM_AUDIT",
    "SMRS_LLM_AUDIT_DEFAULT",
    "SMRS_LLM_AUDIT_ENV",
    "SMRS_PAIRS",
    "SMRS_PURE_BT_DEFAULT",
    "SMRS_PURE_BT_ENV",
    "SmrsProductionSpec",
    "STRATEGY_ABBREV",
    "STRATEGY_FULL_NAME",
    "STRATEGY_LETTER",
    "configure_smrs_defense_env",
]
