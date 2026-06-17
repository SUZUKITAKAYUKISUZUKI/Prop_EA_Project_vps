"""Analytics engines package."""

from src.analytics.correlation_engine import rolling_corr, strategy_correlation, symbol_correlation
from src.analytics.equity_engine import attach_equity_columns, build_equity_curve
from src.analytics.risk_metrics_engine import (
    cagr,
    expectancy,
    losing_streak,
    max_dd,
    profit_factor,
    recovery_factor,
    risk_of_ruin,
    sharpe_ratio,
    ulcer_index,
)

__all__ = [
    "attach_equity_columns",
    "build_equity_curve",
    "cagr",
    "expectancy",
    "losing_streak",
    "max_dd",
    "profit_factor",
    "recovery_factor",
    "risk_of_ruin",
    "rolling_corr",
    "sharpe_ratio",
    "strategy_correlation",
    "symbol_correlation",
    "ulcer_index",
]
