"""Live bar buffer requirements — keep in sync with mt5/PropEA_Bridge.mq5 RequiredHistoryBars()."""

from __future__ import annotations

from strategies.smrs_pure import SMRS_PAIRS

# VAMR WARMUP_BARS=120 on H1; DBBS needs len(h1)>=60.
# M5: 1800 bars ≈ 150 H1 bars. M1 (SMRS charts): 7500 bars ≈ 125 H1 + SMRS rolling stats.
LIVE_M5_BAR_BUFFER_MAX = 2000
LIVE_M1_BAR_BUFFER_MAX = 8000
LIVE_M5_HISTORY_BARS = 1800
LIVE_M1_HISTORY_BARS = 7500

M1_CHART_PAIRS = frozenset(SMRS_PAIRS)


def live_bar_buffer_max(pair: str) -> int:
    if str(pair).upper() in M1_CHART_PAIRS:
        return LIVE_M1_BAR_BUFFER_MAX
    return LIVE_M5_BAR_BUFFER_MAX


def max_spread_points_for_pair(pair: str, base: int = 30) -> int:
    """Keep in sync with mt5/PropEA_Bridge.mq5 MaxSpreadForSymbol()."""
    canonical = str(pair).upper()
    if canonical == "XAUUSD":
        return max(base, 80)
    if canonical in ("AUDNZD", "EURGBP", "NZDUSD"):
        return max(base, 45)
    if canonical == "USDCAD":
        return max(base, 40)
    return base


def infer_bar_minutes(df) -> int:
    """Infer bar size from the last two timestamps (1=M1 chart, 5=M5 chart)."""
    if df is None or df.empty or len(df) < 2:
        return 5
    delta = df["datetime"].iloc[-1] - df["datetime"].iloc[-2]
    minutes = int(round(delta.total_seconds() / 60.0))
    return minutes if minutes > 0 else 5
