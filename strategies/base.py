"""Shared strategy layer types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StrategyResult:
    """Strategy layer output consumed by audit / logging layers."""

    is_setup: bool
    setup_type: str
    direction: str
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    candidate_score: float = 0.0
    raw_features: dict[str, Any] = field(default_factory=dict)
    # v3.4 multi-regime: 戦略層 Pre-L0 判定（L4.5 は base_risk_pct=None 時プロファイル既定を使用）
    strategy_action: str = "ALLOW"
    base_risk_pct: float | None = None
