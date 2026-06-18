"""Governor decision type enumeration."""
from __future__ import annotations

from enum import Enum


class DecisionType(str, Enum):
    PROFILE_SWITCH = "PROFILE_SWITCH"
    ALLOCATION_REBALANCE = "ALLOCATION_REBALANCE"
    PROMOTE_STRATEGY = "PROMOTE_STRATEGY"
    DEMOTE_STRATEGY = "DEMOTE_STRATEGY"
    RETIRE_STRATEGY = "RETIRE_STRATEGY"
    ENTER_RECOVERY = "ENTER_RECOVERY"
    EXIT_RECOVERY = "EXIT_RECOVERY"
    REDUCE_RISK = "REDUCE_RISK"
    RISK_ALERT = "RISK_ALERT"
    HEALTH_ALERT = "HEALTH_ALERT"
    NO_ACTION = "NO_ACTION"

    @classmethod
    def values(cls) -> set[str]:
        return {item.value for item in cls}
