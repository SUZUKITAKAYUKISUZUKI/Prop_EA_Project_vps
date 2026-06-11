"""
lgr_prop_controls.py — LGR Prop-Focused Optuna パラメータ制御

エントリー条件は触らず、EV サイジング倍率・Rank 境界・Daily Stop R・
最大同時保有・セッションオープン時間のみを環境変数経由で上書きする。
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

DISQUALIFY_SCORE = -1_000_000_000.0

ENV_TOP5_RISK = "LGR_EV_TOP5_RISK"
ENV_TOP20_RISK = "LGR_EV_TOP20_RISK"
ENV_MID_RISK = "LGR_EV_MID_RISK"
ENV_BOTTOM_RISK = "LGR_EV_BOTTOM_RISK"
ENV_TOP_PCT = "LGR_EV_TOP_PCT"
ENV_TOP20_PCT = "LGR_EV_TOP20_PCT"
ENV_DAILY_STOP_R = "LGR_DAILY_STOP_R"
ENV_MAX_POSITIONS = "LGR_MAX_POSITIONS"
ENV_SESSION_OPEN_MIN = "LGR_SESSION_OPEN_MIN"
ENV_SESSION_OPEN_MAX = "LGR_SESSION_OPEN_MAX"

_PROP_ENV_KEYS = (
    ENV_TOP5_RISK,
    ENV_TOP20_RISK,
    ENV_MID_RISK,
    ENV_BOTTOM_RISK,
    ENV_TOP_PCT,
    ENV_TOP20_PCT,
    ENV_DAILY_STOP_R,
    ENV_MAX_POSITIONS,
    ENV_SESSION_OPEN_MIN,
    ENV_SESSION_OPEN_MAX,
)


@dataclass(frozen=True)
class LgrPropTrialParams:
    top5_risk: float
    top20_risk: float
    mid_risk: float
    bottom_risk: float
    top_pct: int
    top20_pct: int
    daily_stop_r: float
    max_positions: int
    session_open_min: int
    session_open_max: int

    def validate(self) -> str | None:
        if not (self.top5_risk >= self.top20_risk >= self.mid_risk >= self.bottom_risk):
            return "risk_ordering"
        if self.top_pct >= self.top20_pct:
            return "rank_pct_ordering"
        if self.session_open_min >= self.session_open_max:
            return "session_open_ordering"
        return None

    def as_env(self) -> dict[str, str]:
        return {
            ENV_TOP5_RISK: str(self.top5_risk),
            ENV_TOP20_RISK: str(self.top20_risk),
            ENV_MID_RISK: str(self.mid_risk),
            ENV_BOTTOM_RISK: str(self.bottom_risk),
            ENV_TOP_PCT: str(self.top_pct),
            ENV_TOP20_PCT: str(self.top20_pct),
            ENV_DAILY_STOP_R: str(self.daily_stop_r),
            ENV_MAX_POSITIONS: str(self.max_positions),
            ENV_SESSION_OPEN_MIN: str(self.session_open_min),
            ENV_SESSION_OPEN_MAX: str(self.session_open_max),
        }

    def ev_tiers(self) -> tuple[tuple[float, float], ...]:
        top_cut = 1.0 - self.top_pct / 100.0
        top20_cut = 1.0 - self.top20_pct / 100.0
        return (
            (top_cut, self.top5_risk),
            (top20_cut, self.top20_risk),
            (0.50, self.mid_risk),
            (0.0, self.bottom_risk),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "top5_risk": self.top5_risk,
            "top20_risk": self.top20_risk,
            "mid_risk": self.mid_risk,
            "bottom_risk": self.bottom_risk,
            "top_pct": self.top_pct,
            "top20_pct": self.top20_pct,
            "daily_stop_r": self.daily_stop_r,
            "max_positions": self.max_positions,
            "session_open_min": self.session_open_min,
            "session_open_max": self.session_open_max,
        }


def _float_env(name: str, default: float | None = None) -> float | None:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return float(raw)


def _int_env(name: str, default: int | None = None) -> int | None:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(float(raw))


def resolve_prop_ev_tiers() -> tuple[tuple[float, float], ...] | None:
    top5 = _float_env(ENV_TOP5_RISK)
    top20 = _float_env(ENV_TOP20_RISK)
    mid = _float_env(ENV_MID_RISK)
    bottom = _float_env(ENV_BOTTOM_RISK)
    top_pct = _int_env(ENV_TOP_PCT)
    top20_pct = _int_env(ENV_TOP20_PCT)
    if None in (top5, top20, mid, bottom, top_pct, top20_pct):
        return None
    params = LgrPropTrialParams(
        top5_risk=top5,
        top20_risk=top20,
        mid_risk=mid,
        bottom_risk=bottom,
        top_pct=top_pct,
        top20_pct=top20_pct,
        daily_stop_r=_float_env(ENV_DAILY_STOP_R, -3.0) or -3.0,
        max_positions=_int_env(ENV_MAX_POSITIONS, 1) or 1,
        session_open_min=_int_env(ENV_SESSION_OPEN_MIN, 0) or 0,
        session_open_max=_int_env(ENV_SESSION_OPEN_MAX, 120) or 120,
    )
    if params.validate() is not None:
        return None
    return params.ev_tiers()


def lgr_daily_stop_r_threshold() -> float | None:
    return _float_env(ENV_DAILY_STOP_R)


def lgr_max_open_positions() -> int | None:
    value = _int_env(ENV_MAX_POSITIONS)
    if value is None:
        return None
    return max(1, value)


def lgr_session_open_bounds() -> tuple[int, int] | None:
    min_raw = _int_env(ENV_SESSION_OPEN_MIN)
    max_raw = _int_env(ENV_SESSION_OPEN_MAX)
    if min_raw is None or max_raw is None:
        return None
    if min_raw >= max_raw:
        return None
    return min_raw, max_raw


def session_open_minutes_reject(minutes: int) -> bool:
    bounds = lgr_session_open_bounds()
    if bounds is None:
        return False
    low, high = bounds
    return minutes < low or minutes >= high


def apply_lgr_prop_trial_env(params: LgrPropTrialParams) -> None:
    for key, value in params.as_env().items():
        os.environ[key] = value


def clear_lgr_prop_trial_env() -> None:
    for key in _PROP_ENV_KEYS:
        os.environ.pop(key, None)


def configure_lgr_prop_baseline_env() -> None:
    """LGR 基本設定: L0 + EV Pattern C + 3 安全装置 ON。"""
    os.environ["LGR_DEFENSE_BT"] = "1"
    os.environ["LGR_EV_SIZING"] = "1"
    os.environ.setdefault("LGR_EV_PATTERN", "C")
    os.environ["LGR_L4_BYPASS"] = "1"
    os.environ.pop("LGR_PURE_DATA_MODE", None)
    os.environ.pop("LGR_BAYES_GATE_ENABLED", None)
    os.environ.pop("LGR_BAYES_ONLY_BT", None)
    os.environ["PROFIT_CUSHION_ENABLED"] = "1"
    os.environ["TWIN_BRAKE_ENABLED"] = "1"
    os.environ["DD_THROTTLING_ENABLED"] = "1"
    os.environ.setdefault("LGR_EV_TRAIN_CSV", "backtest_results/archive/lgr/logs/lgr_features.csv")
    os.environ.setdefault("LGR_LOOKBACK_BARS", "0")
    os.environ.setdefault("LGR_SCAN_NUMPY", "1")
    os.environ.setdefault("LGR_MAX_SETUPS_PER_DAY", "0")
    os.environ.setdefault("BT_SCAN_PARALLEL_PAIRS", "1")


@contextmanager
def lgr_prop_trial_env(params: LgrPropTrialParams) -> Iterator[None]:
    saved = {key: os.environ.get(key) for key in _PROP_ENV_KEYS}
    apply_lgr_prop_trial_env(params)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def compute_prop_score(pf: float, sharpe: float, dd_exceed_rate: float) -> float:
    if dd_exceed_rate > 0.10:
        return DISQUALIFY_SCORE
    return pf * sharpe * (1.0 - dd_exceed_rate)
