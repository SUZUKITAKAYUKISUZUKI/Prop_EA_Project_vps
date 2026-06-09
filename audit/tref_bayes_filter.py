"""
audit/tref_bayes_filter.py — TREF L3.5 ベイズフィルター

Expected-R ゲート + BAILOUT ウォームアップ + スリム特徴量（pair + 4軸スコア）。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

TREF_BAYES_WIN_R = 2.0
TREF_BAYES_LOSS_R = 1.0
TREF_EXPECTED_R_PASS = float(os.getenv("TREF_EXPECTED_R_PASS", "0.05"))
TREF_BAILOUT_EVENT_LIMIT = 100
TREF_BAILOUT_MONTHS = 6
TREF_BAYES_PRIOR_ALPHA = 2.0
TREF_BAYES_PRIOR_BETA = 2.0
TREF_BAYES_BASE_WIN_RATE = 0.38
TREF_BAYES_MIN_MATCH_SAMPLES = 3
TREF_LOSS_PATTERN_MIN_LOSSES = int(os.getenv("TREF_LOSS_PATTERN_MIN_LOSSES", "40"))
TREF_LOSS_PATTERN_MIN_COMBO_SAMPLES = int(
    os.getenv("TREF_LOSS_PATTERN_MIN_COMBO_SAMPLES", "6")
)
TREF_LOSS_PATTERN_MAX_WIN_RATE = float(
    os.getenv("TREF_LOSS_PATTERN_MAX_WIN_RATE", "0.22")
)
TREF_LOSS_PATTERN_MIN_COMBO_LOSSES = int(
    os.getenv("TREF_LOSS_PATTERN_MIN_COMBO_LOSSES", "4")
)
TREF_LOSS_IMMATURE_AXIS1_MAX = int(os.getenv("TREF_LOSS_IMMATURE_AXIS1_MAX", "0"))
TREF_LOSS_IMMATURE_AXIS2_MIN = int(os.getenv("TREF_LOSS_IMMATURE_AXIS2_MIN", "20"))


def is_loss_immature_pattern_enabled() -> bool:
    raw = os.getenv("TREF_LOSS_IMMATURE_ENABLED", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


REASON_TREF_LOSS_IMMATURE_FORCED = "TREF_LOSS_IMMATURE_FORCED"
REASON_TREF_LOSS_TOXIC_COMBO = "TREF_LOSS_TOXIC_COMBO"

AXIS1_MAX = 25
AXIS2_MAX = 30
AXIS3_MAX = 20
AXIS4_MAX = 25

AXIS_KEYS = (
    "axis1_range_maturity",
    "axis2_expansion_depth",
    "axis3_time_sync",
    "axis4_total",
)


def normalize_tref_pair(pair: str) -> str:
    upper = pair.upper().replace(".", "").replace("_", "").replace("-", "")
    if "AUDJPY" in upper:
        return "AUDJPY"
    if "USDJPY" in upper:
        return "USDJPY"
    return upper[:6] if upper else "USDJPY"


def parse_tref_score_axes(score_breakdown: dict[str, Any] | None) -> tuple[int, int, int, int]:
    sb = score_breakdown if isinstance(score_breakdown, dict) else {}
    return (
        max(0, min(AXIS1_MAX, int(sb.get("axis1_range_maturity", 0) or 0))),
        max(0, min(AXIS2_MAX, int(sb.get("axis2_expansion_depth", 0) or 0))),
        max(0, min(AXIS3_MAX, int(sb.get("axis3_time_sync", 0) or 0))),
        max(0, min(AXIS4_MAX, int(sb.get("axis4_total", 0) or 0))),
    )


def compute_expected_r(bayes_probability: float) -> float:
    """Expected_R = P(win)*2R - P(loss)*1R  (TREF fixed RR 1:2)."""
    p = max(0.0, min(1.0, float(bayes_probability)))
    return round(p * TREF_BAYES_WIN_R - (1.0 - p) * TREF_BAYES_LOSS_R, 4)


def expected_r_passes(bayes_probability: float) -> bool:
    return compute_expected_r(bayes_probability) > TREF_EXPECTED_R_PASS


def matches_immature_forced_pattern(axis1: int, axis2: int) -> bool:
    """axis1 未成熟（0pt）かつ axis2 深い拡張（H 帯）のみ — 粗 L+M/H より厳格。"""
    return axis1 <= TREF_LOSS_IMMATURE_AXIS1_MAX and axis2 >= TREF_LOSS_IMMATURE_AXIS2_MIN


def _coarse_bucket(value: int, maximum: int) -> str:
    if maximum <= 0:
        return "L"
    ratio = value / maximum
    if ratio < 0.34:
        return "L"
    if ratio < 0.67:
        return "M"
    return "H"


@dataclass
class TrefBayesObservation:
    timestamp: pd.Timestamp
    pair: str
    axis1: int
    axis2: int
    axis3: int
    axis4: int
    won: bool


class TrefBayesFilter:
    """
    TREF 専用 L3.5 フィルター。

    - 特徴量: pair + candidate_score 4軸（生値保存・粗バケット照合）
    - 足切り: Expected_R > 0.05（勝率 ~35% 以上）
    - BAILOUT_MODE: 先頭 100 イベントまたは開始 6 ヶ月は REJECT_BY_BAYES をバイパス
    """

    def __init__(self) -> None:
        self.observations: list[TrefBayesObservation] = []
        self.event_count: int = 0
        self.backtest_start_ts: pd.Timestamp | None = None

    def reset(self) -> None:
        """ウォークフォワード学習データを完全初期化。"""
        self.observations.clear()
        self.event_count = 0
        self.backtest_start_ts = None
        logger.info("[TREF_BAYES] Learning data reset (observations=0, BAILOUT_MODE armed)")

    def _ensure_start(self, timestamp: pd.Timestamp) -> None:
        if self.backtest_start_ts is None:
            self.backtest_start_ts = pd.Timestamp(timestamp)

    def register_event(self, timestamp: pd.Timestamp) -> int:
        """TREF セットアップ評価ごとに 1 カウント（BAILOUT 判定用）。"""
        self._ensure_start(timestamp)
        self.event_count += 1
        return self.event_count

    def is_bailout_active(self, timestamp: pd.Timestamp) -> bool:
        if self.backtest_start_ts is None:
            return True
        if self.event_count <= TREF_BAILOUT_EVENT_LIMIT:
            return True
        ts = pd.Timestamp(timestamp)
        cutoff = self.backtest_start_ts + pd.DateOffset(months=TREF_BAILOUT_MONTHS)
        return ts < cutoff

    def _global_win_anchor(self, history: list[TrefBayesObservation]) -> float:
        if not history:
            return TREF_BAYES_BASE_WIN_RATE
        wins = sum(1 for o in history if o.won)
        return (wins + TREF_BAYES_PRIOR_ALPHA) / (
            len(history) + TREF_BAYES_PRIOR_ALPHA + TREF_BAYES_PRIOR_BETA
        )

    def _match_layers(
        self,
        pair: str,
        axis1: int,
        axis2: int,
        axis3: int,
        axis4: int,
    ) -> list[dict[str, Any]]:
        b1 = _coarse_bucket(axis1, AXIS1_MAX)
        b2 = _coarse_bucket(axis2, AXIS2_MAX)
        b3 = _coarse_bucket(axis3, AXIS3_MAX)
        b4 = _coarse_bucket(axis4, AXIS4_MAX)
        return [
            {"pair": pair, "a1": b1, "a2": b2, "a3": b3, "a4": b4},
            {"pair": pair, "a1": b1, "a2": b2, "a3": b3},
            {"pair": pair, "a1": b1, "a2": b2},
            {"pair": pair},
            {},
        ]

    @staticmethod
    def _obs_matches(obs: TrefBayesObservation, layer: dict[str, Any]) -> bool:
        if not layer:
            return True
        if obs.pair != layer.get("pair", obs.pair):
            return False
        for key, attr, maximum in (
            ("a1", "axis1", AXIS1_MAX),
            ("a2", "axis2", AXIS2_MAX),
            ("a3", "axis3", AXIS3_MAX),
            ("a4", "axis4", AXIS4_MAX),
        ):
            if key not in layer:
                continue
            if _coarse_bucket(getattr(obs, attr), maximum) != layer[key]:
                return False
        return True

    def compute_probability(
        self,
        timestamp: pd.Timestamp,
        pair: str,
        score_breakdown: dict[str, Any] | None,
    ) -> float:
        """P(win | pair, 4-axis scores) — walk-forward, no look-ahead."""
        pair_norm = normalize_tref_pair(pair)
        axis1, axis2, axis3, axis4 = parse_tref_score_axes(score_breakdown)
        history = [o for o in self.observations if o.timestamp < pd.Timestamp(timestamp)]
        anchor = self._global_win_anchor(history)

        if not history:
            return round(TREF_BAYES_BASE_WIN_RATE, 4)

        matched: list[TrefBayesObservation] = []
        for layer in self._match_layers(pair_norm, axis1, axis2, axis3, axis4):
            layer_matched = [o for o in history if self._obs_matches(o, layer)]
            if layer_matched:
                matched = layer_matched
                break

        if not matched:
            return round(anchor, 4)

        wins = sum(1 for o in matched if o.won)
        sample_n = len(matched)
        local_posterior = (wins + TREF_BAYES_PRIOR_ALPHA) / (
            sample_n + TREF_BAYES_PRIOR_ALPHA + TREF_BAYES_PRIOR_BETA
        )

        if sample_n < TREF_BAYES_MIN_MATCH_SAMPLES:
            blend_weight = sample_n / TREF_BAYES_MIN_MATCH_SAMPLES
            posterior = blend_weight * local_posterior + (1.0 - blend_weight) * anchor
        else:
            posterior = local_posterior

        return round(posterior, 4)

    def _history_before(self, timestamp: pd.Timestamp) -> list[TrefBayesObservation]:
        cutoff = pd.Timestamp(timestamp)
        return [o for o in self.observations if o.timestamp < cutoff]

    def _toxic_pair_axis_combos(
        self, history: list[TrefBayesObservation]
    ) -> set[tuple[str, str, str]]:
        """Walk-forward: 低勝率 (pair, axis1, axis2) バケット組み合わせ。"""
        stats: dict[tuple[str, str, str], list[bool]] = {}
        for obs in history:
            key = (
                obs.pair,
                _coarse_bucket(obs.axis1, AXIS1_MAX),
                _coarse_bucket(obs.axis2, AXIS2_MAX),
            )
            stats.setdefault(key, []).append(obs.won)

        toxic: set[tuple[str, str, str]] = set()
        for key, outcomes in stats.items():
            sample_n = len(outcomes)
            if sample_n < TREF_LOSS_PATTERN_MIN_COMBO_SAMPLES:
                continue
            wins = sum(1 for won in outcomes if won)
            loss_count = sample_n - wins
            win_rate = wins / sample_n
            if (
                win_rate <= TREF_LOSS_PATTERN_MAX_WIN_RATE
                and loss_count >= TREF_LOSS_PATTERN_MIN_COMBO_LOSSES
            ):
                toxic.add(key)
        return toxic

    def check_loss_pattern_reject(
        self,
        timestamp: pd.Timestamp,
        pair: str,
        score_breakdown: dict[str, Any] | None,
    ) -> tuple[bool, str | None]:
        """
        BAILOUT 終了後: 失敗トレードの共通特性に基づく動的カット。

        Expected-R に関わらず REJECT する追加フィルタ。
        """
        if self.is_bailout_active(timestamp):
            return False, None

        history = self._history_before(timestamp)
        loss_count = sum(1 for o in history if not o.won)
        if loss_count < TREF_LOSS_PATTERN_MIN_LOSSES:
            return False, None

        pair_norm = normalize_tref_pair(pair)
        axis1, axis2, _, _ = parse_tref_score_axes(score_breakdown)
        b1 = _coarse_bucket(axis1, AXIS1_MAX)
        b2 = _coarse_bucket(axis2, AXIS2_MAX)

        if is_loss_immature_pattern_enabled() and matches_immature_forced_pattern(axis1, axis2):
            logger.debug(
                "[TREF_BAYES] LOSS_PATTERN reject immature_forced pair=%s a1=%d a2=%d",
                pair_norm,
                axis1,
                axis2,
            )
            return True, REASON_TREF_LOSS_IMMATURE_FORCED

        if (pair_norm, b1, b2) in self._toxic_pair_axis_combos(history):
            logger.debug(
                "[TREF_BAYES] LOSS_PATTERN reject toxic_combo pair=%s a1=%s a2=%s",
                pair_norm,
                b1,
                b2,
            )
            return True, REASON_TREF_LOSS_TOXIC_COMBO

        return False, None

    def check_hard_reject(self, timestamp: pd.Timestamp, bayes_probability: float) -> bool:
        """
        Expected-R ベース足切り。BAILOUT_MODE 中は常に False（LLM へ直通）。
        """
        if self.is_bailout_active(timestamp):
            logger.info(
                "[BAYES_BAILOUT] Warmup active: Event #%d - Forced ALLOW "
                "(bayes_p=%.4f expected_r=%.4f)",
                self.event_count,
                bayes_probability,
                compute_expected_r(bayes_probability),
            )
            return False
        expected_r = compute_expected_r(bayes_probability)
        reject = expected_r <= TREF_EXPECTED_R_PASS
        if reject:
            logger.debug(
                "[TREF_BAYES] REJECT_BY_BAYES expected_r=%.4f bayes_p=%.4f event=#%d",
                expected_r,
                bayes_probability,
                self.event_count,
            )
        return reject

    def record_outcome(
        self,
        timestamp: pd.Timestamp,
        pair: str,
        score_breakdown: dict[str, Any] | None,
        won: bool,
    ) -> None:
        axis1, axis2, axis3, axis4 = parse_tref_score_axes(score_breakdown)
        self.observations.append(
            TrefBayesObservation(
                timestamp=pd.Timestamp(timestamp),
                pair=normalize_tref_pair(pair),
                axis1=axis1,
                axis2=axis2,
                axis3=axis3,
                axis4=axis4,
                won=bool(won),
            )
        )
