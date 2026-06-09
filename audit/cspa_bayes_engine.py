"""
audit/cspa_bayes_engine.py — CSPA 多次元動的ベイズゲート推論エンジン

3-Tier 階層（Primary Gate → Regime Index → Dynamic Multiplier）で
MT5 リアルタイム特徴量から ALLOW/REJECT とロット/TP 倍率を決定する。

Tier 1 閾値マトリクスは 3y pure BT（111,437 件）の decile グリッドから導出。
将来は JSON 設定ファイルからロード可能。
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path
from typing import Any, Literal, TypedDict

Decision = Literal["ALLOW", "REJECT"]
SessionType = Literal["ASIA", "LONDON", "NY"]
AtrRegime = Literal["Low-Vol", "Mid-Vol", "High-Vol"]


class TradeEvaluation(TypedDict):
    """``evaluate_trade`` の戻り値。"""

    decision: Decision
    reason: str
    lot_multiplier: float
    tp_multiplier: float


class RegimeDefaults(TypedDict):
    lot_multiplier: float
    tp_multiplier: float


class Tier1Config(TypedDict):
    rf_edges: list[float]
    ra_edges: list[float]
    wr_matrix: list[list[float]]
    avg_r_matrix: list[list[float]]
    min_win_rate: float
    min_avg_r: float


# --- Tier 1: decile グリッド（reaccel_follow_through × reacceleration_score）---
# Source: backtest_results/logs/cspa_bayes_features_pure_3y.csv (111,437 rows)
_DEFAULT_RF_EDGES: list[float] = [
    -0.00558, -0.00015, -9e-05, -5e-05, -2e-05, 0.0, 2e-05, 5e-05, 9e-05, 0.00015, 0.00492,
]
_DEFAULT_RA_EDGES: list[float] = [
    0.2132, 0.3667, 0.4266, 0.4872, 0.5508, 0.594, 0.6326, 0.6757, 0.7553, 0.8603, 1.0,
]
_DEFAULT_WR_MATRIX: list[list[float]] = [
    [0.2449, 0.2414, 0.2899, 0.2880, 0.3385, 0.3454, 0.3654, 0.4242, 0.0000, 0.0000],
    [0.3913, 0.4344, 0.4221, 0.4204, 0.4497, 0.4681, 0.4187, 0.4182, 0.0000, 0.0000],
    [0.4995, 0.4995, 0.5190, 0.4908, 0.5332, 0.5256, 0.5238, 0.4828, 0.0000, 0.0000],
    [0.6005, 0.5852, 0.6009, 0.6159, 0.6038, 0.6141, 0.5827, 0.4675, 0.0000, 0.0000],
    [0.6908, 0.6719, 0.6932, 0.6861, 0.6529, 0.6834, 0.6710, 0.6203, 0.0000, 0.0000],
    [0.7679, 0.7632, 0.7686, 0.7715, 0.7608, 0.7544, 0.7385, 0.7679, 0.7811, 0.8588],
    [0.8571, 0.8379, 0.8350, 0.8544, 0.8459, 0.8676, 0.8584, 0.8608, 0.8687, 0.8776],
    [1.0000, 0.6923, 0.9406, 0.8964, 0.9453, 0.9407, 0.9497, 0.9446, 0.9307, 0.9260],
    [0.0000, 1.0000, 0.9048, 0.9255, 0.9825, 0.9814, 0.9883, 0.9817, 0.9719, 0.9596],
    [0.0000, 0.0000, 1.0000, 1.0000, 0.9940, 0.9937, 0.9958, 0.9971, 0.9935, 0.9873],
]
_DEFAULT_AVG_R_MATRIX: list[list[float]] = [
    [-0.7151, -0.7131, -0.6409, -0.6313, -0.5728, -0.5551, -0.5127, -0.4527, -1.0000, -1.0000],
    [-0.5427, -0.4844, -0.4858, -0.4860, -0.4414, -0.3929, -0.4575, -0.4299, -1.0000, -1.0000],
    [-0.4191, -0.3812, -0.3653, -0.3923, -0.3344, -0.3316, -0.2974, -0.3589, -1.0000, -1.0000],
    [-0.2805, -0.3088, -0.2648, -0.2283, -0.2280, -0.1983, -0.2429, -0.3402, -1.0000, -1.0000],
    [-0.1561, -0.1717, -0.1338, -0.1580, -0.1694, -0.1165, -0.1082, -0.1621, -1.0000, -1.0000],
    [-0.0571, -0.0893, -0.0791, -0.0274, -0.0049, -0.0442, -0.0425, -0.0054, 0.0926, 0.3043],
    [0.0620, -0.0196, -0.0190, 0.0389, 0.0442, 0.0768, 0.0540, 0.0714, 0.1200, 0.1677],
    [0.1606, -0.0898, 0.0529, 0.0675, 0.1433, 0.1446, 0.1621, 0.1764, 0.1513, 0.1967],
    [-1.0000, 0.1678, 0.0243, 0.1489, 0.1938, 0.2117, 0.2018, 0.2137, 0.2127, 0.2392],
    [-1.0000, -1.0000, 0.0937, 0.3637, 0.3353, 0.3409, 0.3792, 0.3577, 0.3580, 0.3814],
]

# --- Tier 3: High 判定用パーセンタイル境界（75%ile, 3y BT）---
_DEFAULT_PERCENTILE_HIGH: dict[str, float] = {
    "rhythm_score": 0.8447,
    "market_breath_score": 47.83,
    "breakout_velocity": 1.2,
}

# --- Tier 2: ATR H1 3分位（Low/Mid/High-Vol）---
_DEFAULT_ATR_TERTILES: tuple[float, float] = (0.001201, 0.001669)

# --- Tier 2: 9 レジーム別ベース倍率 ---
_DEFAULT_REGIME_DEFAULTS: dict[tuple[SessionType, AtrRegime], RegimeDefaults] = {
    ("ASIA", "Low-Vol"): {"lot_multiplier": 1.0, "tp_multiplier": 1.0},
    ("ASIA", "Mid-Vol"): {"lot_multiplier": 1.0, "tp_multiplier": 1.0},
    ("ASIA", "High-Vol"): {"lot_multiplier": 1.0, "tp_multiplier": 1.0},
    ("LONDON", "Low-Vol"): {"lot_multiplier": 1.0, "tp_multiplier": 1.0},
    ("LONDON", "Mid-Vol"): {"lot_multiplier": 1.0, "tp_multiplier": 1.0},
    ("LONDON", "High-Vol"): {"lot_multiplier": 1.0, "tp_multiplier": 1.0},
    ("NY", "Low-Vol"): {"lot_multiplier": 0.9, "tp_multiplier": 1.0},
    ("NY", "Mid-Vol"): {"lot_multiplier": 0.9, "tp_multiplier": 1.0},
    ("NY", "High-Vol"): {"lot_multiplier": 0.8, "tp_multiplier": 1.5},
}

_TIER3_RHYTHM_BREATH_LOT_BOOST = 1.5
_TIER3_VELOCITY_LOT_FACTOR = 0.8
_TIER3_VELOCITY_TP_FACTOR = 2.0


def _decile_bin(value: float, edges: list[float]) -> int:
    """連続値を decile ビン (0..9) に割り当てる。"""
    idx = bisect.bisect_right(edges, value) - 1
    if idx < 0:
        return 0
    if idx >= len(edges) - 1:
        return len(edges) - 2
    return idx


def _normalize_session(raw: str) -> SessionType:
    key = str(raw).strip().upper()
    if key not in ("ASIA", "LONDON", "NY"):
        return "ASIA"
    return key  # type: ignore[return-value]


def _normalize_atr_regime(raw: str) -> AtrRegime | None:
    key = str(raw).strip()
    if key in ("Low-Vol", "Mid-Vol", "High-Vol"):
        return key  # type: ignore[return-value]
    return None


class CSPABayesEngine:
    """
    CSPA 多次元動的ベイズゲート。

    Tier 1: reaccel_follow_through × reacceleration_score の decile マトリクスで即時 REJECT。
    Tier 2: session × ATR レジームでベース lot/tp を決定。
    Tier 3: rhythm/breath/velocity で lot/tp を動的補正。
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Parameters
        ----------
        config
            省略時は 3y BT から導出したハードコード既定値を使用。
            キー例: ``tier1``, ``percentile_high``, ``atr_tertiles``, ``regime_defaults``。
        """
        cfg = config or {}
        tier1_raw = cfg.get("tier1", {})
        self._tier1: Tier1Config = {
            "rf_edges": list(tier1_raw.get("rf_edges", _DEFAULT_RF_EDGES)),
            "ra_edges": list(tier1_raw.get("ra_edges", _DEFAULT_RA_EDGES)),
            "wr_matrix": [list(row) for row in tier1_raw.get("wr_matrix", _DEFAULT_WR_MATRIX)],
            "avg_r_matrix": [list(row) for row in tier1_raw.get("avg_r_matrix", _DEFAULT_AVG_R_MATRIX)],
            "min_win_rate": float(tier1_raw.get("min_win_rate", 0.90)),
            "min_avg_r": float(tier1_raw.get("min_avg_r", 0.15)),
        }
        self._percentile_high: dict[str, float] = dict(
            cfg.get("percentile_high", _DEFAULT_PERCENTILE_HIGH)
        )
        tert = cfg.get("atr_tertiles", _DEFAULT_ATR_TERTILES)
        self._atr_tertiles: tuple[float, float] = (float(tert[0]), float(tert[1]))
        self._regime_defaults = self._load_regime_defaults(cfg.get("regime_defaults"))
        tier3 = cfg.get("tier3", {})
        self._rhythm_breath_lot_boost = float(
            tier3.get("rhythm_breath_lot_boost", _TIER3_RHYTHM_BREATH_LOT_BOOST)
        )
        self._velocity_lot_factor = float(tier3.get("velocity_lot_factor", _TIER3_VELOCITY_LOT_FACTOR))
        self._velocity_tp_factor = float(tier3.get("velocity_tp_factor", _TIER3_VELOCITY_TP_FACTOR))

    @classmethod
    def from_json_path(cls, path: str | Path) -> CSPABayesEngine:
        """JSON 設定ファイルからエンジンを構築する。"""
        with Path(path).open(encoding="utf-8") as fh:
            return cls(json.load(fh))

    @staticmethod
    def _load_regime_defaults(
        raw: dict[str, Any] | None,
    ) -> dict[tuple[SessionType, AtrRegime], RegimeDefaults]:
        if not raw:
            return dict(_DEFAULT_REGIME_DEFAULTS)
        out: dict[tuple[SessionType, AtrRegime], RegimeDefaults] = {}
        for key, val in raw.items():
            session_str, regime_str = key.split("|", 1)
            session = _normalize_session(session_str)
            regime = _normalize_atr_regime(regime_str)
            if regime is None:
                continue
            out[(session, regime)] = {
                "lot_multiplier": float(val.get("lot_multiplier", 1.0)),
                "tp_multiplier": float(val.get("tp_multiplier", 1.0)),
            }
        return out or dict(_DEFAULT_REGIME_DEFAULTS)

    def tier1_segment_stats(self, rf: float, ra: float) -> tuple[float, float, int, int]:
        """decile ビンに対応する (win_rate, avg_r, rf_bin, ra_bin) を返す。"""
        rf_bin = _decile_bin(rf, self._tier1["rf_edges"])
        ra_bin = _decile_bin(ra, self._tier1["ra_edges"])
        wr = self._tier1["wr_matrix"][rf_bin][ra_bin]
        avg_r = self._tier1["avg_r_matrix"][rf_bin][ra_bin]
        return wr, avg_r, rf_bin, ra_bin

    def _tier1_segment_stats(self, rf: float, ra: float) -> tuple[float, float, int, int]:
        return self.tier1_segment_stats(rf, ra)

    def _tier1_reject(self, rf: float, ra: float) -> tuple[bool, str]:
        """
        Tier 1 Primary Gate。

        勝率 < min_win_rate または avg_R < min_avg_r のセグメントは REJECT。
        """
        wr, avg_r, rf_bin, ra_bin = self._tier1_segment_stats(rf, ra)
        if wr < self._tier1["min_win_rate"] or avg_r < self._tier1["min_avg_r"]:
            return True, (
                f"REJECTED_BY_TIER1: rf_bin={rf_bin}, ra_bin={ra_bin}, "
                f"wr={wr:.3f}, avg_r={avg_r:.4f}"
            )
        return False, ""

    def _resolve_atr_regime(self, features: dict[str, Any]) -> AtrRegime:
        """``current_atr_h1_regime`` または ``current_atr_h1`` から ATR レジームを決定。"""
        explicit = features.get("current_atr_h1_regime")
        if explicit is not None:
            parsed = _normalize_atr_regime(str(explicit))
            if parsed is not None:
                return parsed
        atr = float(features.get("current_atr_h1", 0.0))
        lo, hi = self._atr_tertiles
        if atr <= lo:
            return "Low-Vol"
        if atr <= hi:
            return "Mid-Vol"
        return "High-Vol"

    def _tier2_base_multipliers(
        self, session: SessionType, atr_regime: AtrRegime
    ) -> RegimeDefaults:
        """Tier 2: レジーム別ベース lot/tp 倍率。"""
        return self._regime_defaults.get(
            (session, atr_regime),
            {"lot_multiplier": 1.0, "tp_multiplier": 1.0},
        )

    def _is_high(self, feature: dict[str, Any], key: str) -> bool:
        """上位パーセンタイル（High）突破判定。"""
        if key not in self._percentile_high:
            return False
        value = feature.get(key)
        if value is None:
            return False
        return float(value) >= self._percentile_high[key]

    def _tier3_adjust(
        self,
        features: dict[str, Any],
        lot: float,
        tp: float,
    ) -> tuple[float, float, str]:
        """
        Tier 3 Dynamic Multiplier。

        rhythm + market_breath High → lot ×1.5
        breakout_velocity High → lot ×0.8, tp ×2.0
        """
        rhythm_breath_boost = (
            self._is_high(features, "rhythm_score")
            and self._is_high(features, "market_breath_score")
        )
        velocity_boost = self._is_high(features, "breakout_velocity")

        if rhythm_breath_boost:
            lot *= self._rhythm_breath_lot_boost
        if velocity_boost:
            lot *= self._velocity_lot_factor
            tp *= self._velocity_tp_factor

        if rhythm_breath_boost and velocity_boost:
            reason = "ALLOWED_TIER3_COMBINED"
        elif rhythm_breath_boost:
            reason = "ALLOWED_TIER3_BOOSTED"
        elif velocity_boost:
            reason = "ALLOWED_TIER3_VELOCITY"
        else:
            reason = "ALLOWED_BASE"

        return lot, tp, reason

    def evaluate_trade(self, feature_dict: dict[str, Any]) -> TradeEvaluation:
        """
        特徴量辞書から執行判定と lot/tp 倍率を返す。

        Parameters
        ----------
        feature_dict
            MT5 から送られる特徴量。最低限
            ``reaccel_follow_through``, ``reacceleration_score`` が必要。
            任意: ``session_type``, ``current_atr_h1`` / ``current_atr_h1_regime``,
            ``rhythm_score``, ``market_breath_score``, ``breakout_velocity``。

        Returns
        -------
        TradeEvaluation
            decision, reason, lot_multiplier, tp_multiplier
        """
        rf = float(feature_dict.get("reaccel_follow_through", 0.0))
        ra = float(feature_dict.get("reacceleration_score", 0.0))

        rejected, reject_reason = self._tier1_reject(rf, ra)
        if rejected:
            return {
                "decision": "REJECT",
                "reason": reject_reason,
                "lot_multiplier": 0.0,
                "tp_multiplier": 0.0,
            }

        session = _normalize_session(str(feature_dict.get("session_type", "ASIA")))
        atr_regime = self._resolve_atr_regime(feature_dict)
        base = self._tier2_base_multipliers(session, atr_regime)

        lot = base["lot_multiplier"]
        tp = base["tp_multiplier"]
        lot, tp, reason = self._tier3_adjust(feature_dict, lot, tp)

        return {
            "decision": "ALLOW",
            "reason": reason,
            "lot_multiplier": round(lot, 4),
            "tp_multiplier": round(tp, 4),
        }


def _demo() -> None:
    """モックデータによる動作確認。"""
    engine = CSPABayesEngine()

    cases: list[tuple[str, dict[str, Any]]] = [
        (
            "Tier1 REJECT (低 decile セグメント)",
            {
                "reaccel_follow_through": -0.001,
                "reacceleration_score": 0.25,
                "session_type": "ASIA",
                "current_atr_h1": 0.0015,
            },
        ),
        (
            "Tier1 ALLOW + Tier3 rhythm/breath boost",
            {
                "reaccel_follow_through": 0.0002,
                "reacceleration_score": 0.95,
                "session_type": "ASIA",
                "current_atr_h1": 0.0015,
                "rhythm_score": 0.90,
                "market_breath_score": 50.0,
            },
        ),
        (
            "Tier1 ALLOW + Tier3 breakout velocity (NY High-Vol)",
            {
                "reaccel_follow_through": 0.0002,
                "reacceleration_score": 0.95,
                "session_type": "NY",
                "current_atr_h1_regime": "High-Vol",
                "breakout_velocity": 1.5,
            },
        ),
        (
            "Tier1 ALLOW base (elite decile, no Tier3 catalyst)",
            {
                "reaccel_follow_through": 0.0002,
                "reacceleration_score": 0.95,
                "session_type": "LONDON",
                "current_atr_h1": 0.0014,
            },
        ),
    ]

    print("=== CSPABayesEngine demo ===")
    for label, features in cases:
        result = engine.evaluate_trade(features)
        print(f"\n[{label}]")
        for k, v in result.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    _demo()
