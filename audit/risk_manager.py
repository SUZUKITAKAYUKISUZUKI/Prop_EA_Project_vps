"""
audit/risk_manager.py — 共通リスク管理・プロファイル・日次DDテーパリング

Fintokei 2大プロファイル (challenge / funded) と利益進捗連動型リスク縮小。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

PropFirmProfile = Literal["challenge", "funded"]

# --- 口座・リスク定数 ---
STARTING_EQUITY = 100_000.0
MAX_DAILY_DD_PCT = 5.0
MAX_MONTHLY_DD_PCT = 10.0
DAILY_DD_TAPER_MAX_PCT = 4.5  # 絶対安全防衛線 — ここで lot 乗数 → 0.0
LOT_FACTOR_FLOOR = 0.05       # 6連動乗算後の下限（0.0 完全停止は優先維持）
# 当日累積エクスポージャー上限（小数比率: 0.040 = 4.0%）。日次DD 5% の手前マージン。
MAX_DAILY_EXPOSURE_LIMIT_PCT = float(
    os.getenv("MAX_DAILY_EXPOSURE_LIMIT_PCT", "0.040"),
)
PIP_SIZE = 0.0001
PIP_VALUE_PER_LOT = 10.0

# --- 同ペア戦略間相互排他（L0/L2） ---
# daily: 同日先着1戦略 / concurrent: 実ポジション [entry, close) のみ遮断（L5 確定 close で動的解放）
MutualExclusionMode = Literal["daily", "concurrent", "off"]
REASON_SAME_DAY_PAIR_MUTUAL_EXCLUSION = "SAME_DAY_PAIR_MUTUAL_EXCLUSION"
REASON_CONCURRENT_PAIR_MUTUAL_EXCLUSION = "CONCURRENT_PAIR_MUTUAL_EXCLUSION"
DECISION_MUTUAL_EXCLUSION_LOCK = "MUTUAL_EXCLUSION_LOCK"
# 実執行とみなす trade_result（シャドー NOT_EXECUTED は除外）
EXECUTED_TRADE_RESULTS = frozenset({"WIN", "LOSS", "PENDING", "TIMEOUT"})
# ポートフォリオ全体のベースロット倍率（v3.8 デフォルト 1.2 — DD 余力活用）
DEFAULT_PORTFOLIO_LOT_MULTIPLIER = 1.2
PORTFOLIO_LOT_MULTIPLIER = float(
    os.getenv("PORTFOLIO_LOT_MULTIPLIER", str(DEFAULT_PORTFOLIO_LOT_MULTIPLIER))
)

# --- 利益クッション（+N% 防護壁まで lot 縮小 — 失格リスク低減）---
DEFAULT_PROFIT_CUSHION_TARGET_PCT = 2.0
DEFAULT_PROFIT_CUSHION_LOT_MULT = 0.85
REASON_PROFIT_CUSHION_BRAKE = "PROFIT_CUSHION_BRAKE"


def is_profit_cushion_enabled() -> bool:
    raw = os.getenv("PROFIT_CUSHION_ENABLED", "1").strip().lower()
    return raw not in ("0", "false", "off", "no", "disabled")


def profit_cushion_target_pct() -> float:
    return max(0.0, float(os.getenv("PROFIT_CUSHION_TARGET_PCT", str(DEFAULT_PROFIT_CUSHION_TARGET_PCT))))


def profit_cushion_lot_mult_below() -> float:
    return max(0.0, min(1.0, float(os.getenv("PROFIT_CUSHION_LOT_MULT", str(DEFAULT_PROFIT_CUSHION_LOT_MULT)))))


def is_profit_cushion_active_for_profile(profile: str) -> bool:
    if not is_profit_cushion_enabled():
        return False
    raw = os.getenv("PROFIT_CUSHION_PROFILES", "challenge").strip().lower()
    if raw in ("all", "*"):
        return True
    allowed = {p.strip() for p in raw.split(",") if p.strip()}
    return normalize_profile(profile) in allowed


def profit_cushion_target_equity(phase_start_equity: float) -> float:
    if phase_start_equity <= 0.0:
        return STARTING_EQUITY * (1.0 + profit_cushion_target_pct() / 100.0)
    return phase_start_equity * (1.0 + profit_cushion_target_pct() / 100.0)


def profit_cushion_lot_multiplier(
    phase_start_equity: float,
    current_equity: float,
    profile: str = "challenge",
) -> float:
    """
    フェーズ開始残高に対する +N% 防護壁（デフォルト +2%）を下回る間、
    lot_factor を一律縮小（デフォルト ×0.85）。到達後は ×1.0。
    """
    if not is_profit_cushion_active_for_profile(profile):
        return 1.0
    if current_equity >= profit_cushion_target_equity(phase_start_equity):
        return 1.0
    return profit_cushion_lot_mult_below()


def apply_profit_cushion_brake(
    lot_factor: float,
    equity: float,
    sl_distance: float,
    base_risk_pct: float,
    phase_start_equity: float,
    profile: str = "challenge",
) -> tuple[float, float, float, float]:
    """
    利益クッション未達時に lot_factor を縮小し RiskBudget / LotSize を再計算。

    Returns: lot_factor, risk_budget, lot_size, cushion_mult
    """
    cushion_mult = profit_cushion_lot_multiplier(phase_start_equity, equity, profile)
    if cushion_mult >= 1.0 or lot_factor <= 0.0:
        risk_budget = round(equity * base_risk_pct * lot_factor, 2)
        lot_size = lot_from_risk_budget(risk_budget, sl_distance, lot_factor)
        return lot_factor, risk_budget, lot_size, cushion_mult

    lot_factor = round(lot_factor * cushion_mult, 4)
    lot_factor = apply_lot_factor_floor(lot_factor)
    risk_budget = round(equity * base_risk_pct * lot_factor, 2)
    lot_size = lot_from_risk_budget(risk_budget, sl_distance, lot_factor)
    return lot_factor, risk_budget, lot_size, cushion_mult


def get_mutual_exclusion_mode() -> MutualExclusionMode:
    """MUTUAL_EXCLUSION_MODE=daily|concurrent|off（未設定時 concurrent）。"""
    mode = os.getenv("MUTUAL_EXCLUSION_MODE", "concurrent").strip().lower()
    if mode in ("off", "none", "disabled", "0", "false"):
        return "off"
    if mode == "daily":
        return "daily"
    return "concurrent"


def is_mutual_exclusion_enabled() -> bool:
    return get_mutual_exclusion_mode() != "off"


def mutual_exclusion_reason_tag() -> str:
    if get_mutual_exclusion_mode() == "concurrent":
        return REASON_CONCURRENT_PAIR_MUTUAL_EXCLUSION
    return REASON_SAME_DAY_PAIR_MUTUAL_EXCLUSION


def portfolio_lot_multiplier() -> float:
    raw = os.getenv(
        "PORTFOLIO_LOT_MULTIPLIER",
        str(DEFAULT_PORTFOLIO_LOT_MULTIPLIER),
    )
    return max(0.0, float(raw))


def apply_portfolio_lot_multiplier(lot_factor: float) -> float:
    mult = portfolio_lot_multiplier()
    if mult == 1.0:
        return lot_factor
    return round(lot_factor * mult, 4)


@dataclass
class OpenPosition:
    """同時保有排他: 実執行ポジションの [entry, close) 区間。"""

    pair: str
    setup_type: str
    entry_ts: Any
    close_ts: Any

# --- プロファイル別ベースリスク ---
CHALLENGE_PROFIT_TARGET_PCT = 8.0
# Challenge 初期ベースリスク（+0% 付近）。v3.4 オフェンス型 LLM: 2.5% 据置 + 確信度倍率で攻める
# 環境変数 CHALLENGE_BASE_RISK_PCT_MAX でシナリオ検証も可能（例: 0.030）
CHALLENGE_BASE_RISK_PCT_MAX = float(os.getenv("CHALLENGE_BASE_RISK_PCT_MAX", "0.025"))
CHALLENGE_BASE_RISK_PCT_MIN = 0.005  # +8% 付近
FUNDED_BASE_RISK_PCT = 0.010        # Funded 固定 1.0%

PROFILE_L2_MIN_SCORE: dict[str, int] = {"challenge": 30, "funded": 30}  # v3.4 仕様変更テスト: Challenge も 30
PROFILE_BAYES_ALLOW: dict[str, float] = {"challenge": 0.48, "funded": 0.55}
PROFILE_LLM_CAUTION_MULT: dict[str, float] = {"challenge": 0.5, "funded": 0.25}

# --- v3.4 オフェンス型 LLM 確信度 → 動的ロット倍率（L0/L2 アクセル機構）---
# 一律ベースリスク引上げは MaxDD 8.5% 失格ラインに抵触するため、
# 高確信度（≥85）のみ 1.4x、低確信度（40–59）は 0.6x で生存優先する。
LLM_CONFIDENCE_REJECT_BELOW = 40
LLM_CONFIDENCE_HIGH_MIN = 85
LLM_CONFIDENCE_NORMAL_MIN = 60
LLM_CONFIDENCE_LOW_MIN = 40
CONFIDENCE_LOT_MULT_HIGH = 1.4
CONFIDENCE_LOT_MULT_NORMAL = 1.0
CONFIDENCE_LOT_MULT_LOW = 0.6


def confidence_lot_multiplier(confidence_score: int) -> float:
    """LLM confidence_score に応じたロット倍率。40 未満は 0.0（エントリー拒否）。"""
    score = max(0, min(100, int(confidence_score)))
    if score >= LLM_CONFIDENCE_HIGH_MIN:
        return CONFIDENCE_LOT_MULT_HIGH
    if score >= LLM_CONFIDENCE_NORMAL_MIN:
        return CONFIDENCE_LOT_MULT_NORMAL
    if score >= LLM_CONFIDENCE_LOW_MIN:
        return CONFIDENCE_LOT_MULT_LOW
    return 0.0


def confidence_to_llm_decision(confidence_score: int) -> str:
    """確信度帯 → 意思決定ラベル（L4 出力と L0 lot_multiplier の同期用）。"""
    score = max(0, min(100, int(confidence_score)))
    if score < LLM_CONFIDENCE_REJECT_BELOW:
        return "REJECT_BY_LLM"
    if score < LLM_CONFIDENCE_NORMAL_MIN:
        return "CAUTION"
    return "ALLOW"


def apply_confidence_lot_scaling_with_mult(
    confidence_mult: float,
    lot_factor: float,
    equity: float,
    sl_distance: float,
    base_risk_pct: float,
) -> tuple[float, float, float, float]:
    """L4.5: 明示的 confidence 倍率を lot_factor に適用。"""
    if confidence_mult <= 0.0:
        return 0.0, 0.0, 0.0, 0.0

    lot_factor = round(lot_factor * confidence_mult, 4)
    lot_factor = apply_lot_factor_floor(lot_factor)
    risk_budget = round(equity * base_risk_pct * lot_factor, 2)
    lot_size = lot_from_risk_budget(risk_budget, sl_distance, lot_factor)
    return lot_factor, risk_budget, lot_size, confidence_mult


def apply_confidence_lot_scaling(
    confidence_score: int,
    lot_factor: float,
    equity: float,
    sl_distance: float,
    base_risk_pct: float,
) -> tuple[float, float, float, float]:
    """
    L4.5 六連動乗算後に confidence 倍率を適用し RiskBudget / LotSize を再計算。

    Returns: lot_factor, risk_budget, lot_size, confidence_mult
    confidence_mult=0 の場合は lot_factor/risk/lot すべて 0（REJECT_BY_LLM）。
    """
    return apply_confidence_lot_scaling_with_mult(
        confidence_lot_multiplier(confidence_score),
        lot_factor,
        equity,
        sl_distance,
        base_risk_pct,
    )


def normalize_profile(profile: str) -> PropFirmProfile:
    key = profile.strip().lower()
    if key not in PROFILE_L2_MIN_SCORE:
        raise ValueError(f"Invalid profile '{profile}'. Use 'challenge' or 'funded'.")
    return key  # type: ignore[return-value]


def challenge_profit_progress_pct(phase_start_equity: float, current_equity: float) -> float:
    """チャレンジ開始エクイティに対する累積利益率 (%)。"""
    if phase_start_equity <= 0:
        return 0.0
    return max(0.0, (current_equity - phase_start_equity) / phase_start_equity * 100.0)


def pure_bt_flat_base_risk_pct(profile: str) -> float:
    """Pure BT: 利益進捗テーパーを使わない固定ベースリスク。"""
    prof = normalize_profile(profile)
    if prof == "funded":
        return FUNDED_BASE_RISK_PCT
    return CHALLENGE_BASE_RISK_PCT_MAX


def effective_base_risk_pct(
    profile: str,
    phase_start_equity: float,
    current_equity: float,
) -> float:
    """
    プロファイル別ベースリスク (%)。

    challenge: 利益 +8% 接近に伴い CHALLENGE_BASE_RISK_PCT_MAX → 0.5% へ線形テーパー
    funded: 一律 1.0% 固定
    """
    prof = normalize_profile(profile)
    if prof == "funded":
        return FUNDED_BASE_RISK_PCT

    profit_pct = challenge_profit_progress_pct(phase_start_equity, current_equity)
    if profit_pct >= CHALLENGE_PROFIT_TARGET_PCT:
        return CHALLENGE_BASE_RISK_PCT_MIN

    ratio = profit_pct / CHALLENGE_PROFIT_TARGET_PCT
    span = CHALLENGE_BASE_RISK_PCT_MAX - CHALLENGE_BASE_RISK_PCT_MIN
    risk = CHALLENGE_BASE_RISK_PCT_MAX - span * ratio
    return max(CHALLENGE_BASE_RISK_PCT_MIN, min(CHALLENGE_BASE_RISK_PCT_MAX, risk))


def multiplier_daily_dd(current_daily_loss_pct: float) -> float:
    """
    当日累積損失が 0% 超から 4.5% 安全線へ近づくにつれ lot 乗数を 1.0→0.0 へ線形縮小。
    """
    max_limit = DAILY_DD_TAPER_MAX_PCT
    if current_daily_loss_pct <= 0.0:
        return 1.0
    multiplier = (max_limit - current_daily_loss_pct) / max_limit
    return max(0.0, min(1.0, multiplier))


def apply_lot_factor_floor(lot_factor: float) -> float:
    """
    L4.5 6連動乗算後の中間値集積リスク対策。

    - lot_factor > 0 → 最低 LOT_FACTOR_FLOOR (0.05) を維持し発注不可を防止
    - lot_factor == 0 → 日次DDテーパー (m_daily=0) / L0 完全停止指示を優先し 0.0 を維持
    """
    if lot_factor <= 0.0:
        return 0.0
    return round(max(LOT_FACTOR_FLOOR, lot_factor), 4)


def compute_trade_risk_pct(base_risk_pct: float, lot_factor: float) -> float:
    """L4.5 確定後のエクスポージャー比率（小数: 0.025 × 0.5 = 0.0125）。"""
    if lot_factor <= 0.0 or base_risk_pct <= 0.0:
        return 0.0
    return round(base_risk_pct * lot_factor, 6)


def cap_lot_factor_to_daily_exposure(
    lot_factor: float,
    base_risk_pct: float,
    daily_committed_risk_pct: float,
) -> tuple[float, bool]:
    """
    当日残りエクスポージャー上限内に lot_factor を収める。
    完全拒否ではなく縮小執行する（L0 遮断の代替）。
    """
    if lot_factor <= 0.0 or base_risk_pct <= 0.0:
        return lot_factor, False
    remaining = MAX_DAILY_EXPOSURE_LIMIT_PCT - daily_committed_risk_pct
    if remaining <= 0.0:
        return 0.0, True
    trade_risk = compute_trade_risk_pct(base_risk_pct, lot_factor)
    if trade_risk <= remaining:
        return lot_factor, False
    capped = round(remaining / base_risk_pct, 4)
    if capped <= 0.0:
        return 0.0, True
    return capped, True


def multiplier_candidate(score: float) -> float:
    if score >= 90:
        return 1.2
    if score >= 80:
        return 1.0
    if score >= 70:
        return 0.8
    return 0.5


def multiplier_dd(monthly_remaining: float) -> float:
    if monthly_remaining >= 8.0:
        return 1.0
    if monthly_remaining >= 5.0:
        return 0.5
    if monthly_remaining >= 3.0:
        return 0.25
    return 0.0


def multiplier_streak(losses: int) -> float:
    table = {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4}
    return table.get(losses, 0.0)


def multiplier_llm(decision: str, profile: str = "challenge") -> float:
    """Funded では CAUTION 時のロット倍率をより厳格に縮小。"""
    prof = normalize_profile(profile)
    if decision == "ALLOW":
        return 1.0
    if decision == "CAUTION":
        return PROFILE_LLM_CAUTION_MULT[prof]
    return 0.0


def multiplier_bayes(
    bayes_probability: float,
    bayes_allow_thres: float,
    bayes_reject_thres: float,
) -> float:
    if bayes_probability >= bayes_allow_thres:
        return 1.0
    if bayes_probability >= bayes_reject_thres:
        return 0.85
    return 1.0


def lot_from_risk_budget(
    risk_budget: float,
    sl_distance: float,
    lot_factor: float = 1.0,
    pip_size: float = PIP_SIZE,
    pip_value_per_lot: float = PIP_VALUE_PER_LOT,
) -> float:
    if sl_distance <= 0 or lot_factor <= 0 or risk_budget <= 0:
        return 0.0
    pips_at_risk = sl_distance / pip_size
    if pips_at_risk <= 0:
        return 0.0
    lot_size = risk_budget / (pips_at_risk * pip_value_per_lot)
    return round(lot_size, 4)


def calc_position_size(
    equity: float,
    candidate_score: float,
    monthly_dd_remaining: float,
    consecutive_losses: int,
    llm_decision: str,
    sl_distance: float,
    base_risk_pct: float,
    bayes_probability: float = 0.0,
    bayes_allow_thres: float = 0.55,
    bayes_reject_thres: float = 0.40,
    profile: str = "challenge",
    *,
    skip_portfolio_multiplier: bool = False,
    skip_defense_sizing: bool = False,
) -> tuple[float, float, float]:
    """RiskBudget, LotSize, lot_factor を返す。"""
    if skip_defense_sizing:
        lot_factor = 1.0
    else:
        prof = normalize_profile(profile)
        m_c = multiplier_candidate(candidate_score)
        m_d = multiplier_dd(monthly_dd_remaining)
        m_s = multiplier_streak(consecutive_losses)
        m_l = multiplier_llm(llm_decision, prof)
        m_b = multiplier_bayes(bayes_probability, bayes_allow_thres, bayes_reject_thres)

        lot_factor = round(m_c * m_d * m_s * m_l * m_b, 4)
        if not skip_portfolio_multiplier:
            lot_factor = apply_portfolio_lot_multiplier(lot_factor)
    lot_factor = apply_lot_factor_floor(lot_factor)
    risk_budget = round(equity * base_risk_pct * lot_factor, 2)
    lot_size = lot_from_risk_budget(risk_budget, sl_distance, lot_factor)
    return risk_budget, lot_size, lot_factor


def apply_daily_dd_brake(
    lot_factor: float,
    equity: float,
    sl_distance: float,
    base_risk_pct: float,
    daily_loss_pct: float = 0.0,
) -> tuple[float, float, float]:
    """日次DDテーパリング: multiplier_daily_dd で lot_factor を連続縮小。"""
    m_daily = multiplier_daily_dd(daily_loss_pct)
    lot_factor = round(lot_factor * m_daily, 4)
    lot_factor = apply_portfolio_lot_multiplier(lot_factor)
    lot_factor = apply_lot_factor_floor(lot_factor)
    risk_budget = round(equity * base_risk_pct * lot_factor, 2)
    lot_size = lot_from_risk_budget(risk_budget, sl_distance, lot_factor)
    return risk_budget, lot_size, lot_factor


def compute_l45_multipliers(
    candidate_score: float,
    monthly_dd_remaining: float,
    consecutive_losses: int,
    llm_decision: str,
    bayes_probability: float,
    bayes_allow_thres: float,
    bayes_reject_thres: float,
    profile: str = "challenge",
) -> dict[str, float]:
    prof = normalize_profile(profile)
    m_c = multiplier_candidate(candidate_score)
    m_d = multiplier_dd(monthly_dd_remaining)
    m_s = multiplier_streak(consecutive_losses)
    m_l = multiplier_llm(llm_decision, prof)
    m_b = multiplier_bayes(bayes_probability, bayes_allow_thres, bayes_reject_thres)
    lot_factor = round(m_c * m_d * m_s * m_l * m_b, 4)
    lot_factor = apply_lot_factor_floor(lot_factor)
    return {
        "m_candidate": m_c,
        "m_dd": m_d,
        "m_streak": m_s,
        "m_llm": m_l,
        "m_bayes": m_b,
        "lot_factor": lot_factor,
    }


@dataclass
class AccountState:
    """シミュレーション口座の動的状態。"""

    equity: float = STARTING_EQUITY
    equity_high_water_mark: float = field(default=STARTING_EQUITY)
    daily_start_equity: float = STARTING_EQUITY
    monthly_start_equity: float = STARTING_EQUITY
    profile: PropFirmProfile = "challenge"
    phase_start_equity: float = field(default=STARTING_EQUITY)
    current_day: Any = None
    current_month: Any = None
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    daily_consecutive_losses: int = 0
    recovery_boost_armed: bool = False
    trade_counter: int = 0
    last_event_timestamp: Any = None
    daily_committed_risk_pct: float = 0.0
    last_trade_date: Any = None
    # (calendar_date, pair) → その日そのペアで先着執行された setup_type（daily モード）
    daily_pair_executed_setup: dict[tuple[Any, str], str] = field(default_factory=dict)
    # concurrent モード: 実執行ポジションの保有区間
    open_positions: list[OpenPosition] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.profile = normalize_profile(self.profile)
        self.equity_high_water_mark = max(self.equity_high_water_mark, self.equity)

    def update_equity_high_water_mark(self) -> None:
        if self.equity > self.equity_high_water_mark:
            self.equity_high_water_mark = self.equity

    def current_drawdown_pct(self) -> float:
        """直近ピーク資産（HWM）からの Current_DD_%。"""
        peak = self.equity_high_water_mark
        if peak <= 0:
            return 0.0
        return max(0.0, (peak - self.equity) / peak * 100.0)

    def purge_closed_positions(self, ts) -> None:
        """entry <= ts < close を満たすポジションのみ残す。"""
        import pandas as pd

        now = pd.Timestamp(ts)
        self.open_positions = [
            pos
            for pos in self.open_positions
            if pd.Timestamp(pos.close_ts) > now
        ]

    def reset_daily_exposure_if_new_day(self, ts) -> None:
        """サーバー日付変更時に当日コミット済みリスクをゼロリセット。"""
        day = ts.date() if hasattr(ts, "date") else ts
        if self.last_trade_date != day:
            self.daily_committed_risk_pct = 0.0
            self.last_trade_date = day

    def _calendar_day(self, ts) -> Any:
        return ts.date() if hasattr(ts, "date") else ts

    def _pair_key(self, pair: str) -> str:
        return pair.strip().upper()

    def is_blocked_by_mutual_exclusion(
        self,
        ts,
        pair: str,
        setup_type: str,
    ) -> tuple[bool, str | None]:
        """
        daily: 同日・同ペアに別 setup_type の先着執行がある場合 True。
        concurrent: 同ペアで別 setup_type のアクティブポジションと区間重複時 True。
        off: 常に False（相互排他なし）。
        """
        if not is_mutual_exclusion_enabled():
            return False, None
        if get_mutual_exclusion_mode() == "daily":
            key = (self._calendar_day(ts), self._pair_key(pair))
            existing = self.daily_pair_executed_setup.get(key)
            if existing is None or existing == setup_type:
                return False, None
            return True, existing

        import pandas as pd

        self.purge_closed_positions(ts)
        pair_u = self._pair_key(pair)
        setup_u = setup_type.strip()
        now = pd.Timestamp(ts)
        for pos in self.open_positions:
            if pos.pair != pair_u or pos.setup_type == setup_u:
                continue
            entry = pd.Timestamp(pos.entry_ts)
            close = pd.Timestamp(pos.close_ts)
            if entry <= now < close:
                return True, pos.setup_type
        return False, None

    def register_mutual_exclusion_execution(self, ts, pair: str, setup_type: str) -> None:
        """daily モード: 当日・当ペアの支配戦略を記録（先着のみ）。"""
        key = (self._calendar_day(ts), self._pair_key(pair))
        if key not in self.daily_pair_executed_setup:
            self.daily_pair_executed_setup[key] = setup_type

    def register_open_position(
        self,
        ts,
        pair: str,
        setup_type: str,
        holding_minutes: int,
    ) -> None:
        """concurrent モード: 実執行ポジションの [entry, close) を登録。"""
        import pandas as pd

        entry = pd.Timestamp(ts)
        close = entry + pd.Timedelta(minutes=max(int(holding_minutes), 1))
        self.open_positions.append(
            OpenPosition(
                pair=self._pair_key(pair),
                setup_type=setup_type.strip(),
                entry_ts=entry,
                close_ts=close,
            )
        )

    def register_executed_position(
        self,
        entry_ts,
        pair: str,
        setup_type: str,
        holding_minutes: int,
    ) -> None:
        """
        L5 確定後: 実際のクローズ時刻（SL/TP/タイムアウト）で相互排他区間を登録。

        concurrent モードでは Phase-1 の固定暫定区間を使わず、
        close_ts 到達時に purge_closed_positions で即時解放される。
        """
        import pandas as pd

        entry = pd.Timestamp(entry_ts)
        pair_u = self._pair_key(pair)
        setup_u = setup_type.strip()
        close = entry + pd.Timedelta(minutes=max(int(holding_minutes), 1))
        self.open_positions = [
            pos
            for pos in self.open_positions
            if not (
                pos.pair == pair_u
                and pos.setup_type == setup_u
                and pd.Timestamp(pos.entry_ts) == entry
            )
        ]
        self.open_positions.append(
            OpenPosition(
                pair=pair_u,
                setup_type=setup_u,
                entry_ts=entry,
                close_ts=close,
            )
        )

    def register_provisional_open_position(
        self,
        ts,
        pair: str,
        setup_type: str,
        max_holding_minutes: int,
    ) -> None:
        """後方互換: register_executed_position へ委譲（固定暫定区間は非推奨）。"""
        self.register_open_position(ts, pair, setup_type, max_holding_minutes)

    def refine_open_position_close(
        self,
        entry_ts,
        pair: str,
        setup_type: str,
        holding_minutes: int,
    ) -> None:
        """後方互換: L5 後の実クローズ時刻登録。"""
        self.register_executed_position(entry_ts, pair, setup_type, holding_minutes)

    def update_calendar(self, ts) -> None:
        import pandas as pd

        day = ts.date()
        month = (ts.year, ts.month)
        if self.current_day != day:
            self.current_day = day
            self.daily_start_equity = self.equity
            self.daily_consecutive_losses = 0
            if get_mutual_exclusion_mode() == "daily":
                self.daily_pair_executed_setup.clear()
            self.reset_daily_exposure_if_new_day(ts)
        self.purge_closed_positions(ts)
        if self.current_month != month:
            self.current_month = month
            self.monthly_start_equity = self.equity
        self.apply_consecutive_loss_cooldown(ts)

    def would_exceed_daily_exposure(self, additional_risk_pct: float) -> bool:
        """追加リスク込みで当日上限を超えるか（L0 遮断判定）。"""
        if additional_risk_pct <= 0.0:
            return self.daily_committed_risk_pct >= MAX_DAILY_EXPOSURE_LIMIT_PCT
        return (
            self.daily_committed_risk_pct + additional_risk_pct
        ) > MAX_DAILY_EXPOSURE_LIMIT_PCT

    def commit_daily_risk(self, trade_risk_pct: float) -> None:
        """執行確定時: テーパリング後リスク比率を当日累積へ加算。"""
        if trade_risk_pct <= 0.0:
            return
        self.daily_committed_risk_pct = round(
            self.daily_committed_risk_pct + trade_risk_pct,
            6,
        )

    def apply_consecutive_loss_cooldown(self, ts) -> None:
        import pandas as pd

        if self.last_event_timestamp is None:
            return
        elapsed = ts - self.last_event_timestamp
        if elapsed >= pd.Timedelta(hours=24):
            self.consecutive_losses = 0

    def daily_dd_remaining(self) -> float:
        dd_used = max(0.0, (self.daily_start_equity - self.equity) / self.daily_start_equity * 100.0)
        return MAX_DAILY_DD_PCT - dd_used

    def daily_loss_fraction(self) -> float:
        if self.daily_start_equity <= 0:
            return 0.0
        return max(0.0, (self.daily_start_equity - self.equity) / self.daily_start_equity)

    def monthly_dd_remaining(self) -> float:
        dd_used = max(0.0, (self.monthly_start_equity - self.equity) / self.monthly_start_equity * 100.0)
        return MAX_MONTHLY_DD_PCT - dd_used

    def profit_progress_pct(self) -> float:
        return challenge_profit_progress_pct(self.phase_start_equity, self.equity)

    def resolved_base_risk_pct(self) -> float:
        return effective_base_risk_pct(self.profile, self.phase_start_equity, self.equity)

    def next_trade_id(self, ts) -> str:
        self.trade_counter += 1
        return f"TX_{ts.strftime('%Y%m%d')}_{self.trade_counter:03d}"


def is_executed_trade_result(trade_result: str) -> bool:
    """相互排他の「実執行」判定 — lot_factor>0 で執行されたトレードのみ対象。"""
    return str(trade_result).upper() in EXECUTED_TRADE_RESULTS


def check_mutual_exclusion_from_records(
    records: list[dict[str, Any]],
    signal_ts,
    pair: str,
    setup_type: str,
) -> tuple[bool, str | None]:
    """
    バックテスト records から相互排他を走査（AccountState と同じモード）。

    daily: 同日・同ペアの先着 setup_type。
    concurrent: シグナル時点で区間重複する別 setup_type の実執行。
    off: 常に False。
    """
    if not is_mutual_exclusion_enabled():
        return False, None

    import pandas as pd

    pair_u = pair.strip().upper()
    setup_u = setup_type.strip()
    signal = pd.Timestamp(signal_ts)

    if get_mutual_exclusion_mode() == "daily":
        day_str = signal.strftime("%Y-%m-%d")
        for rec in records:
            ts_raw = str(rec.get("timestamp", ""))
            if not ts_raw.startswith(day_str):
                continue
            if str(rec.get("pair", "")).upper() != pair_u:
                continue
            if not is_executed_trade_result(str(rec.get("trade_result", ""))):
                continue
            existing = str(rec.get("setup_type", ""))
            if existing and existing != setup_u:
                return True, existing
        return False, None

    for rec in records:
        if str(rec.get("pair", "")).upper() != pair_u:
            continue
        if not is_executed_trade_result(str(rec.get("trade_result", ""))):
            continue
        existing = str(rec.get("setup_type", ""))
        if not existing or existing == setup_u:
            continue
        entry = pd.Timestamp(rec.get("timestamp"))
        holding = int(rec.get("holding_time", 0) or 0)
        if holding <= 0:
            continue
        close = entry + pd.Timedelta(minutes=holding)
        if entry <= signal < close:
            return True, existing
    return False, None
