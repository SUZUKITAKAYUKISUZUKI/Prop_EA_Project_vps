"""
strategies/ttm_kalman.py — TTM 用 1 次元カルマン価格トラッカー

方向予測ではなく市場状態推定エンジンとして利用する。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanState:
    """カルマン更新後の状態スナップショット。"""

    price: float
    slope: float
    velocity: float
    acceleration: float
    residual: float
    zscore: float


class KalmanPriceTracker:
    """
    ローカルレベル + トレンド（2 状態）カルマンフィルタ。

    状態 x = [level, velocity]^T
    観測 y = close
    """

    def __init__(
        self,
        *,
        process_level_var: float = 1e-6,
        process_velocity_var: float = 1e-8,
        measurement_var: float = 1e-4,
        residual_lookback: int = 20,
    ) -> None:
        self._q_level = float(process_level_var)
        self._q_velocity = float(process_velocity_var)
        self._r = float(measurement_var)
        self._residual_lookback = max(5, int(residual_lookback))
        self._level = 0.0
        self._velocity = 0.0
        self._p = np.eye(2, dtype=float) * 1.0
        self._prev_velocity = 0.0
        self._residuals: list[float] = []
        self._initialized = False

    def reset(self, price: float) -> KalmanState:
        """系列開始時に状態をリセット。"""
        self._level = float(price)
        self._velocity = 0.0
        self._prev_velocity = 0.0
        self._p = np.eye(2, dtype=float) * 1.0
        self._residuals = []
        self._initialized = True
        return self._snapshot(float(price))

    def update(self, price: float) -> KalmanState:
        """1 観測でカルマン更新。"""
        y = float(price)
        if not self._initialized:
            return self.reset(y)

        # 状態遷移: level += velocity, velocity 持続
        f = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)
        q = np.array(
            [[self._q_level, 0.0], [0.0, self._q_velocity]],
            dtype=float,
        )
        x = np.array([self._level, self._velocity], dtype=float)
        p = f @ self._p @ f.T + q

        h = np.array([1.0, 0.0], dtype=float)
        y_pred = float(h @ x)
        residual = y - y_pred
        s = float(h @ p @ h.T + self._r)
        k = (p @ h) / max(s, 1e-12)
        x_new = x + k * residual
        p_new = (np.eye(2) - np.outer(k, h)) @ p

        acceleration = float(x_new[1] - self._prev_velocity)
        self._level = float(x_new[0])
        self._velocity = float(x_new[1])
        self._prev_velocity = self._velocity
        self._p = p_new
        self._residuals.append(residual)
        if len(self._residuals) > self._residual_lookback:
            self._residuals.pop(0)

        return self._snapshot(y, residual=residual)

    def _snapshot(self, observed: float, *, residual: float | None = None) -> KalmanState:
        res = float(residual if residual is not None else observed - self._level)
        if len(self._residuals) >= 3:
            std = float(np.std(self._residuals, ddof=1))
            z = res / std if std > 1e-12 else 0.0
        else:
            z = 0.0
        return KalmanState(
            price=float(self._level),
            slope=float(self._velocity),
            velocity=float(self._velocity),
            acceleration=float(self._velocity - self._prev_velocity),
            residual=res,
            zscore=float(z),
        )
