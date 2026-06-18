"""State transition probability from account_state_history."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.state_analytics.state_history_repository import StateHistoryRepository

NORMALIZED_STATES = ("challenge", "funded", "recovery", "live", "unknown")


class TransitionProbabilityEngine:
    def __init__(self, *, history: StateHistoryRepository | None = None, owns_connection: bool = False) -> None:
        self._history = history or StateHistoryRepository(owns_connection=owns_connection)
        self._owns = owns_connection or history is None
        self._matrix: dict[str, dict[str, float]] | None = None

    def close(self) -> None:
        if self._owns:
            self._history.close()

    def build_matrix(self, *, limit: int = 5000) -> dict[str, dict[str, float]]:
        rows = self._history.list_history(limit=limit)
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for i in range(len(rows) - 1):
            a = _norm_state(rows[i].get("state"))
            b = _norm_state(rows[i + 1].get("state"))
            counts[a][b] += 1

        matrix: dict[str, dict[str, float]] = {}
        for src, dests in counts.items():
            total = sum(dests.values()) or 1
            matrix[src] = {dst: round(c / total, 4) for dst, c in dests.items()}
        self._matrix = matrix
        return matrix

    def probability(self, from_state: str, to_state: str) -> float:
        matrix = self._matrix or self.build_matrix()
        src = _norm_state(from_state)
        dst = _norm_state(to_state)
        return float(matrix.get(src, {}).get(dst, _default_probability(src, dst)))

    def transition_probability(self, from_state: str, to_state: str) -> float:
        return self.probability(from_state, to_state)

    def most_likely_next(self, from_state: str) -> tuple[str, float]:
        matrix = self._matrix or self.build_matrix()
        src = _norm_state(from_state)
        dests = matrix.get(src) or _fallback_dests(src)
        if not dests:
            return src, 1.0
        best = max(dests.items(), key=lambda x: x[1])
        return best[0], best[1]

    def cumulative_transition(self, from_state: str, to_state: str, *, steps: int = 3) -> float:
        """Approximate multi-step transition probability."""
        if steps <= 1:
            return self.probability(from_state, to_state)
        matrix = self._matrix or self.build_matrix()
        src = _norm_state(from_state)
        dst = _norm_state(to_state)
        dist = {src: 1.0}
        for _ in range(steps):
            nxt: dict[str, float] = defaultdict(float)
            for state, prob in dist.items():
                for target, p in (matrix.get(state) or _fallback_dests(state)).items():
                    nxt[target] += prob * p
            dist = dict(nxt)
        return round(dist.get(dst, 0.0), 4)

    def to_dict(self) -> dict[str, Any]:
        matrix = self._matrix or self.build_matrix()
        return {"transition_probability": matrix}


def _norm_state(state: Any) -> str:
    s = str(state or "unknown").lower().strip()
    if s in NORMALIZED_STATES:
        return s
    if "challenge" in s:
        return "challenge"
    if "recover" in s:
        return "recovery"
    if "fund" in s:
        return "funded"
    if "live" in s:
        return "live"
    return "unknown"


def _default_probability(src: str, dst: str) -> float:
    if src == dst:
        return 0.85
    defaults = {
        ("funded", "recovery"): 0.08,
        ("recovery", "funded"): 0.35,
        ("challenge", "funded"): 0.25,
        ("funded", "live"): 0.05,
        ("live", "recovery"): 0.10,
    }
    return defaults.get((src, dst), 0.02)


def _fallback_dests(src: str) -> dict[str, float]:
    stay = 0.80
    out: dict[str, float] = {src: stay}
    remainder = 1.0 - stay
    if src == "funded":
        out["recovery"] = remainder * 0.6
        out["live"] = remainder * 0.4
    elif src == "recovery":
        out["funded"] = remainder
    elif src == "challenge":
        out["funded"] = remainder
    else:
        out[src] = 1.0
    return out
