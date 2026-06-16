"""PET Phase 5.2 — rank open positions for selective closure (MEDIUM mode)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RankedPosition:
    ticket: str
    setup_type: str
    bayes_probability: float
    expected_r: float
    risk_contribution: float
    priority_score: float


def priority_score(
    *,
    bayes_probability: float,
    expected_r: float,
    risk_contribution: float,
) -> float:
    risk = max(risk_contribution, 1e-9)
    return (max(bayes_probability, 0.0) * max(expected_r, 0.0)) / risk


def rank_positions(positions: list[dict]) -> list[RankedPosition]:
    ranked: list[RankedPosition] = []
    for pos in positions:
        bayes = float(pos.get("bayes_probability", pos.get("bayes", 0.5)) or 0.5)
        expected_r = float(pos.get("expected_r", pos.get("expected_R", 1.0)) or 1.0)
        risk = float(pos.get("risk_contribution", pos.get("risk_budget", pos.get("lot_size", 1.0))) or 1.0)
        ticket = str(pos.get("ticket", pos.get("position_ticket", pos.get("pair", ""))))
        setup = str(pos.get("setup_type", "UNKNOWN"))
        score = priority_score(
            bayes_probability=bayes,
            expected_r=expected_r,
            risk_contribution=risk,
        )
        ranked.append(
            RankedPosition(
                ticket=ticket,
                setup_type=setup,
                bayes_probability=bayes,
                expected_r=expected_r,
                risk_contribution=risk,
                priority_score=round(score, 6),
            )
        )
    return sorted(ranked, key=lambda item: item.priority_score)


def select_lowest_ranked(positions: list[dict], count: int = 1) -> list[str]:
    ranked = rank_positions(positions)
    if not ranked or count <= 0:
        return []
    return [item.ticket for item in ranked[:count]]
