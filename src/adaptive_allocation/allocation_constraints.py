"""Allocation constraint enforcement for PAAE."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AllocationConstraints:
    min_weight: float = 0.05
    max_weight: float = 0.50
    max_delta: float = 0.20
    rebalance_interval_days: int = 7
    drift_threshold: float = 0.10


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    positive = {k: max(0.0, float(v)) for k, v in weights.items()}
    total = sum(positive.values())
    if total <= 0:
        n = len(positive) or 1
        return {k: round(1.0 / n, 4) for k in positive}
    return {k: round(v / total, 4) for k, v in positive.items()}


def apply_delta_cap(
    current: dict[str, float],
    proposed: dict[str, float],
    *,
    max_delta: float,
) -> dict[str, float]:
    keys = set(current) | set(proposed)
    capped: dict[str, float] = {}
    for key in keys:
        cur = float(current.get(key, 0.0))
        prop = float(proposed.get(key, cur))
        delta = max(-max_delta, min(max_delta, prop - cur))
        capped[key] = round(cur + delta, 4)
    return capped


def apply_bounds(
    weights: dict[str, float],
    *,
    min_weight: float,
    max_weight: float,
    disabled: set[str] | None = None,
) -> dict[str, float]:
    disabled = disabled or set()
    bounded: dict[str, float] = {}
    for key, val in weights.items():
        if key in disabled:
            bounded[key] = 0.0
            continue
        bounded[key] = round(max(min_weight, min(max_weight, float(val))), 4)
    return bounded


def apply_bounds_and_normalize(
    weights: dict[str, float],
    *,
    min_weight: float,
    max_weight: float,
    disabled: set[str] | None = None,
) -> dict[str, float]:
    """Clip weights to bounds while keeping the active set summing to 1.0."""
    disabled = disabled or set()
    keys = [k for k in weights if k not in disabled]
    if not keys:
        return normalize_weights(weights)

    work = normalize_weights({k: max(0.0, float(weights.get(k, 0.0))) for k in keys})
    for _ in range(50):
        clipped: dict[str, float] = {}
        at_min: list[str] = []
        at_max: list[str] = []
        free: list[str] = []

        for key in keys:
            value = work[key]
            if value <= min_weight + 1e-9:
                clipped[key] = min_weight
                at_min.append(key)
            elif value >= max_weight - 1e-9:
                clipped[key] = max_weight
                at_max.append(key)
            else:
                clipped[key] = round(value, 4)
                free.append(key)

        total = sum(clipped.values())
        if abs(total - 1.0) <= 1e-4:
            for key in disabled:
                clipped.setdefault(key, 0.0)
            return clipped

        fixed = set(at_min) | set(at_max)
        remaining = 1.0 - sum(clipped[k] for k in fixed)
        adjustable = [k for k in free if k not in fixed] or [k for k in keys if k not in fixed]

        if remaining <= 0 or not adjustable:
            result = normalize_weights(clipped)
            for key in disabled:
                result[key] = 0.0
            return result

        adj_total = sum(clipped[k] for k in adjustable)
        if adj_total <= 0:
            share = remaining / len(adjustable)
            for key in adjustable:
                clipped[key] = round(share, 4)
        else:
            for key in adjustable:
                clipped[key] = round(clipped[key] / adj_total * remaining, 4)

        work = clipped

    result = normalize_weights(work)
    for key in disabled:
        result[key] = 0.0
    return result


def enforce_constraints(
    current: dict[str, float],
    proposed: dict[str, float],
    constraints: AllocationConstraints | None = None,
    *,
    disabled: set[str] | None = None,
) -> dict[str, float]:
    c = constraints or AllocationConstraints()
    work = apply_delta_cap(current, proposed, max_delta=c.max_delta)
    active = {k: v for k, v in work.items() if k not in (disabled or set()) and v > 0}
    if not active:
        return normalize_weights(current)
    return apply_bounds_and_normalize(
        work,
        min_weight=c.min_weight,
        max_weight=c.max_weight,
        disabled=disabled,
    )


def validate_weights(weights: dict[str, float], constraints: AllocationConstraints | None = None) -> list[str]:
    c = constraints or AllocationConstraints()
    errors: list[str] = []
    total = sum(weights.values())
    if abs(total - 1.0) > 0.02:
        errors.append(f"weights sum to {total:.4f}, expected 1.0")
    for key, val in weights.items():
        if val <= 0:
            continue
        if val < c.min_weight - 1e-6:
            errors.append(f"{key} below min weight {c.min_weight}")
        if val > c.max_weight + 1e-6:
            errors.append(f"{key} above max weight {c.max_weight}")
    return errors
