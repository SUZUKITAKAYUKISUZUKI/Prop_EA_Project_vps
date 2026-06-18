"""Generate strategic decision branches from governance context."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.ai_governor.governor_context import GovernorContext
from src.profile_manager.resolver import resolve_profile_from_state


class DecisionScenarioGenerator:
    """Build candidate governance branches — never executes trades."""

    def generate(
        self,
        context: GovernorContext,
        *,
        age_v3_report: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        v3 = age_v3_report or {}
        branches: list[dict[str, Any]] = [
            {
                "branch_id": "A",
                "action_type": "DO_NOTHING",
                "action_label": "DO_NOTHING",
                "description": "Maintain current profile, allocation, and lifecycle.",
                "modification": {},
            }
        ]

        recovery_profile = resolve_profile_from_state("recovery")
        if context.current_profile != recovery_profile:
            branches.append(
                {
                    "branch_id": "B",
                    "action_type": "PROFILE_SWITCH",
                    "action_label": f"SWITCH_TO_{recovery_profile.upper()}",
                    "description": f"Switch operating profile to {recovery_profile}.",
                    "modification": {
                        "current_profile": recovery_profile,
                        "profile_id": recovery_profile,
                        "target_state": "recovery",
                    },
                }
            )

        for code in self._overweight_strategies(context):
            branches.append(
                {
                    "branch_id": f"C_{code}",
                    "action_type": "REDUCE_ALLOCATION",
                    "action_label": f"REDUCE_{code}",
                    "description": f"Reduce {code} allocation toward PAAE recommendation.",
                    "modification": {"allocation_delta": {code: -0.10}},
                }
            )

        for code in self._underweight_strategies(context):
            branches.append(
                {
                    "branch_id": f"D_{code}",
                    "action_type": "INCREASE_ALLOCATION",
                    "action_label": f"INCREASE_{code}",
                    "description": f"Increase {code} weight toward PAAE recommendation.",
                    "modification": {"allocation_delta": {code: 0.10}},
                }
            )

        if not any(b["action_type"] == "REDUCE_ALLOCATION" for b in branches):
            top_over = self._largest_overweight(context)
            if top_over:
                branches.append(
                    {
                        "branch_id": f"C_{top_over}",
                        "action_type": "REDUCE_ALLOCATION",
                        "action_label": f"REDUCE_{top_over}",
                        "description": f"Reduce {top_over} allocation toward PAAE recommendation.",
                        "modification": {"allocation_delta": {top_over: -0.08}},
                    }
                )

        if not any(b["action_type"] == "INCREASE_ALLOCATION" for b in branches):
            top_under = self._largest_underweight(context)
            if top_under:
                branches.append(
                    {
                        "branch_id": f"D_{top_under}",
                        "action_type": "INCREASE_ALLOCATION",
                        "action_label": f"INCREASE_{top_under}",
                        "description": f"Increase {top_under} weight toward PAAE recommendation.",
                        "modification": {"allocation_delta": {top_under: 0.08}},
                    }
                )

        for candidate in self._promotion_candidates(context):
            code = str(candidate.get("strategy") or "")
            if not code:
                continue
            branches.append(
                {
                    "branch_id": f"E_{code}",
                    "action_type": "PROMOTE_STRATEGY",
                    "action_label": f"PROMOTE_{code}",
                    "description": f"Promote {code} to next lifecycle stage.",
                    "modification": {"promote_strategy": code},
                }
            )

        funded_profile = resolve_profile_from_state("funded")
        if context.current_state == "recovery" and context.current_profile != funded_profile:
            branches.append(
                {
                    "branch_id": "F",
                    "action_type": "PROFILE_SWITCH",
                    "action_label": f"SWITCH_TO_{funded_profile.upper()}",
                    "description": f"Return to funded profile {funded_profile}.",
                    "modification": {
                        "current_profile": funded_profile,
                        "profile_id": funded_profile,
                        "target_state": "funded",
                    },
                }
            )

        lifecycle = v3.get("strategy_forecast") or {}
        for code, data in (lifecycle.get("strategies") or {}).items():
            flags = data.get("flags") or []
            if "DEMOTION_CANDIDATE" in flags and not any(b["action_label"] == f"REDUCE_{code}" for b in branches):
                branches.append(
                    {
                        "branch_id": f"G_{code}",
                        "action_type": "DEMOTE_STRATEGY",
                        "action_label": f"DEMOTE_{code}",
                        "description": f"Demote {code} based on projected lifecycle degradation.",
                        "modification": {"demote_strategy": code},
                    }
                )

        return self._dedupe_branches(branches)

    def apply_modification(self, context: GovernorContext, modification: dict[str, Any]) -> GovernorContext:
        ctx = deepcopy(context)
        if not modification:
            return ctx

        if modification.get("current_profile"):
            ctx.current_profile = str(modification["current_profile"])
            ctx.profile_id = str(modification.get("profile_id") or ctx.current_profile)
            ctx.profile = dict(ctx.profile)
            ctx.profile["profile_id"] = ctx.profile_id
        if modification.get("target_state"):
            ctx.current_state = str(modification["target_state"]).lower()
            ctx.source_state = ctx.current_state

        delta = modification.get("allocation_delta") or {}
        if delta:
            alloc = dict(ctx.current_allocation)
            for code, change in delta.items():
                alloc[code] = max(0.0, min(1.0, float(alloc.get(code, 0.0)) + float(change)))
            ctx.current_allocation = self._normalize_allocation(alloc)

        promote = modification.get("promote_strategy")
        if promote:
            stages = dict(ctx.strategy_stages)
            current = stages.get(promote, "INCUBATION").upper()
            next_stage = {"INCUBATION": "PRODUCTION", "PRODUCTION": "CORE", "RECOVERY": "PRODUCTION"}.get(
                current, current
            )
            stages[promote] = next_stage
            ctx.strategy_stages = stages
            scores = dict(ctx.strategy_scores)
            scores[promote] = min(100.0, float(scores.get(promote, 50.0)) + 8.0)
            ctx.strategy_scores = scores

        demote = modification.get("demote_strategy")
        if demote:
            stages = dict(ctx.strategy_stages)
            current = stages.get(demote, "PRODUCTION").upper()
            next_stage = {"CORE": "PRODUCTION", "PRODUCTION": "RECOVERY", "RECOVERY": "RETIRED"}.get(
                current, "RECOVERY"
            )
            stages[demote] = next_stage
            ctx.strategy_stages = stages
            scores = dict(ctx.strategy_scores)
            scores[demote] = max(0.0, float(scores.get(demote, 50.0)) - 10.0)
            ctx.strategy_scores = scores

        return ctx

    def _overweight_strategies(self, context: GovernorContext) -> list[str]:
        current = context.current_allocation
        recommended = context.recommended_allocation
        return [
            code
            for code in sorted(current)
            if float(current.get(code, 0)) > float(recommended.get(code, 0)) + 0.02
        ][:2]

    def _underweight_strategies(self, context: GovernorContext) -> list[str]:
        current = context.current_allocation
        recommended = context.recommended_allocation
        return [
            code
            for code in sorted(recommended)
            if float(recommended.get(code, 0)) > float(current.get(code, 0)) + 0.02
        ][:2]

    def _largest_overweight(self, context: GovernorContext) -> str | None:
        current = context.current_allocation
        recommended = context.recommended_allocation
        best: tuple[float, str] | None = None
        for code in current:
            delta = float(current.get(code, 0)) - float(recommended.get(code, 0))
            if delta > 0 and (best is None or delta > best[0]):
                best = (delta, code)
        return best[1] if best else None

    def _largest_underweight(self, context: GovernorContext) -> str | None:
        current = context.current_allocation
        recommended = context.recommended_allocation
        best: tuple[float, str] | None = None
        for code in recommended:
            delta = float(recommended.get(code, 0)) - float(current.get(code, 0))
            if delta > 0 and (best is None or delta > best[0]):
                best = (delta, code)
        return best[1] if best else None

    def _promotion_candidates(self, context: GovernorContext) -> list[dict[str, Any]]:
        slm = context.slm or {}
        candidates = list(slm.get("promotion_candidates") or [])
        if candidates:
            return candidates[:2]
        for row in slm.get("strategies") or []:
            stage = str(row.get("stage") or "").upper()
            score = float(row.get("score") or 0)
            if stage in {"INCUBATION", "PRODUCTION"} and score >= 75.0:
                candidates.append(row)
        return candidates[:2]

    def _normalize_allocation(self, alloc: dict[str, float]) -> dict[str, float]:
        total = sum(alloc.values())
        if total <= 0:
            return alloc
        return {code: round(weight / total, 4) for code, weight in alloc.items()}

    def _dedupe_branches(self, branches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for branch in branches:
            key = branch.get("action_label") or branch.get("branch_id")
            if key in seen:
                continue
            seen.add(str(key))
            unique.append(branch)
        return unique
