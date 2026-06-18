"""Portfolio state consistency checks for RC1."""
from __future__ import annotations

from typing import Any


class PortfolioConsistencyChecker:
    def evaluate(
        self,
        *,
        profile_id: str,
        ai_cio_report: dict[str, Any] | None,
        cil_report: dict[str, Any] | None,
        chain_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        issues: list[str] = []
        ctx = chain_context or {}

        if not profile_id:
            issues.append("Invalid profile reference: missing profile_id")

        cio_profile = (ai_cio_report or {}).get("profile_id")
        cil_profile = (cil_report or {}).get("profile_id")
        if cio_profile and cil_profile and cio_profile != cil_profile:
            issues.append(f"Profile mismatch: AI CIO ({cio_profile}) vs CIL ({cil_profile})")

        strategies: set[str] = set()
        for rec in (ai_cio_report or {}).get("recommendations") or []:
            strat = rec.get("strategy")
            if strat:
                strategies.add(str(strat))
        for item in (cil_report or {}).get("top_opportunities") or []:
            strat = item.get("strategy")
            if strat and strat != "PORTFOLIO":
                strategies.add(str(strat))

        paae = ctx.get("paae") or {}
        weights = paae.get("current_weights") or paae.get("strategy_allocations") or {}
        for strat in strategies:
            if strat and strat not in weights and strat != "PORTFOLIO":
                issues.append(f"Broken strategy reference: {strat} not in portfolio weights")

        score = max(0.0, 100.0 - len(issues) * 20)
        return {
            "portfolio_consistency": round(score, 2),
            "issues": issues,
            "healthy": not issues,
            "strategy_references": sorted(strategies),
        }
