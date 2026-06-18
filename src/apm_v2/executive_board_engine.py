"""Executive board voting and consensus."""
from __future__ import annotations

from collections import Counter
from typing import Any


class ExecutiveBoardEngine:
    DIRECTOR_MODULES = {
        "RISK_DIRECTOR": ("PRAE", "AGE"),
        "GROWTH_DIRECTOR": ("PAAE", "PDTS"),
        "STRATEGY_DIRECTOR": ("SLM",),
        "CONFIDENCE_DIRECTOR": ("CACE",),
        "INTELLIGENCE_DIRECTOR": ("MIE",),
    }

    def evaluate(
        self,
        *,
        apm_v1_report: dict[str, Any],
        mie_report: dict[str, Any],
        cace_v17_report: dict[str, Any],
        upstream: dict[str, Any],
    ) -> dict[str, Any]:
        votes = [
            self._risk_director(upstream, apm_v1_report),
            self._growth_director(upstream, apm_v1_report),
            self._strategy_director(upstream, apm_v1_report),
            self._confidence_director(cace_v17_report, apm_v1_report),
            self._intelligence_director(mie_report, apm_v1_report),
        ]
        actions = [v["recommendation"] for v in votes]
        counts = Counter(actions)
        majority_action, agree_count = counts.most_common(1)[0]
        agreement_ratio = round(agree_count / len(votes), 3)
        avg_confidence = round(sum(v["confidence"] for v in votes) / len(votes), 2)
        board_consensus = round(agreement_ratio * avg_confidence, 2)

        return {
            "board_members": votes,
            "board_consensus": board_consensus,
            "agreement_ratio": agreement_ratio,
            "average_confidence": avg_confidence,
            "majority_recommendation": majority_action,
            "agree_count": agree_count,
            "total_directors": len(votes),
        }

    def _risk_director(self, upstream: dict[str, Any], apm: dict[str, Any]) -> dict[str, Any]:
        prae = upstream.get("prae_v2") or {}
        health = float((prae.get("health_report") or {}).get("health_score") or 70)
        risks = prae.get("strategy_risk") or []
        top_risk = max((float(r.get("risk_score") or 0) for r in risks), default=0)
        rec = "ENTER_RECOVERY" if health < 65 or top_risk > 80 else "NO_ACTION"
        return {
            "director": "RISK_DIRECTOR",
            "focus": "DD / Recovery",
            "recommendation": rec,
            "confidence": round(max(50.0, health - top_risk * 0.2), 2),
        }

    def _growth_director(self, upstream: dict[str, Any], apm: dict[str, Any]) -> dict[str, Any]:
        paae = upstream.get("paae") or {}
        pdts = upstream.get("pdts") or {}
        ranking = pdts.get("recommendation_ranking") or []
        score = float(ranking[0].get("score") or 82) if ranking else 82.0
        return {
            "director": "GROWTH_DIRECTOR",
            "focus": "Growth / R",
            "recommendation": "ALLOCATION_REBALANCE" if paae.get("recommended_weights") else "NO_ACTION",
            "confidence": score,
        }

    def _strategy_director(self, upstream: dict[str, Any], apm: dict[str, Any]) -> dict[str, Any]:
        slm = upstream.get("slm") or {}
        if slm.get("promotion_candidates"):
            return {
                "director": "STRATEGY_DIRECTOR",
                "focus": "Promotion / Retirement",
                "recommendation": "PROMOTE_STRATEGY",
                "confidence": float(slm["promotion_candidates"][0].get("score") or 80),
            }
        if slm.get("retirement_candidates"):
            return {
                "director": "STRATEGY_DIRECTOR",
                "focus": "Promotion / Retirement",
                "recommendation": "RETIRE_STRATEGY",
                "confidence": 75.0,
            }
        return {
            "director": "STRATEGY_DIRECTOR",
            "focus": "Promotion / Retirement",
            "recommendation": "NO_ACTION",
            "confidence": 70.0,
        }

    def _confidence_director(self, cace_v17: dict[str, Any], apm: dict[str, Any]) -> dict[str, Any]:
        conf = float(cace_v17.get("confidence") or apm.get("executive_score") or 70)
        rec = str(apm.get("recommendations", {}).get("recommended_action") or "NO_ACTION")
        return {
            "director": "CONFIDENCE_DIRECTOR",
            "focus": "Reliability",
            "recommendation": rec,
            "confidence": conf,
        }

    def _intelligence_director(self, mie: dict[str, Any], apm: dict[str, Any]) -> dict[str, Any]:
        strongest = (mie.get("strongest_module") or {}).get("module") or "AGE"
        trust = float((mie.get("strongest_module") or {}).get("trust_score") or 80)
        return {
            "director": "INTELLIGENCE_DIRECTOR",
            "focus": "Module Trust",
            "recommendation": apm.get("recommendations", {}).get("recommended_action") or "NO_ACTION",
            "confidence": trust,
        }
