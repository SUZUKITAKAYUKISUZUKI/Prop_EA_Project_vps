"""End-to-end validation across Portfolio OS chain."""
from __future__ import annotations

from typing import Any

from src.production_hardening.config import CHAIN_LAYERS


class EndToEndValidator:
    def evaluate(self, *, chain_context: dict[str, Any]) -> dict[str, Any]:
        issues: list[str] = []
        layers_present: dict[str, bool] = {}
        missing_data: list[str] = []
        stale_cache: list[str] = []
        orphan_recs: list[str] = []

        layer_map = {
            "prae": chain_context.get("prae_report"),
            "paae": chain_context.get("paae_report"),
            "pdts": chain_context.get("pdts_report"),
            "slm": chain_context.get("slm_report"),
            "age": chain_context.get("age_report"),
            "cace": chain_context.get("cace_report"),
            "mie": chain_context.get("mie_report"),
            "apm": chain_context.get("apm_report"),
            "cil": chain_context.get("cil_report"),
            "ai_cio": chain_context.get("ai_cio_report"),
            "orl": chain_context.get("orl_report"),
        }

        for layer in CHAIN_LAYERS:
            present = bool(layer_map.get(layer))
            layers_present[layer] = present
            if not present:
                missing_data.append(f"Missing {layer.upper()} data")

        cil = chain_context.get("cil_report") or {}
        layer_flags = cil.get("layers") or {}
        for layer, flag in layer_flags.items():
            if flag is False:
                stale_cache.append(f"CIL reports absent upstream layer: {layer}")

        ai_cio = chain_context.get("ai_cio_report") or {}
        for rec in ai_cio.get("recommendations") or []:
            if rec.get("category") == "STRATEGY" and not rec.get("strategy"):
                orphan_recs.append("Strategy recommendation without strategy reference")

        if not chain_context.get("profile_id"):
            issues.append("Invalid profile reference in chain context")

        circular = self._detect_circular_deps(chain_context)
        issues.extend(missing_data)
        issues.extend(stale_cache)
        issues.extend(orphan_recs)
        issues.extend(circular)

        present_count = sum(1 for v in layers_present.values() if v)
        score = round((present_count / len(CHAIN_LAYERS)) * 100, 2)
        if issues:
            score = max(0.0, score - len(issues) * 5)

        return {
            "end_to_end_score": score,
            "end_to_end_validation": score >= 85 and not missing_data,
            "layers_present": layers_present,
            "missing_data": missing_data,
            "stale_cache": stale_cache,
            "orphan_recommendations": orphan_recs,
            "circular_dependencies": circular,
            "issues": issues,
            "healthy": score >= 85 and not missing_data,
        }

    def _detect_circular_deps(self, ctx: dict[str, Any]) -> list[str]:
        issues: list[str] = []
        apm = ctx.get("apm_report") or {}
        ai_cio = ctx.get("ai_cio_report") or {}
        apm_action = str((apm.get("recommendations") or {}).get("recommended_action") or "").lower()
        cio_actions = [str(r.get("action") or "").lower() for r in ai_cio.get("recommendations") or []]
        if apm_action and apm_action in cio_actions and len(cio_actions) == 1:
            issues.append("Circular dependency: APM and AI CIO mirror identical action")
        return issues
