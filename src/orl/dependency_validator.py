"""Module dependency validation for ORL v1."""
from __future__ import annotations

import importlib
from typing import Any


class DependencyValidator:
    REQUIRED_MODULES = (
        "src.ai_cio.engine",
        "src.cio_intelligence.engine",
        "src.apm_v2.engine",
        "src.orl.engine",
    )

    def evaluate(self) -> dict[str, Any]:
        issues: list[str] = []
        loaded = 0
        for module in self.REQUIRED_MODULES:
            try:
                importlib.import_module(module)
                loaded += 1
            except Exception as exc:
                issues.append(f"{module}: {exc}")

        score = round((loaded / len(self.REQUIRED_MODULES)) * 100, 2)
        return {
            "dependency_health": score,
            "modules_checked": len(self.REQUIRED_MODULES),
            "modules_loaded": loaded,
            "issues": issues,
            "healthy": score >= 85,
        }
