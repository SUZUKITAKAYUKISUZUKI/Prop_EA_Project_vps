"""Write adaptive weights to Profile Manager."""
from __future__ import annotations

from typing import Any

from src.database.profile_migrations import DASHBOARD_STRATEGY_CODES
from src.repositories.profile_repository import ProfileRepository
from src.services.profile_service import ProfileService


class ProfileWeightWriter:
    def __init__(
        self,
        repo: ProfileRepository | None = None,
        *,
        profile_service: ProfileService | None = None,
        owns_connection: bool = False,
    ) -> None:
        self._owns = owns_connection or repo is None
        self._repo = repo or ProfileRepository(owns_connection=self._owns)
        self._profiles = profile_service or ProfileService(self._repo)

    def close(self) -> None:
        if self._profiles and self._owns:
            self._profiles.close()
        elif self._owns:
            self._repo.close()

    def apply_weights(
        self,
        profile_id: str,
        weights: dict[str, float],
        *,
        activate: bool = True,
    ) -> dict[str, Any]:
        record = self._repo.get_profile(profile_id)
        if record is None:
            raise KeyError(f"Unknown profile: {profile_id}")

        strategy_enabled = dict(record.get("strategy_enabled") or {})
        normalized: dict[str, float] = {}
        for code in DASHBOARD_STRATEGY_CODES:
            w = float(weights.get(code, 0.0))
            normalized[code] = w
            strategy_enabled[code] = w > 0.0

        total = sum(normalized.values()) or 1.0
        normalized = {k: round(v / total, 4) for k, v in normalized.items()}

        payload = dict(record)
        payload["strategy_allocations"] = normalized
        payload["strategy_enabled"] = strategy_enabled
        payload["allocations"] = normalized
        saved = self._repo.save_profile(profile_id, payload)

        if activate:
            self._profiles.apply_profile(profile_id)

        return {
            "profile_id": profile_id,
            "weights": normalized,
            "activated": activate,
            "record": saved,
        }
