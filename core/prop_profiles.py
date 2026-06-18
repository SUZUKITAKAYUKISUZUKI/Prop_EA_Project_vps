"""Load and resolve prop firm challenge profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES_PATH = PROJECT_ROOT / "configs" / "prop_profiles.json"


@dataclass(frozen=True)
class PropProfile:
    name: str
    target_profit: float
    daily_dd_limit: float
    total_dd_limit: float
    max_days: int
    starting_equity: float
    profile_key: str = "challenge"


def load_prop_profiles(path: Path | None = None) -> dict[str, PropProfile]:
    cfg_path = path or DEFAULT_PROFILES_PATH
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    out: dict[str, PropProfile] = {}
    for name, spec in raw.items():
        out[name] = PropProfile(
            name=name,
            target_profit=float(spec["target_profit"]),
            daily_dd_limit=float(spec["daily_dd_limit"]),
            total_dd_limit=float(spec["total_dd_limit"]),
            max_days=int(spec.get("max_days", 0)),
            starting_equity=float(spec.get("starting_equity", 100_000.0)),
            profile_key=str(spec.get("profile_key", "challenge")),
        )
    return out


def get_profile(name: str | None = None, path: Path | None = None) -> PropProfile:
    """Resolve prop profile from SQLite Profile Manager, falling back to JSON."""
    legacy_map = {
        "Fintokei_100K": "PROP_FINTOKEI",
        "FTMO_100K": "PROP_FTMO",
        "FundingPips_100K": "PROP_FTMO",
    }
    try:
        from src.services.profile_service import ProfileContext, ProfileService

        svc = ProfileService()
        try:
            if name is None:
                ctx = svc.load_active_profile()
            else:
                profile_id = legacy_map.get(name, name)
                ctx = ProfileContext.from_record(svc.get_profile(profile_id))
            return ctx.to_prop_profile()
        finally:
            svc.close()
    except Exception:
        profiles = load_prop_profiles(path)
        resolved = name or "Fintokei_100K"
        if resolved not in profiles:
            raise KeyError(
                f"Unknown prop profile '{resolved}'. Available: {', '.join(profiles)}"
            )
        return profiles[resolved]


def load_pfoo_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or (PROJECT_ROOT / "configs" / "pfoo_config.json")
    return json.loads(cfg_path.read_text(encoding="utf-8"))
