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


def get_profile(name: str, path: Path | None = None) -> PropProfile:
    profiles = load_prop_profiles(path)
    if name not in profiles:
        raise KeyError(f"Unknown prop profile '{name}'. Available: {', '.join(profiles)}")
    return profiles[name]


def load_pfoo_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or (PROJECT_ROOT / "configs" / "pfoo_config.json")
    return json.loads(cfg_path.read_text(encoding="utf-8"))
