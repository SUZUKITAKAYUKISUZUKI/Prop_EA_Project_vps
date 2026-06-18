"""Profile Manager package — state-aware profile resolution."""
from src.profile_manager.resolver import (
    STATE_PROFILE_MAP,
    resolve_profile_from_state,
    runtime_flags_for_profile,
)

__all__ = [
    "STATE_PROFILE_MAP",
    "resolve_profile_from_state",
    "runtime_flags_for_profile",
]
