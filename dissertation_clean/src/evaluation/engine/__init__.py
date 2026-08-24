"""Engine — data generation & orchestration (Layer 1). The only torch-touching layer."""
from .rollout import TASKS, obs_layout, rollout
from .types import (
    BASELINE_COMPETENCE,
    BASELINE_HOLD,
    IDENTITY,
    Intervention,
    ObsLayout,
    RolloutResult,
    VariantSpec,
)

__all__ = [
    "BASELINE_COMPETENCE",
    "BASELINE_HOLD",
    "IDENTITY",
    "TASKS",
    "Intervention",
    "ObsLayout",
    "RolloutResult",
    "VariantSpec",
    "obs_layout",
    "rollout",
]